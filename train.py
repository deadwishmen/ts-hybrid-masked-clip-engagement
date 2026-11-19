"""
CLI script tối ưu hóa cho NVIDIA A100 GPU (BF16 + TF32 + SDPA + Fused AdamW + torch.compile).
Tuyệt đối chỉ đọc (Read-Only) dữ liệu từ ViMELD_Dataset.

Ví dụ chạy benchmark tốc độ trên A100:
    python train.py --verify_only --benchmark --batch_size 32

Ví dụ huấn luyện trên A100:
    python train.py \
        --data_root "/content/local_data_cache/ViMELD_Dataset" \
        --batch_size 32 \
        --num_workers 4 \
        --amp_dtype bf16 \
        --compile \
        --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import ViMELDEngagementDataset
from src.model import StudentEngagementTSHybridCLIP
from src.trainer import (
    configure_optimizer,
    enable_a100_speedups,
    evaluate,
    resolve_amp_dtype,
    save_peft_checkpoint,
    train_epoch,
    train_one_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train T-S Hybrid Masked CLIP on A100 GPU (Read-Only ViMELD Data)"
    )
    parser.add_argument("--data_root", type=str, default="/content/drive/MyDrive/ViMELD_Dataset")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_ts_hybrid_clip")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=3, choices=[3, 4])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--batch_size", type=int, default=32, help="A100 40GB/80GB khuyến nghị batch_size=32 hoặc 64.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_lora", type=float, default=2e-4)
    parser.add_argument("--lr_new", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true", help="Bật torch.compile(mode='reduce-overhead') cho A100.")
    parser.add_argument("--use_cropped_segments", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Đo tốc độ samples/sec và VRAM tiêu thụ.")
    return parser.parse_args()


def run_verify_and_benchmark(args: argparse.Namespace) -> None:
    enable_a100_speedups()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    print(f"[Verify/Benchmark] Device: {device} | AMP dtype: {amp_dtype}")

    model = StudentEngagementTSHybridCLIP(
        num_frames=args.num_frames,
        num_classes=args.num_classes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_pretrained_clip=False,
    ).to(device)
    model.print_parameter_summary()

    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        print("[Compile] Đang biên dịch mô hình với torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = configure_optimizer(model, lr_lora=args.lr_lora, lr_new_modules=args.lr_new)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    bs = args.batch_size if device.type == "cuda" else 2
    dummy_frames = torch.randn(bs, args.num_frames, 3, 224, 224, device=device)
    dummy_boxes = torch.rand(bs, args.num_frames, 4, device=device)
    dummy_labels = torch.randint(0, args.num_classes, (bs,), device=device)
    batch = {"frames": dummy_frames, "boxes": dummy_boxes, "label": dummy_labels}

    # Warmup
    for _ in range(3 if args.benchmark else 1):
        loss_val, probs = train_one_step(
            model, batch, optimizer, criterion, device, amp_dtype=amp_dtype, scaler=scaler
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    if args.benchmark:
        iters = 20
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            loss_val, _ = train_one_step(
                model, batch, optimizer, criterion, device, amp_dtype=amp_dtype, scaler=scaler
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        samples_per_sec = (iters * bs) / elapsed
        vram_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )
        print(
            f"[Benchmark Result] Batch={bs} | {samples_per_sec:.1f} samples/sec | "
            f"Peak VRAM: {vram_gb:.2f} GB | Last Loss: {loss_val:.4f}"
        )
    else:
        print(f"[Verify OK] Forward/Backward hoàn tất | Probs shape: {tuple(probs.shape)} | Loss: {loss_val:.4f}")


def main() -> None:
    args = parse_args()
    if args.verify_only:
        run_verify_and_benchmark(args)
        return

    enable_a100_speedups()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    print(f"[A100 Train] Device: {device} | AMP dtype: {amp_dtype} | Batch Size: {args.batch_size}")

    manifest = args.manifest_path if args.manifest_path else None
    train_dataset = ViMELDEngagementDataset(
        root_dir=args.data_root,
        split="train",
        num_frames=args.num_frames,
        num_classes=args.num_classes,
        manifest_path=manifest,
        use_full_scene_video=not args.use_cropped_segments,
    )
    valid_dataset = ViMELDEngagementDataset(
        root_dir=args.data_root,
        split="valid",
        num_frames=args.num_frames,
        num_classes=args.num_classes,
        manifest_path=manifest,
        use_full_scene_video=not args.use_cropped_segments,
    )

    print(f"[Dataset Read-Only] Train: {len(train_dataset):,} | Valid: {len(valid_dataset):,}")
    if len(train_dataset) == 0:
        print("[Cảnh báo] Không tìm thấy dữ liệu. Kiểm tra đường dẫn hoặc chạy --verify_only --benchmark.")
        return

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

    model = StudentEngagementTSHybridCLIP(
        clip_model_name=args.clip_model,
        num_frames=args.num_frames,
        num_classes=args.num_classes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_pretrained_clip=True,
    ).to(device)
    model.print_parameter_summary()

    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        print("[Compile] Đang biên dịch mô hình với torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = configure_optimizer(model, lr_lora=args.lr_lora, lr_new_modules=args.lr_new)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    best_val_acc = -1.0
    out_dir = Path(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        tr_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, amp_dtype=amp_dtype, scaler=scaler
        )
        print(
            f"[Epoch {epoch}] Train Loss: {tr_metrics['loss']:.4f} | "
            f"Acc: {tr_metrics['accuracy']:.2f}% | "
            f"Throughput: {tr_metrics['samples_per_sec']:.1f} samples/s"
        )

        if len(valid_dataset) > 0:
            va_metrics = evaluate(model, valid_loader, criterion, device, amp_dtype=amp_dtype)
            print(f"[Epoch {epoch}] Valid Loss: {va_metrics['loss']:.4f} | Acc: {va_metrics['accuracy']:.2f}%")
            if va_metrics["accuracy"] > best_val_acc:
                best_val_acc = va_metrics["accuracy"]
                ckpt_path = out_dir / "best_ts_hybrid_clip_peft.pt"
                save_peft_checkpoint(model, ckpt_path, epoch, va_metrics)
                print(f"  -> Đã lưu PEFT checkpoint tốt nhất: {ckpt_path}")


if __name__ == "__main__":
    main()
