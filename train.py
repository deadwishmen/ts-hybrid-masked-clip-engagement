"""
CLI script để huấn luyện hoặc kiểm thử T-S Hybrid Masked CLIP trên Google Colab / Server.
Tuyệt đối chỉ đọc (Read-Only) dữ liệu từ thư mục ViMELD_Dataset trên Google Drive.

Ví dụ chạy kiểm thử nhanh (Dry-run verification):
    python train.py --verify_only

Ví dụ huấn luyện với dữ liệu từ Google Drive:
    python train.py \
        --data_root "/content/drive/MyDrive/ViMELD_Dataset" \
        --manifest_path "/content/drive/MyDrive/Restructured_Dataset_NPY_Engagement/manifest.parquet" \
        --output_dir "/content/checkpoints_ts_hybrid_clip" \
        --num_frames 8 \
        --num_classes 3 \
        --batch_size 8 \
        --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import ViMELDEngagementDataset
from src.model import StudentEngagementTSHybridCLIP
from src.trainer import configure_optimizer, evaluate, save_peft_checkpoint, train_epoch
from ts_hybrid_masked_clip_engagement import verify_architecture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train T-S Hybrid Masked CLIP for Student Engagement Recognition (Read-Only Data)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/content/drive/MyDrive/ViMELD_Dataset",
        help="Đường dẫn thư mục ViMELD_Dataset trên Google Drive (chỉ đọc).",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="",
        help="Tùy chọn: Đường dẫn file manifest.parquet hoặc manifest.csv (chỉ đọc).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_ts_hybrid_clip",
        help="Thư mục cục bộ lưu checkpoint LoRA (tách biệt hoàn toàn với thư mục dữ liệu).",
    )
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--num_frames", type=int, default=8, help="Số frame T mỗi clip (mặc định T=8 -> L=1577).")
    parser.add_argument("--num_classes", type=int, default=3, choices=[3, 4], help="3 lớp (Low/Medium/High) hoặc 4 lớp gốc.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_lora", type=float, default=2e-4)
    parser.add_argument("--lr_new", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--use_cropped_segments",
        action="store_true",
        help="Nếu bật, đọc trực tiếp 16 frame PNG trong thư mục seg_* thay vì đọc video toàn cảnh .mp4.",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Chỉ chạy kiểm thử kiến trúc 4 module, ma trận mask (1577x1577) và backward pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        verify_architecture()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Thiết bị: {device}")

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

    print(f"[Dataset Read-Only] Số mẫu Train: {len(train_dataset):,} | Valid: {len(valid_dataset):,}")
    if len(train_dataset) == 0:
        print(
            "[Cảnh báo] Không tìm thấy mẫu dữ liệu tại đường dẫn đã cho. "
            "Hãy kiểm tra lại đường dẫn mount Google Drive hoặc chạy với --verify_only."
        )
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = StudentEngagementTSHybridCLIP(
        clip_model_name=args.clip_model,
        num_frames=args.num_frames,
        num_classes=args.num_classes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_pretrained_clip=True,
    ).to(device)
    model.print_parameter_summary()

    optimizer = configure_optimizer(
        model,
        lr_lora=args.lr_lora,
        lr_new_modules=args.lr_new,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    out_dir = Path(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )
        print(
            f"[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}%"
        )

        if len(valid_dataset) > 0:
            val_metrics = evaluate(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                device=device,
            )
            print(
                f"[Epoch {epoch}] Valid Loss: {val_metrics['loss']:.4f} | "
                f"Valid Acc: {val_metrics['accuracy']:.2f}%"
            )
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                ckpt_path = out_dir / "best_ts_hybrid_clip_peft.pt"
                save_peft_checkpoint(model, ckpt_path, epoch, val_metrics)
                print(f"  -> Đã lưu PEFT checkpoint tốt nhất tại: {ckpt_path}")
        else:
            ckpt_path = out_dir / f"ts_hybrid_clip_epoch_{epoch}.pt"
            save_peft_checkpoint(model, ckpt_path, epoch, train_metrics)


if __name__ == "__main__":
    main()
