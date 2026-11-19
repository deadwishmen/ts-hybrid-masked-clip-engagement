"""
Training & Evaluation Pipeline tối ưu hóa cho GPU NVIDIA A100:
  - TF32 Tensor Cores (`allow_tf32 = True`, `float32_matmul_precision = 'high'`)
  - Native BFloat16 (`torch.bfloat16`) autocast (không cần GradScaler, nhanh và ổn định hơn FP16)
  - Fused AdamW (`fused=True` trên CUDA)
  - Đo lường tốc độ huấn luyện (samples/sec)
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import StudentEngagementTSHybridCLIP


def enable_a100_speedups() -> torch.dtype:
    """
    Kích hoạt các tối ưu hóa phần cứng cho GPU NVIDIA A100 (Ampere):
      - TF32 cho MatMul và cuDNN
      - Tự động chọn BFloat16 nếu GPU hỗ trợ (A100/H100), ngược lại dùng FP16 (T4/V100).
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_amp_dtype(amp_mode: str = "auto") -> torch.dtype:
    if not torch.cuda.is_available() or amp_mode == "fp32":
        return torch.float32
    if amp_mode == "bf16":
        return torch.bfloat16
    if amp_mode == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def configure_optimizer(
    model: StudentEngagementTSHybridCLIP,
    lr_lora: float = 2e-4,
    lr_new_modules: float = 5e-4,
    weight_decay: float = 1e-2,
    use_fused: bool = True,
) -> torch.optim.Optimizer:
    groups = model.get_trainable_parameter_groups()
    optimizer_grouped_parameters = [
        {"params": groups["lora_qv"], "lr": lr_lora, "weight_decay": weight_decay},
        {"params": groups["task_token"], "lr": lr_new_modules, "weight_decay": 0.0},
        {"params": groups["temporal_embed"], "lr": lr_new_modules, "weight_decay": 0.0},
        {"params": groups["coord_mlp"], "lr": lr_new_modules, "weight_decay": weight_decay},
        {"params": groups["engagement_head"], "lr": lr_new_modules, "weight_decay": weight_decay},
    ]
    can_fuse = use_fused and torch.cuda.is_available()
    try:
        return torch.optim.AdamW(optimizer_grouped_parameters, fused=can_fuse)
    except TypeError:
        return torch.optim.AdamW(optimizer_grouped_parameters)


def train_one_step(
    model: StudentEngagementTSHybridCLIP,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> Tuple[float, torch.Tensor]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    frames = batch["frames"].to(device, non_blocking=True)
    boxes = batch["boxes"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    use_amp = (device.type == "cuda") and (amp_dtype in (torch.bfloat16, torch.float16))
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        outputs = model(frames=frames, boxes=boxes)
        loss = criterion(outputs["logits"], labels)

    if scaler is not None and amp_dtype == torch.float16:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_grad_norm
        )
        optimizer.step()

    return float(loss.item()), outputs["probs"].detach()


def train_epoch(
    model: StudentEngagementTSHybridCLIP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    scaler: Optional[torch.amp.GradScaler] = None,
    log_interval: int = 20,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t0 = time.perf_counter()

    for step, batch in enumerate(dataloader, start=1):
        loss_val, probs = train_one_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        labels = batch["label"].to(device, non_blocking=True)
        preds = probs.argmax(dim=-1)
        batch_size = labels.size(0)

        running_loss += loss_val * batch_size
        correct += int((preds == labels).sum().item())
        total += batch_size

        if step % log_interval == 0 or step == len(dataloader):
            elapsed = max(time.perf_counter() - t0, 1e-6)
            throughput = total / elapsed
            print(
                f"  [Step {step:4d}/{len(dataloader):4d}] "
                f"Loss: {running_loss / max(total, 1):.4f} | "
                f"Acc: {100.0 * correct / max(total, 1):.2f}% | "
                f"Speed: {throughput:.1f} samples/s"
            )

    elapsed = max(time.perf_counter() - t0, 1e-6)
    return {
        "loss": running_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
        "samples_per_sec": total / elapsed,
    }


@torch.no_grad()
def evaluate(
    model: StudentEngagementTSHybridCLIP,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = (device.type == "cuda") and (amp_dtype in (torch.bfloat16, torch.float16))
    for batch in dataloader:
        frames = batch["frames"].to(device, non_blocking=True)
        boxes = batch["boxes"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(frames=frames, boxes=boxes)
            loss = criterion(outputs["logits"], labels)

        preds = outputs["probs"].argmax(dim=-1)
        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        correct += int((preds == labels).sum().item())
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }


def save_peft_checkpoint(
    model: StudentEngagementTSHybridCLIP,
    save_path: Path,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Hỗ trợ cả model đã torch.compile() (_orig_mod)
    raw_model = getattr(model, "_orig_mod", model)
    trainable_state = {
        k: v.cpu()
        for k, v in raw_model.state_dict().items()
        if any(
            token in k
            for token in (
                "lora_A",
                "lora_B",
                "task_token",
                "coord_mlp",
                "p_temporal",
                "engagement_head",
            )
        )
    }
    torch.save(
        {
            "epoch": epoch,
            "trainable_state_dict": trainable_state,
            "metrics": metrics,
        },
        save_path,
    )
