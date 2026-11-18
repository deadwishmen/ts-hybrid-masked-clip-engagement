"""
Training & Evaluation Pipeline with Mixed Precision (AMP) for Google Colab T4 GPU.
Strictly updates only LoRA (Q, V), Task Token, Coordinate MLP, Temporal Embeddings,
and Engagement Head via Cross-Entropy Loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import StudentEngagementTSHybridCLIP


def configure_optimizer(
    model: StudentEngagementTSHybridCLIP,
    lr_lora: float = 2e-4,
    lr_new_modules: float = 5e-4,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """
    Khởi tạo AdamW Optimizer chỉ cập nhật trọng số cho:
    LoRA (Q, V), Task Token, Coordinate MLP, Temporal Embeddings và Engagement Head.
    """
    groups = model.get_trainable_parameter_groups()
    optimizer_grouped_parameters = [
        {"params": groups["lora_qv"], "lr": lr_lora, "weight_decay": weight_decay},
        {"params": groups["task_token"], "lr": lr_new_modules, "weight_decay": 0.0},
        {"params": groups["temporal_embed"], "lr": lr_new_modules, "weight_decay": 0.0},
        {"params": groups["coord_mlp"], "lr": lr_new_modules, "weight_decay": weight_decay},
        {"params": groups["engagement_head"], "lr": lr_new_modules, "weight_decay": weight_decay},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters)


def train_one_step(
    model: StudentEngagementTSHybridCLIP,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> Tuple[float, torch.Tensor]:
    """Một bước huấn luyện (hỗ trợ FP16 Mixed Precision trên GPU T4 để tiết kiệm VRAM)."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    frames = batch["frames"].to(device, non_blocking=True)  # (B, T, 3, 224, 224)
    boxes = batch["boxes"].to(device, non_blocking=True)    # (B, T, 4)
    labels = batch["label"].to(device, non_blocking=True)   # (B,)

    use_amp = scaler is not None and device.type == "cuda"
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(frames=frames, boxes=boxes)
        logits = outputs["logits"]
        loss = criterion(logits, labels)

    if use_amp:
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
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 20,
) -> Dict[str, float]:
    """Huấn luyện 1 epoch trên tập Train."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, batch in enumerate(dataloader, start=1):
        loss_val, probs = train_one_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )
        labels = batch["label"].to(device)
        preds = probs.argmax(dim=-1)
        batch_size = labels.size(0)

        running_loss += loss_val * batch_size
        correct += int((preds == labels).sum().item())
        total += batch_size

        if step % log_interval == 0 or step == len(dataloader):
            print(
                f"  [Step {step:4d}/{len(dataloader):4d}] "
                f"Loss: {running_loss / max(total, 1):.4f} | "
                f"Acc: {100.0 * correct / max(total, 1):.2f}%"
            )

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: StudentEngagementTSHybridCLIP,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Đánh giá mô hình trên tập Valid hoặc Test."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        frames = batch["frames"].to(device, non_blocking=True)
        boxes = batch["boxes"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(frames=frames, boxes=boxes)
            logits = outputs["logits"]
            loss = criterion(logits, labels)

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
    """
    Chỉ lưu các trọng số Trainable (LoRA Q/V, Task Token, Coordinate MLP,
    P_temporal, Engagement Head) ra thư mục checkpoint riêng (không đụng tới thư mục dữ liệu).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
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
