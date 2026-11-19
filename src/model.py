"""
4-Module T-S Hybrid Masked CLIP Architecture (Tối ưu cho NVIDIA A100 GPU)
Dựa trên nghiên cứu Falcon Perception (arXiv:2603.27365 - https://arxiv.org/abs/2603.27365):
  - SDPA (Scaled Dot-Product Attention) tốc độ cao trên Tensor Cores A100.
  - Cache sẵn Additive Hybrid Attention Mask theo (L, dtype, device) để tránh cấp phát lặp.
  - Tương thích hoàn toàn với torch.bfloat16 (BF16) và torch.compile().
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPVisionModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class CoordinateMLP(nn.Module):
    """Mạng Coordinate MLP: Linear(4 -> D) -> GELU -> Linear(D -> D)."""

    def __init__(self, in_features: int = 4, embed_dim: int = 768) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        return self.mlp(boxes)


class InputEncodingModule(nn.Module):
    """Module 1 — Mã hóa Đầu vào: E_task (1, D), E_box (T, D), E_patch (T, N, D)."""

    def __init__(self, patch_embedding_conv: nn.Conv2d, embed_dim: int = 768) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.task_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.coord_mlp = CoordinateMLP(in_features=4, embed_dim=embed_dim)

        self.patch_embedding = patch_embedding_conv
        for param in self.patch_embedding.parameters():
            param.requires_grad = False

    def forward(
        self, frames: torch.Tensor, boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_frames, channels, height, width = frames.shape

        e_task = self.task_token.unsqueeze(0).expand(batch_size, -1, -1)
        e_box = self.coord_mlp(boxes)

        frames_flat = frames.reshape(batch_size * num_frames, channels, height, width)
        patch_embeds = self.patch_embedding(frames_flat)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        num_patches = patch_embeds.shape[1]
        e_patch = patch_embeds.reshape(batch_size, num_frames, num_patches, self.embed_dim)
        return e_task, e_box, e_patch


class SpatiotemporalInjectionModule(nn.Module):
    """Module 2 — Nhúng Không gian (P_spatial) & Thời gian (P_temporal) và Ghép chuỗi."""

    def __init__(
        self,
        pretrained_spatial_pos_embed: torch.Tensor,
        num_frames: int = 8,
        num_patches: int = 196,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        if pretrained_spatial_pos_embed.shape[0] == num_patches + 1:
            spatial_weight = pretrained_spatial_pos_embed[1:, :].detach().clone()
        elif pretrained_spatial_pos_embed.shape[0] == num_patches:
            spatial_weight = pretrained_spatial_pos_embed.detach().clone()
        else:
            raise ValueError(f"Kích thước P_spatial không khớp: {pretrained_spatial_pos_embed.shape}")

        self.register_buffer("p_spatial", spatial_weight)
        self.p_temporal = nn.Parameter(torch.randn(num_frames, embed_dim) * 0.02)

    def forward(
        self, e_task: torch.Tensor, e_box: torch.Tensor, e_patch: torch.Tensor
    ) -> torch.Tensor:
        batch_size, t_frames, n_patches, d_dim = e_patch.shape
        p_temp = self.p_temporal[:t_frames, :]

        e_box_injected = e_box + p_temp.unsqueeze(0)
        e_patch_injected = (
            e_patch
            + self.p_spatial.unsqueeze(0).unsqueeze(0)
            + p_temp.unsqueeze(0).unsqueeze(2)
        )
        e_patch_flat = e_patch_injected.reshape(batch_size, t_frames * n_patches, d_dim)
        return torch.cat([e_task, e_box_injected, e_patch_flat], dim=1)


class LoRALinear(nn.Module):
    """Adapter LoRA cho các ma trận Query (Q) và Value (V) tương thích với BF16 trên A100."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out


def build_ts_hybrid_attention_mask(
    num_frames: int = 8,
    num_patches: int = 196,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Xây dựng ma trận T-S Hybrid Attention Mask kích thước (L, L) với L = 1 + T + T*N."""
    seq_len = 1 + num_frames + (num_frames * num_patches)
    allowed = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    # 1. Task Token (0) -> nhìn chính nó (0) và toàn bộ Box Tokens (1..T)
    allowed[0, 0 : 1 + num_frames] = True
    # 2. Box Tokens (1..T) -> nhìn toàn bộ Box Tokens (1..T) xuyên thời gian
    allowed[1 : 1 + num_frames, 1 : 1 + num_frames] = True
    # 3. Box Token t <-> Patches frame t, và Patches frame t <-> nhau
    for t in range(num_frames):
        box_idx = 1 + t
        patch_start = 1 + num_frames + t * num_patches
        patch_end = patch_start + num_patches
        allowed[box_idx, patch_start:patch_end] = True
        allowed[patch_start:patch_end, box_idx] = True
        allowed[patch_start:patch_end, patch_start:patch_end] = True
    return allowed


def boolean_to_additive_attention_mask(
    allowed_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    additive = torch.zeros_like(allowed_mask, dtype=dtype)
    additive = additive.masked_fill(~allowed_mask, torch.finfo(dtype).min)
    return additive.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


class NativeCLIPViTBlock(nn.Module):
    """Block Transformer sử dụng PyTorch SDPA tối ưu hóa trên Tensor Cores A100."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    @staticmethod
    def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        additive_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        x = self.layer_norm1(hidden_states)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=additive_attn_mask, dropout_p=0.0
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        hidden_states = residual + self.out_proj(attn_out)

        residual = hidden_states
        x = self.layer_norm2(hidden_states)
        x = self.fc2(self._quick_gelu(self.fc1(x)))
        return residual + x


class TSHybridMaskedTransformer(nn.Module):
    """Module 3 — T-S Hybrid Masked Transformer với LoRA (Q, V) & Cache Mask cho A100."""

    def __init__(
        self,
        pre_layernorm: nn.LayerNorm,
        encoder_layers: nn.ModuleList,
        use_hf_layers: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.pre_layernorm = pre_layernorm
        self.layers = encoder_layers
        self.use_hf_layers = use_hf_layers

        for param in self.pre_layernorm.parameters():
            param.requires_grad = False
        for param in self.layers.parameters():
            param.requires_grad = False

        self._inject_lora_into_qv(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    def _inject_lora_into_qv(self, r: int, alpha: float, dropout: float) -> None:
        for layer in self.layers:
            if self.use_hf_layers and hasattr(layer, "self_attn"):
                layer.self_attn.q_proj = LoRALinear(
                    layer.self_attn.q_proj, r=r, alpha=alpha, dropout=dropout
                )
                layer.self_attn.v_proj = LoRALinear(
                    layer.self_attn.v_proj, r=r, alpha=alpha, dropout=dropout
                )
            else:
                layer.q_proj = LoRALinear(layer.q_proj, r=r, alpha=alpha, dropout=dropout)
                layer.v_proj = LoRALinear(layer.v_proj, r=r, alpha=alpha, dropout=dropout)

    def forward(self, sequence: torch.Tensor, additive_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_layernorm(sequence)
        if additive_mask.dtype != hidden_states.dtype:
            additive_mask = additive_mask.to(dtype=hidden_states.dtype)

        if self.use_hf_layers:
            batch_size = hidden_states.shape[0]
            hf_mask = additive_mask.expand(batch_size, -1, -1, -1)
            for layer in self.layers:
                layer_out = layer(
                    hidden_states,
                    attention_mask=None,
                    causal_attention_mask=hf_mask,
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states, additive_attn_mask=additive_mask)

        return hidden_states


class EngagementHead(nn.Module):
    """Module 4 — Engagement Head: LayerNorm -> Linear -> GELU -> Dropout -> Linear."""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 384,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sequence_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        task_token_final = sequence_output[:, 0, :]
        logits = self.mlp_head(task_token_final)
        probs = F.softmax(logits.float(), dim=-1)
        return logits, probs


class StudentEngagementTSHybridCLIP(nn.Module):
    """Kiến trúc tổng hợp 4 Module tối ưu hóa cho A100 GPU."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        num_frames: int = 8,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_classes: int = 4,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        head_dropout: float = 0.2,
        load_pretrained_clip: bool = True,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        patch_conv, spatial_pos_embed, pre_ln, encoder_layers, use_hf = (
            self._build_or_load_clip_backbone(
                clip_model_name=clip_model_name,
                embed_dim=embed_dim,
                patch_size=patch_size,
                num_patches=self.num_patches,
                num_layers=num_layers,
                num_heads=num_heads,
                load_pretrained=load_pretrained_clip,
            )
        )

        self.input_encoder = InputEncodingModule(patch_embedding_conv=patch_conv, embed_dim=embed_dim)
        self.spatiotemporal_injector = SpatiotemporalInjectionModule(
            pretrained_spatial_pos_embed=spatial_pos_embed,
            num_frames=num_frames,
            num_patches=self.num_patches,
            embed_dim=embed_dim,
        )
        self.transformer = TSHybridMaskedTransformer(
            pre_layernorm=pre_ln,
            encoder_layers=encoder_layers,
            use_hf_layers=use_hf,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        base_mask = build_ts_hybrid_attention_mask(num_frames=num_frames, num_patches=self.num_patches)
        self.register_buffer("hybrid_attention_mask", base_mask)
        self.register_buffer("additive_mask_fp32", boolean_to_additive_attention_mask(base_mask, torch.float32))
        self.register_buffer("additive_mask_fp16", boolean_to_additive_attention_mask(base_mask, torch.float16))
        self.register_buffer("additive_mask_bf16", boolean_to_additive_attention_mask(base_mask, torch.bfloat16))
        self.engagement_head = EngagementHead(
            embed_dim=embed_dim,
            hidden_dim=embed_dim // 2,
            num_classes=num_classes,
            dropout=head_dropout,
        )

    @staticmethod
    def _build_or_load_clip_backbone(
        clip_model_name: str,
        embed_dim: int,
        patch_size: int,
        num_patches: int,
        num_layers: int,
        num_heads: int,
        load_pretrained: bool,
    ) -> Tuple[nn.Conv2d, torch.Tensor, nn.LayerNorm, nn.ModuleList, bool]:
        if HAS_TRANSFORMERS and load_pretrained:
            try:
                # Ưu tiên SDPA trên A100 để tăng tốc Attention
                try:
                    clip_vision = CLIPVisionModel.from_pretrained(
                        clip_model_name, attn_implementation="sdpa"
                    )
                except TypeError:
                    clip_vision = CLIPVisionModel.from_pretrained(clip_model_name)

                vm = clip_vision.vision_model
                return (
                    vm.embeddings.patch_embedding,
                    vm.embeddings.position_embedding.weight,
                    vm.pre_layrnorm,
                    vm.encoder.layers,
                    True,
                )
            except Exception as exc:
                print(
                    f"[Thông báo] Không thể tải '{clip_model_name}' ({exc}). "
                    f"Sử dụng backbone CLIP ViT-B/16 nội bộ (SDPA)."
                )

        patch_conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        spatial_pos = torch.randn(num_patches + 1, embed_dim) * 0.02
        pre_ln = nn.LayerNorm(embed_dim, eps=1e-5)
        encoder_layers = nn.ModuleList([
            NativeCLIPViTBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        return patch_conv, spatial_pos, pre_ln, encoder_layers, False

    def forward(self, frames: torch.Tensor, boxes: torch.Tensor) -> Dict[str, torch.Tensor]:
        t_frames = frames.shape[1]
        e_task, e_box, e_patch = self.input_encoder(frames, boxes)
        sequence = self.spatiotemporal_injector(e_task, e_box, e_patch)

        if t_frames == self.num_frames:
            if sequence.dtype == torch.bfloat16:
                additive_mask = self.additive_mask_bf16
            elif sequence.dtype == torch.float16:
                additive_mask = self.additive_mask_fp16
            else:
                additive_mask = self.additive_mask_fp32
        else:
            dyn_mask = build_ts_hybrid_attention_mask(t_frames, self.num_patches, sequence.device)
            additive_mask = boolean_to_additive_attention_mask(dyn_mask, dtype=sequence.dtype)
        encoded_sequence = self.transformer(sequence, additive_mask)
        logits, probs = self.engagement_head(encoded_sequence)
        return {
            "logits": logits,
            "probs": probs,
            "task_embedding": encoded_sequence[:, 0, :],
        }

    def get_trainable_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        lora_params = [
            param
            for name, param in self.transformer.named_parameters()
            if param.requires_grad and ("lora_A" in name or "lora_B" in name)
        ]
        return {
            "lora_qv": lora_params,
            "task_token": [self.input_encoder.task_token],
            "coord_mlp": list(self.input_encoder.coord_mlp.parameters()),
            "temporal_embed": [self.spatiotemporal_injector.p_temporal],
            "engagement_head": list(self.engagement_head.parameters()),
        }

    def print_parameter_summary(self) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ratio = 100.0 * trainable_params / max(total_params, 1)

        print("=" * 72)
        print("THỐNG KÊ THAM SỐ MÔ HÌNH (T-S HYBRID MASKED CLIP + LoRA)")
        print("=" * 72)
        print(f"Tổng số tham số (Total Parameters)     : {total_params:,}")
        print(f"Tham số huấn luyện (Trainable Params)  : {trainable_params:,} ({ratio:.2f}%)")
        print(f"Tham số đóng băng (Frozen CLIP Params) : {total_params - trainable_params:,}")
        print("-" * 72)
        for group_name, params in self.get_trainable_parameter_groups().items():
            count = sum(p.numel() for p in params)
            print(f"  + {group_name:<20}: {count:>10,} params")
        print("=" * 72)
        return total_params, trainable_params
