"""
4-Module T-S Hybrid Masked CLIP Architecture for Student Engagement Recognition
Inspired by Falcon Perception (arXiv:2603.27365 - https://arxiv.org/abs/2603.27365).
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


# ==============================================================================
# 2. MODULE 1: MÃ HÓA ĐẦU VÀO (INPUT ENCODING MODULE)
# ==============================================================================

class CoordinateMLP(nn.Module):
    """
    Mạng Coordinate MLP: Linear(4 -> D) -> GELU -> Linear(D -> D)
    Nâng tọa độ Bounding Box [x_min, y_min, x_max, y_max] đã chuẩn hóa từ (T, 4) lên (T, D).
    Lấy cảm hứng từ Coordinate & Size Encoder trong Falcon Perception (arXiv:2603.27365, Section 2.2).
    """

    def __init__(self, in_features: int = 4, embed_dim: int = 768) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes: Tensor kích thước (B, T, 4) chứa [x_min, y_min, x_max, y_max] đã chuẩn hóa.
        Returns:
            E_box: Tensor kích thước (B, T, D).
        """
        return self.mlp(boxes)


class InputEncodingModule(nn.Module):
    """
    Module 1 — Mã hóa Đầu vào:
    Quy 3 thành phần đầu vào của 1 sample (1 học sinh trong 1 clip dài T frames)
    về cùng không gian nhúng có số chiều D = 768 (CLIP ViT-B/16):
      1. Task Token (E_task): Learnable Vector (1, D) -> mở rộng theo batch thành (B, 1, D).
      2. Box Trajectory (E_box): (B, T, 4) qua CoordinateMLP -> (B, T, D).
      3. Image Patches (E_patch): (B, T, C, H, W) qua Conv2D Patch Embedding của CLIP ViT
         (kernel=16x16, stride=16x16) -> (B, T, N, D) với N = (224/16)^2 = 196 patches/frame.
    """

    def __init__(
        self,
        patch_embedding_conv: nn.Conv2d,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Task Token (E_task): Learnable Vector khởi tạo ngẫu nhiên kích thước (1, D)
        self.task_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # 2. Coordinate MLP cho Box Trajectory (E_box)
        self.coord_mlp = CoordinateMLP(in_features=4, embed_dim=embed_dim)

        # 3. Lớp Conv2D Patch Embedding mượn từ CLIP ViT-B/16 (đóng băng theo chiến lược PEFT)
        self.patch_embedding = patch_embedding_conv
        for param in self.patch_embedding.parameters():
            param.requires_grad = False

    def forward(
        self,
        frames: torch.Tensor,
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: (B, T, C, H, W) với H=W=224, C=3.
            boxes:  (B, T, 4) chứa [x_min, y_min, x_max, y_max] đã chuẩn hóa.
        Returns:
            e_task:  (B, 1, D)
            e_box:   (B, T, D)
            e_patch: (B, T, N, D) với N=196, D=768.
        """
        batch_size, num_frames, channels, height, width = frames.shape

        # 1. Task Token: (1, D) -> (B, 1, D)
        e_task = self.task_token.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Box Trajectory: (B, T, 4) -> (B, T, D)
        e_box = self.coord_mlp(boxes)

        # 3. Image Patches: gộp (B*T, C, H, W) chạy qua Conv2D của CLIP ViT
        frames_flat = frames.reshape(batch_size * num_frames, channels, height, width)
        # patch_embeds: (B*T, D, H/16, W/16) = (B*T, 768, 14, 14)
        patch_embeds = self.patch_embedding(frames_flat)
        # Trải phẳng không gian 14x14 -> N=196: (B*T, D, 196) -> (B*T, 196, D)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        num_patches = patch_embeds.shape[1]
        # Khôi phục chiều thời gian T: (B, T, N, D)
        e_patch = patch_embeds.reshape(batch_size, num_frames, num_patches, self.embed_dim)

        return e_task, e_box, e_patch


# ==============================================================================
# 3. MODULE 2: NHÚNG KHÔNG GIAN & THỜI GIAN (SPATIOTEMPORAL INJECTION)
# ==============================================================================

class SpatiotemporalInjectionModule(nn.Module):
    """
    Module 2 — Nhúng Không gian & Thời gian và Ghép chuỗi (Sequence Concatenation):
      - Spatial Positional Embedding (P_spatial): Mượn trực tiếp từ CLIP ViT, kích thước (N, D) = (196, 768).
        Cộng P_spatial vào các Image Patches ở từng frame.
      - Temporal Positional Embedding (P_temporal): Learnable Parameter kích thước (T, D).
        + Cộng P_temporal[t] vào Box Token thứ t.
        + Cộng P_temporal[t] vào tất cả N Image Patches thuộc frame thứ t.
      - Ghép chuỗi (Sequence Concatenation):
        Sequence = [E_task] (+) E_box (+) E_patch
        Chiều dài chuỗi: L = 1 + T + (T * N). Với T=8, N=196 -> L = 1 + 8 + 1568 = 1577 tokens.
    """

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

        # P_spatial: (N, D) mượn trực tiếp từ CLIP ViT (bỏ vị trí 0 của class token gốc trong CLIP)
        if pretrained_spatial_pos_embed.shape[0] == num_patches + 1:
            spatial_weight = pretrained_spatial_pos_embed[1:, :].detach().clone()
        elif pretrained_spatial_pos_embed.shape[0] == num_patches:
            spatial_weight = pretrained_spatial_pos_embed.detach().clone()
        else:
            raise ValueError(
                f"Kích thước P_spatial không khớp: nhận {pretrained_spatial_pos_embed.shape}, "
                f"kỳ vọng ({num_patches}, {embed_dim}) hoặc ({num_patches + 1}, {embed_dim})."
            )
        # Đóng băng P_spatial của CLIP backbone theo chiến lược PEFT
        self.register_buffer("p_spatial", spatial_weight)

        # P_temporal: Learnable Parameter kích thước (T, D) được cập nhật qua Cross-Entropy Loss
        self.p_temporal = nn.Parameter(torch.randn(num_frames, embed_dim) * 0.02)

    def forward(
        self,
        e_task: torch.Tensor,
        e_box: torch.Tensor,
        e_patch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            e_task:  (B, 1, D)
            e_box:   (B, T, D)
            e_patch: (B, T, N, D)
        Returns:
            sequence: (B, L, D) với L = 1 + T + T*N.
        """
        batch_size, t_frames, n_patches, d_dim = e_patch.shape
        if t_frames > self.num_frames:
            raise ValueError(
                f"Số frame đầu vào T={t_frames} vượt quá kích thước P_temporal={self.num_frames}."
            )

        p_temp = self.p_temporal[:t_frames, :]  # (T, D)

        # 1. Cộng P_temporal[t] vào Box Token thứ t: (B, T, D) + (1, T, D)
        e_box_injected = e_box + p_temp.unsqueeze(0)

        # 2. Cộng P_spatial (1, 1, N, D) và P_temporal[t] (1, T, 1, D) vào Image Patches (B, T, N, D)
        e_patch_injected = (
            e_patch
            + self.p_spatial.unsqueeze(0).unsqueeze(0)
            + p_temp.unsqueeze(0).unsqueeze(2)
        )

        # 3. Trải phẳng (flatten) các patch từ (B, T, N, D) -> (B, T*N, D)
        e_patch_flat = e_patch_injected.reshape(batch_size, t_frames * n_patches, d_dim)

        # 4. Nối 3 thành phần thành một chuỗi duy nhất: [E_task] (+) E_box (+) E_patch
        sequence = torch.cat([e_task, e_box_injected, e_patch_flat], dim=1)
        return sequence


# ==============================================================================
# 4. MODULE 3: T-S HYBRID MASKED TRANSFORMER & CHIẾN LƯỢC LoRA
# ==============================================================================

class LoRALinear(nn.Module):
    """
    Adapter LoRA (Low-Rank Adaptation) bơm vào ma trận Query (Q) và Value (V)
    của các lớp Transformer trong CLIP ViT-B/16.
    Trọng số gốc W_0 bị đóng băng hoàn toàn; chỉ huấn luyện hai ma trận hạng thấp A và B:
        h = W_0 x + (alpha / r) * B(A(x))
    """

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

        # Khởi tạo Kaiming cho A và 0 cho B để ΔW = 0 tại bước đầu tiên
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
    """
    Xây dựng ma trận T-S Hybrid Attention Mask kích thước (L, L) với L = 1 + T + T*N.
    Quy ước cấu trúc chuỗi:
      - Index 0: Task Token
      - Index 1 .. T: Box Tokens của frame t in [0, T-1] (index = 1 + t)
      - Index 1 + T + t*N .. 1 + T + (t+1)*N - 1: N Image Patches thuộc frame t

    Quy tắc kiểm soát "ai được nhìn thấy ai" (XAI - Explainable AI):
      1. Task Token (idx 0):
         Được quyền nhìn chính nó (idx 0) & toàn bộ chuỗi Box Tokens (idx 1..T).
         Lý do XAI: Hút thông tin quỹ đạo hành động (ví dụ: gục đầu, giơ tay) để ra quyết định.
      2. Box Token thứ t (idx 1 + t):
         Được quyền nhìn tất cả Box Tokens xuyên thời gian T (idx 1..T, bao gồm chính nó)
         & tất cả các Image Patches cùng frame t.
         Lý do XAI: Hiểu bối cảnh xung quanh học sinh đó ở frame t, đồng thời liên kết
         sự thay đổi tư thế theo thời gian.
      3. Patch (t, i) (thuộc frame t):
         Chỉ được quyền nhìn Box Token cùng frame t (idx 1 + t) & các Patches khác cùng frame t.
         Lý do XAI: Tạo ngữ cảnh xã hội (social context) của lớp học tại một thời điểm,
         giới hạn không nhìn xuyên frame để tránh nhiễu và OOM.

    Returns:
        allowed_mask: Tensor kiểu bool kích thước (L, L), trong đó True = được phép nhìn (attend),
                      False = bị khóa (masked out).
    """
    seq_len = 1 + num_frames + (num_frames * num_patches)
    allowed = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    # 1. Task Token (index 0) -> nhìn chính nó (0) và toàn bộ Box Tokens (1 .. T)
    allowed[0, 0 : 1 + num_frames] = True

    # 2. Box Tokens (index 1 .. T) -> nhìn toàn bộ Box Tokens (1 .. T) xuyên thời gian
    allowed[1 : 1 + num_frames, 1 : 1 + num_frames] = True

    # 3. Liên kết nội bộ từng frame t (Box Token t <-> Patches cùng frame t, và Patches cùng frame t <-> nhau)
    for t in range(num_frames):
        box_idx = 1 + t
        patch_start = 1 + num_frames + t * num_patches
        patch_end = patch_start + num_patches

        # Box Token (t) nhìn các Patches cùng frame (t)
        allowed[box_idx, patch_start:patch_end] = True

        # Patch (t, i) nhìn Box Token cùng frame (t)
        allowed[patch_start:patch_end, box_idx] = True

        # Patch (t, i) nhìn các Patches khác cùng frame (t)
        allowed[patch_start:patch_end, patch_start:patch_end] = True

    return allowed


def boolean_to_additive_attention_mask(
    allowed_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Chuyển đổi boolean mask (L, L) sang additive attention mask (1, 1, L, L)
    để cộng trực tiếp vào ma trận attention scores (Q @ K^T / sqrt(d_k)):
      - Vị trí True  -> 0.0
      - Vị trí False -> -inf (torch.finfo(dtype).min)
    """
    additive = torch.zeros_like(allowed_mask, dtype=dtype)
    additive = additive.masked_fill(~allowed_mask, torch.finfo(dtype).min)
    return additive.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


class NativeCLIPViTBlock(nn.Module):
    """
    Block Transformer tương thích 1-1 với kiến trúc CLIP ViT-B/16 (D=768, 12 heads, QuickGELU/GELU).
    Dùng khi chạy offline hoặc khi khởi tạo từ cấu hình chuẩn của CLIP ViT-B/16.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

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
        """
        Args:
            hidden_states: (B, L, D)
            additive_attn_mask: (1, 1, L, L) với 0.0 cho vị trí hợp lệ và -inf cho vị trí bị khóa.
        """
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        x = self.layer_norm1(hidden_states)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Sử dụng PyTorch Scaled Dot-Product Attention (FlashAttention / Memory-Efficient kernel trên T4)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive_attn_mask,
            dropout_p=0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        hidden_states = residual + self.out_proj(attn_out)

        residual = hidden_states
        x = self.layer_norm2(hidden_states)
        x = self.fc2(self._quick_gelu(self.fc1(x)))
        hidden_states = residual + x
        return hidden_states


class TSHybridMaskedTransformer(nn.Module):
    """
    Module 3 — T-S Hybrid Masked Transformer với chiến lược LoRA (PEFT):
      - Đóng băng toàn bộ backbone CLIP ViT-B/16 (pre_layrnorm, 12 Transformer layers, post_layernorm).
      - Chỉ bơm adapter LoRA vào các ma trận Query (q_proj) và Value (v_proj) của tất cả K=12 lớp.
      - Áp dụng ma trận T-S Hybrid Attention Mask kích thước (L, L) trong suốt quá trình lan truyền.
    """

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

        # 1. Đóng băng toàn bộ tham số gốc của CLIP ViT backbone
        for param in self.pre_layernorm.parameters():
            param.requires_grad = False
        for param in self.layers.parameters():
            param.requires_grad = False

        # 2. Bơm LoRA vào đúng các ma trận Query (Q) và Value (V) của K=12 lớp Transformer
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

    def forward(
        self,
        sequence: torch.Tensor,
        allowed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sequence:     (B, L, D) chuỗi 1577 tokens từ Module 2.
            allowed_mask: (L, L) ma trận boolean T-S Hybrid Attention Mask.
        Returns:
            hidden_states: (B, L, D) đầu ra sau K=12 lớp Transformer.
        """
        hidden_states = self.pre_layernorm(sequence)
        additive_mask = boolean_to_additive_attention_mask(
            allowed_mask, dtype=hidden_states.dtype
        )

        if self.use_hf_layers:
            # HuggingFace CLIPEncoderLayer nhận causal_attention_mask: (B, 1, L, L)
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


# ==============================================================================
# 5. MODULE 4: MODULE PHÂN LOẠI (ENGAGEMENT HEAD)
# ==============================================================================

class EngagementHead(nn.Module):
    """
    Module 4 — Module Phân loại (Engagement Head):
    Sau khi đi qua K=12 lớp Transformer Blocks, chỉ trích xuất vector đầu ra
    của Task Token (index 0) ở lớp cuối cùng và đưa qua mạng MLP nhỏ:
        LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Logits / Softmax
    Đầu ra là logits và xác suất của 3 lớp (Low, Medium, High Engagement).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 384,
        num_classes: int = 3,
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
        """
        Args:
            sequence_output: (B, L, D) đầu ra từ lớp cuối cùng của T-S Hybrid Masked Transformer.
        Returns:
            logits: (B, num_classes) dùng trực tiếp cho nn.CrossEntropyLoss().
            probs:  (B, num_classes) xác suất Softmax của (Low, Medium, High Engagement).
        """
        # Chỉ lấy vector đầu ra của Task Token ở vị trí index 0: (B, D)
        task_token_final = sequence_output[:, 0, :]
        logits = self.mlp_head(task_token_final)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


# ==============================================================================
# 6. KIẾN TRÚC TỔNG HỢP: T-S HYBRID MASKED CLIP FOR ENGAGEMENT RECOGNITION
# ==============================================================================

class StudentEngagementTSHybridCLIP(nn.Module):
    """
    Kiến trúc tổng hợp kết nối 4 Module:
      1. InputEncodingModule        (E_task, E_box qua CoordinateMLP, E_patch qua CLIP Conv2D)
      2. SpatiotemporalInjection    (P_spatial từ CLIP + P_temporal học được -> Sequence L=1577)
      3. TSHybridMaskedTransformer  (12 lớp CLIP ViT-B/16 với Hybrid Mask (L, L) & LoRA trên Q, V)
      4. EngagementHead             (Task Token [:, 0, :] -> LayerNorm -> Linear -> GELU -> Dropout -> Linear)
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        num_frames: int = 8,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_classes: int = 3,
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
        self.num_patches = (image_size // patch_size) ** 2  # 196 patches
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Khởi tạo hoặc nạp trọng số CLIP ViT-B/16
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

        # Module 1: Mã hóa Đầu vào
        self.input_encoder = InputEncodingModule(
            patch_embedding_conv=patch_conv,
            embed_dim=embed_dim,
        )

        # Module 2: Nhúng Không gian & Thời gian
        self.spatiotemporal_injector = SpatiotemporalInjectionModule(
            pretrained_spatial_pos_embed=spatial_pos_embed,
            num_frames=num_frames,
            num_patches=self.num_patches,
            embed_dim=embed_dim,
        )

        # Module 3: T-S Hybrid Masked Transformer với LoRA trên Q, V
        self.transformer = TSHybridMaskedTransformer(
            pre_layernorm=pre_ln,
            encoder_layers=encoder_layers,
            use_hf_layers=use_hf,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Lưu sẵn ma trận T-S Hybrid Attention Mask (L, L) dưới dạng buffer không cần tính đạo hàm
        default_mask = build_ts_hybrid_attention_mask(
            num_frames=num_frames,
            num_patches=self.num_patches,
        )
        self.register_buffer("hybrid_attention_mask", default_mask)

        # Module 4: Engagement Classification Head
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
        """
        Tải trọng số CLIP ViT-B/16 từ HuggingFace (`openai/clip-vit-base-patch16`).
        Nếu môi trường không có kết nối mạng hoặc chỉ chạy kiểm thử kiến trúc,
        tự động khởi tạo cấu hình CLIP ViT-B/16 chuẩn tương đương.
        """
        if HAS_TRANSFORMERS and load_pretrained:
            try:
                clip_vision = CLIPVisionModel.from_pretrained(clip_model_name)
                vm = clip_vision.vision_model
                patch_conv = vm.embeddings.patch_embedding
                spatial_pos = vm.embeddings.position_embedding.weight  # (197, 768)
                pre_ln = vm.pre_layrnorm
                encoder_layers = vm.encoder.layers
                return patch_conv, spatial_pos, pre_ln, encoder_layers, True
            except Exception as exc:
                print(
                    f"[Thông báo] Không thể tải trực tuyến '{clip_model_name}' ({exc}). "
                    f"Sử dụng backbone CLIP ViT-B/16 nội bộ có cùng kích thước (D={embed_dim}, K={num_layers})."
                )

        # Khởi tạo kiến trúc CLIP ViT-B/16 nội bộ chuẩn xác tới từng tensor
        patch_conv = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        spatial_pos = torch.randn(num_patches + 1, embed_dim) * 0.02
        pre_ln = nn.LayerNorm(embed_dim, eps=1e-5)
        encoder_layers = nn.ModuleList([
            NativeCLIPViTBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        return patch_conv, spatial_pos, pre_ln, encoder_layers, False

    def forward(
        self,
        frames: torch.Tensor,
        boxes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, T, 3, 224, 224) — T frames ảnh toàn cảnh lớp học.
            boxes:  (B, T, 4) — Tọa độ Bounding Box chuẩn hóa [x_min, y_min, x_max, y_max] của học sinh.
        Returns:
            Dict chứa:
              - 'logits': (B, num_classes)
              - 'probs':  (B, num_classes)
              - 'task_embedding': (B, D) vector đại diện của Task Token ở lớp cuối cùng
              - 'sequence_length': chiều dài chuỗi L = 1 + T + T*N
        """
        t_frames = frames.shape[1]

        # Module 1: Mã hóa Đầu vào -> E_task (B, 1, D), E_box (B, T, D), E_patch (B, T, N, D)
        e_task, e_box, e_patch = self.input_encoder(frames, boxes)

        # Module 2: Nhúng Không gian & Thời gian + Ghép chuỗi -> (B, L, D)
        sequence = self.spatiotemporal_injector(e_task, e_box, e_patch)

        # Lấy hoặc tạo ma trận Hybrid Attention Mask (L, L) tương ứng với số frame T
        if t_frames == self.num_frames:
            mask = self.hybrid_attention_mask
        else:
            mask = build_ts_hybrid_attention_mask(
                num_frames=t_frames,
                num_patches=self.num_patches,
                device=sequence.device,
            )

        # Module 3: Đẩy chuỗi qua K=12 lớp Transformer với Hybrid Mask & LoRA
        encoded_sequence = self.transformer(sequence, mask)

        # Module 4: Phân loại từ Task Token (index 0) ở lớp cuối cùng
        logits, probs = self.engagement_head(encoded_sequence)

        return {
            "logits": logits,
            "probs": probs,
            "task_embedding": encoded_sequence[:, 0, :],
        }

    def get_trainable_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        Phân nhóm các tham số được phép cập nhật qua hàm Loss Cross-Entropy:
          1. LoRA adapters (Q, V của 12 lớp Transformer)
          2. Task Token (E_task)
          3. Coordinate MLP (mã hóa Box Trajectory)
          4. Temporal Positional Embeddings (P_temporal)
          5. Engagement Head (MLP phân loại)
        """
        lora_params: List[nn.Parameter] = []
        for name, param in self.transformer.named_parameters():
            if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                lora_params.append(param)

        return {
            "lora_qv": lora_params,
            "task_token": [self.input_encoder.task_token],
            "coord_mlp": list(self.input_encoder.coord_mlp.parameters()),
            "temporal_embed": [self.spatiotemporal_injector.p_temporal],
            "engagement_head": list(self.engagement_head.parameters()),
        }

    def print_parameter_summary(self) -> Tuple[int, int]:
        """Thống kê số lượng tham số đóng băng và tham số huấn luyện (PEFT)."""
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


