"""T-S Hybrid Masked Transformer for Student Engagement Recognition."""
from .dataset import (
    ENGAGEMENT_3CLASS_MAP,
    ENGAGEMENT_4CLASS_MAP,
    ViMELDEngagementDataset,
)
from .model import (
    CoordinateMLP,
    EngagementHead,
    InputEncodingModule,
    LoRALinear,
    SpatiotemporalInjectionModule,
    StudentEngagementTSHybridCLIP,
    TSHybridMaskedTransformer,
    boolean_to_additive_attention_mask,
    build_ts_hybrid_attention_mask,
)
from .trainer import configure_optimizer, evaluate, train_epoch, train_one_step

__all__ = [
    "ENGAGEMENT_3CLASS_MAP",
    "ENGAGEMENT_4CLASS_MAP",
    "ViMELDEngagementDataset",
    "CoordinateMLP",
    "InputEncodingModule",
    "SpatiotemporalInjectionModule",
    "LoRALinear",
    "build_ts_hybrid_attention_mask",
    "boolean_to_additive_attention_mask",
    "TSHybridMaskedTransformer",
    "EngagementHead",
    "StudentEngagementTSHybridCLIP",
    "configure_optimizer",
    "train_one_step",
    "train_epoch",
    "evaluate",
]
