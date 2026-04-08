"""
dinov3 — DINOv3-Large classifier for retinal fundus images.

ViT-Large, ~304M params. Loaded via HuggingFace Transformers.
Architecture flow:
    1. Input image (B, 3, input_size, input_size)
    2. AutoModel -> last_hidden_state (B, num_patches+1, hidden_size)
    3. CLS token or mean of patch tokens -> (B, hidden_size)
    4. Linear head -> logits (B, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoImageProcessor

from .base import BaseClassifier, PreprocessingConfig


_DINOV3_LARGE_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
# review: fallback defaults used only when pretrained=False to avoid gated
# Hugging Face metadata/config fetches during tests.
_DINOV3_FALLBACK_MODEL_TYPE = "dinov2"
_DINOV3_FALLBACK_INPUT_SIZE = 224
_DINOV3_FALLBACK_MEAN = (0.485, 0.456, 0.406)
_DINOV3_FALLBACK_STD = (0.229, 0.224, 0.225)

VARIANT_CONFIG = {
    "large": {
        "model_id": _DINOV3_LARGE_MODEL_ID,
        # ViT-Large/16 architecture dims — required when pretrained=False so the
        # local config skeleton matches the trained checkpoint (hidden 1024, 24 layers).
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "patch_size": 16,
    },
}


class DINOv3Classifier(BaseClassifier):
    def __init__(
        self,
        variant: str = "large",
        num_classes: int = 1,
        pretrained: bool = True,
        pool: str = "cls",
        # review: removed linear_probe param — trainer calls set_linear_probe_mode()
        input_size_override: int | None = None,
    ) -> None:
        super().__init__()

        if variant not in VARIANT_CONFIG:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(VARIANT_CONFIG.keys())}"
            )
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be 'cls' or 'mean'")
        if num_classes != 1:
            raise ValueError(
                "BaseClassifier currently enforces binary logits with shape (B, 1); "
                f"received num_classes={num_classes}."
            )

        config = VARIANT_CONFIG[variant]
        self._pool = pool

        if pretrained:
            processor = AutoImageProcessor.from_pretrained(config["model_id"])
            self._norm_mean = tuple(processor.image_mean)
            self._norm_std = tuple(processor.image_std)
            size = processor.size
            processor_size = size.get("height") or size.get("shortest_edge", _DINOV3_FALLBACK_INPUT_SIZE)
            self._input_size = input_size_override or processor_size
        else:
            # review: keep pretrained=False fully local/offline — no HF hub calls
            # for processor metadata when running constructor-only tests.
            self._norm_mean = _DINOV3_FALLBACK_MEAN
            self._norm_std = _DINOV3_FALLBACK_STD
            self._input_size = input_size_override or _DINOV3_FALLBACK_INPUT_SIZE

        self._backbone = self._load_backbone(
            model_id=config["model_id"],
            pretrained=pretrained,
            input_size=self._input_size,
            hidden_size=config.get("hidden_size"),
            num_hidden_layers=config.get("num_hidden_layers"),
            num_attention_heads=config.get("num_attention_heads"),
            intermediate_size=config.get("intermediate_size"),
            patch_size=config.get("patch_size"),
        )
        self._feature_dim = self._backbone.config.hidden_size

        self._head = nn.Linear(self._feature_dim, 1)

        # review: removed auto linear_probe — trainer owns mode switching

    @staticmethod
    def _load_backbone(
        model_id: str,
        pretrained: bool,
        input_size: int,
        hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        intermediate_size: int | None = None,
        patch_size: int | None = None,
    ) -> nn.Module:
        if pretrained:
            return AutoModel.from_pretrained(model_id)
        # review: build config locally for pretrained=False to avoid gated
        # repo access from AutoConfig.from_pretrained(model_id).
        # Pass explicit architecture dims so the skeleton matches the checkpoint
        # (e.g. large variant needs hidden_size=1024, not the default base 768).
        hf_config = AutoConfig.for_model(_DINOV3_FALLBACK_MODEL_TYPE)
        hf_config.image_size = input_size
        if hidden_size is not None:
            hf_config.hidden_size = hidden_size
        if num_hidden_layers is not None:
            hf_config.num_hidden_layers = num_hidden_layers
        if num_attention_heads is not None:
            hf_config.num_attention_heads = num_attention_heads
        if intermediate_size is not None:
            hf_config.intermediate_size = intermediate_size
        if patch_size is not None:
            hf_config.patch_size = patch_size
        return AutoModel.from_config(hf_config)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: (B, num_patches + 1, hidden_dim)
        # index 0 is the CLS token; 1: are the patch tokens
        outputs = self._backbone(pixel_values=x)
        if self._pool == "cls":
            return outputs.last_hidden_state[:, 0]
        return outputs.last_hidden_state[:, 1:].mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input tensor with shape (B, C, H, W).")
        features = self.get_features(x)
        logits = self._head(features)
        self._validate_output_shape(logits, x.size(0))
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        # review: removed manual no_grad — trainer handles eval context,
        # this blocks gradient flow during Phase 2 fine-tuning
        return self._encode(x)

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        return PreprocessingConfig(
            input_size=self._input_size,
            mean=self._norm_mean,
            std=self._norm_std,
            crop_strategy="resize",
        )
