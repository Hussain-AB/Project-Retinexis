"""
dinov3 — DINOv3-Large classifier for retinal fundus images.

ViT-Large, ~304M params. Loaded via HuggingFace Transformers.
Architecture flow:
    1. Input image (B, 3, 224, 224)
    2. AutoModel -> last_hidden_state (B, num_patches+1, hidden_size)
    3. CLS token or mean of patch tokens -> (B, hidden_size)
    4. Linear head -> logits (B, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM


# TODO: replace with the real HuggingFace model ID for DINOv3-Large.
# e.g. "facebook/dinov3-large" or whatever the team confirms.
_DINOV3_LARGE_MODEL_ID = "PLACEHOLDER/dinov3-large"

VARIANT_CONFIG = {
    "large": {
        "model_id": _DINOV3_LARGE_MODEL_ID,
        "input_size": 224,
    },
}


class DINOv3Classifier(BaseClassifier):
    def __init__(
        self,
        variant: str = "large",
        num_classes: int = 1,
        pretrained: bool = True,
        pool: str = "cls",
        linear_probe: bool = True,
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
        self._input_size = input_size_override or config["input_size"]
        self._pool = pool

        self._backbone = self._load_backbone(
            model_id=config["model_id"],
            pretrained=pretrained,
        )
        self._feature_dim = self._backbone.config.hidden_size

        self._head = nn.Linear(self._feature_dim, 1)

        if linear_probe:
            self.set_linear_probe_mode()

    @staticmethod
    def _load_backbone(model_id: str, pretrained: bool) -> nn.Module:
        if pretrained:
            return AutoModel.from_pretrained(model_id)
        hf_config = AutoConfig.from_pretrained(model_id)
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
        if all(not p.requires_grad for p in self._backbone.parameters()):
            with torch.no_grad():
                return self._encode(x)
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
            mean=IMAGENET_NORM["mean"],
            std=IMAGENET_NORM["std"],
            crop_strategy="resize",
        )
