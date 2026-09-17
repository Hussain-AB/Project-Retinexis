"""
maxvit — MaxViT-based classifier for retinal fundus images.

HYBRID ARCHITECTURE: MaxViT is not a pure transformer. Each stage is an
MBConv block (the same depthwise-separable convolution block EfficientNet
uses, giving local inductive bias and translation equivariance) followed by
block attention (windowed self-attention within local patches) and grid
attention (dilated, sparse self-attention across the whole feature map for
global context). Convolution and attention run inside the same stage, not
as separate model halves — this is what "Multi-Axis" refers to.

Architecture flow:
    1. Input image (B, 3, input_size, input_size)
    2. timm backbone with num_classes=0 → pooled features (B, feature_dim)
    3. Linear head → logits (B, 1)

Variant configs live in VARIANT_CONFIG, same one-file pattern as
efficientnet.py — adding a size (e.g. large) means adding one entry there.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM


# Variant configuration
VARIANT_CONFIG = {
    # MBConv + block/grid attention stages, ImageNet-1k pretrained.
    "base": {"model_name": "maxvit_base_tf_224.in1k", "feature_dim": 768, "input_size": 224},
}


class MaxViTClassifier(BaseClassifier):
    def __init__(
        self,
        variant: str = "base",
        num_classes: int = 1,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        input_size_override: int | None = None,
    ) -> None:
        super().__init__()

        if variant not in VARIANT_CONFIG:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(VARIANT_CONFIG.keys())}"
            )

        config = VARIANT_CONFIG[variant]
        self._feature_dim = config["feature_dim"]
        self._input_size = input_size_override or config["input_size"]

        self._backbone = timm.create_model(
            config["model_name"],
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )

        self._head = nn.Linear(self._feature_dim, 1)

    # BaseClassifier interface
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._backbone(x)
        logits = self._head(features)
        self._validate_output_shape(logits, x.size(0))
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, feature_dim) after global average pooling."""
        return self._backbone(x)

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
            crop_strategy="pad_to_square",
        )
