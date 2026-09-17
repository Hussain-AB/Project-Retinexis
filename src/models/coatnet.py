"""
coatnet — CoAtNet-based classifier for retinal fundus images.

HYBRID ARCHITECTURE: CoAtNet ("Conv + Attention Network") stacks two
different block types across its stages rather than mixing them within one
block like MaxViT does. Early stages use MBConv (depthwise convolution) to
build local features cheaply at high resolution; later stages switch to
Transformer blocks with relative self-attention once the feature map is
small enough for global attention to be affordable. Convolution handles
the inductive bias that attention lacks; attention handles the long-range
context convolution lacks.

Architecture flow:
    1. Input image (B, 3, input_size, input_size)
    2. timm backbone with num_classes=0 → pooled features (B, feature_dim)
    3. Linear head → logits (B, 1)

Variant configs live in VARIANT_CONFIG, same one-file pattern as
efficientnet.py — adding a size (e.g. coatnet_2) means adding one entry
there. Only the "_rw_" (Ross Wightman recipe) CoAtNet variants ship with
pretrained weights in timm; the plain coatnet_N_224 variants are untrained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM


# Variant configuration
VARIANT_CONFIG = {
    # MBConv stages -> relative-attention Transformer stages, ImageNet-1k pretrained.
    "base": {"model_name": "coatnet_1_rw_224.sw_in1k", "feature_dim": 768, "input_size": 224},
}


class CoAtNetClassifier(BaseClassifier):
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
