"""
efficientnet — EfficientNet-based classifier for retinal fundus images.

This is the CNN baseline in our architecture comparison. It is
not capacity-matched to the ViT models (9M vs 86–307M parameters), but
its role is to represent the *standard practical approach* for medical
imaging with limited data.

EfficientNet-B2 runs at its native 260px resolution even in the primary
comparison. Forcing it down to 224px would cripple the baseline and make
ViT results look artificially stronger.

Architecture flow:
    1. Input image (B, 3, input_size, input_size)
    2. timm backbone with num_classes=0 → pooled features (B, feature_dim)
    3. Linear head → logits (B, 1)

Variant configs live in VARIANT_CONFIG. Adding a variant (like B4) means
adding one entry there with no other code changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM


# Variant configuration
VARIANT_CONFIG = {
    "b0": {"model_name": "efficientnet_b0", "feature_dim": 1280, "input_size": 224},
    "b2": {"model_name": "efficientnet_b2", "feature_dim": 1408, "input_size": 260},
    "b4": {"model_name": "efficientnet_b4", "feature_dim": 1792, "input_size": 380},
}


# Classifier
class EfficientNetClassifier(BaseClassifier):
    def __init__(
        self,
        variant: str = "b2",
        num_classes: int = 1,
        pretrained: bool = True,
        drop_rate: float = 0.3,
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
