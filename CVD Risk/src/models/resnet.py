from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM

# Variant configuration
VARIANT_CONFIG = {
    "resnet50": {"model_name": "resnet50", "feature_dim": 2048, "input_size": 224},
}

class ResNetClassifier(BaseClassifier):
    def __init__(
        self,
        variant: str = "resnet50",
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