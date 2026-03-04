"""
dinov2 - DINOv2-based model-only classifier.

Model flow:
    1. One image is encoded by a DINOv2 backbone
    2. Features are pooled (CLS token or mean patch tokens)
    3. Linear head returns logits (B, 1)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from .base import BaseClassifier, IMAGENET_NORM, PreprocessingConfig


# Backbone variants available from facebookresearch/dinov2 torch.hub.
VARIANT_CONFIG = {
    "base": {"model_name": "dinov2_vitb14", "feature_dim": 768, "input_size": 504},
    "large": {"model_name": "dinov2_vitl14", "feature_dim": 1024, "input_size": 504},
    "giant": {"model_name": "dinov2_vitg14", "feature_dim": 1536, "input_size": 504},
}

class DINOv2Classifier(BaseClassifier):
    def __init__(
        self,
        variant: str = "giant",
        num_classes: int = 1,
        pretrained: bool = True,
        pool: str = "cls",
        input_size_override: int | None = None,
        linear_probe: bool = True,
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
        self._feature_dim = config["feature_dim"]
        self._input_size = input_size_override or config["input_size"]
        self._pool = pool

        self._backbone = self._load_backbone(
            model_name=config["model_name"],
            pretrained=pretrained,
        )

        self._head = nn.Linear(self._feature_dim, 1)

        if linear_probe:
            self.set_linear_probe_mode()

    @staticmethod
    def _load_backbone(model_name: str, pretrained: bool) -> nn.Module:
        if pretrained:
            return torch.hub.load("facebookresearch/dinov2", model_name)
        return torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=False,
        )

    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        features = self._backbone.forward_features(x)
        if self._pool == "cls":
            return features["x_norm_clstoken"]
        return features["x_norm_patchtokens"].mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        images = x
        if images.ndim != 4:
            raise ValueError("Expected input tensor with shape (B, C, H, W).")
        features = self.get_features(images)
        logits = self._head(features)
        self._validate_output_shape(logits, images.size(0))
        return logits

    def get_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        images = x
        if images.ndim != 4:
            raise ValueError("Expected input tensor with shape (B, C, H, W).")

        if all(not p.requires_grad for p in self._backbone.parameters()):
            with torch.no_grad():
                features = self._encode_single(images)
        else:
            features = self._encode_single(images)
        return features

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
