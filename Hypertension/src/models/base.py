"""
base — abstract base class and preprocessing contract for all classifiers.

Every model in this package inherits from BaseClassifier so the base class
enforces a consistent interface so that training scripts never need to
know which architecture they are working with.

Key purposes:
    - Define the forward / backbone / head / get_features contract
    - Own the linear-probe lock that prevents model.train() from
      reverting the backbone to train mode
    - Provide freeze / unfreeze helpers
    - Carry a PreprocessingConfig so each model declares its own
      input size, normalisation, and crop strategy
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


# Preprocessing contract
@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing requirements declared by each model."""

    input_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    crop_strategy: str


# every model here uses ImageNet stats
IMAGENET_NORM = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


# Abstract base classifier
class BaseClassifier(ABC, nn.Module):
    """
    Abstract base class for retinal image classifiers.
    """
    def __init__(self) -> None:
        super().__init__()
        self._linear_probe_locked: bool = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def backbone(self) -> nn.Module: ...

    @property
    @abstractmethod
    def head(self) -> nn.Module: ...

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def feature_dim(self) -> int: ...

    @property
    @abstractmethod
    def preprocessing_config(self) -> PreprocessingConfig: ...

    # parameter counting

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # freeze/unfreeze helpers
    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    # Training-mode override:
    def train(self, mode: bool = True) -> BaseClassifier:
        if self._linear_probe_locked and mode:
            self.training = True
            self.backbone.eval()
            self.head.train(True)
            return self
        return super().train(mode)

    def set_linear_probe_mode(self) -> None:
        self._linear_probe_locked = True
        self.freeze_backbone()
        self.backbone.eval()
        self.head.train()
        self.training = True

    def set_finetune_mode(self) -> None:
        self._linear_probe_locked = False
        self.unfreeze_backbone()
        self.train()

    # output validation
    def _validate_output_shape(self, logits: torch.Tensor, batch_size: int) -> None:
        expected = (batch_size, 1)
        if logits.shape != expected:
            raise RuntimeError(
                f"Output shape must be (B, 1). Got {logits.shape}, expected {expected}."
            )
