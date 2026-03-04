"""
retfound_green - RETFound-Green classifier.

RETFound-Green is the lightweight variant of the retinal specialist model (~22M params).
Published in Nature Communications, July 2025 (Engelmann et al.).

Loading is simpler than other RETFound variants — timm handles everything
via checkpoint_path. No custom weight loading logic needed (at least that's what I found in the repo)

Weights can be downloaded from the official GitHub release:
    wget https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import timm

from .base import BaseClassifier, PreprocessingConfig

import logging
logger = logging.getLogger(__name__)

# RETFound-Green uses 0.5 normalisation, NOT ImageNet stats
_RETFOUND_GREEN_NORM = {
    "mean": (0.5, 0.5, 0.5),  # review: must be tuples — PreprocessingConfig is a frozen dataclass
    "std":  (0.5, 0.5, 0.5),
}

_INPUT_SIZE = 392
_FEATURE_DIM = 384

# Official weights URL from the author's GitHub release
_WEIGHTS_URL = "https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth"
_WEIGHTS_FILENAME = "retfoundgreen_statedict.pth"

# Note: preprocessing must use torchvision.transforms.v2 as specified by the author.
# Key transforms: Resize((392, 392)), ToDtype(torch.float32, scale=True), Normalize((0.5,), (0.5,))

def _get_weights_path(cache_dir: str | Path | None = None) -> str:
    """
    Returns local path to weights, downloading them if not already cached.

    Weights are saved to cache_dir if provided, otherwise to ~/.cache/retfound_green/.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "retfound_green"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    weights_path = cache_dir / _WEIGHTS_FILENAME

    if not weights_path.exists():
        logger.info(f"Downloading RETFound-Green weights to {weights_path}...")
        torch.hub.download_url_to_file(_WEIGHTS_URL, str(weights_path))
        logger.info("Download complete.")
    else:
        logger.info(f"Using cached RETFound-Green weights from {weights_path}")

    return str(weights_path)

class RETFoundGreenClassifier(BaseClassifier):
    def __init__(
        self,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()

        self._feature_dim = _FEATURE_DIM
        self._input_size = _INPUT_SIZE

        weights_path = _get_weights_path(cache_dir) if pretrained else ""

        self._backbone = timm.create_model(
            "vit_small_patch14_reg4_dinov2",
            img_size = (_INPUT_SIZE, _INPUT_SIZE),
            num_classes=0,  # No classifier head
            drop_path_rate=drop_path_rate,
            checkpoint_path=weights_path,
        )

        self._backbone.global_pool = "avg"  # Ensure global average pooling is used

        self._head = nn.Linear(self._feature_dim, 1)  # Binary classification head

        #BaseClassifier stuff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)  # (Batch size, 384)
        logits = self._head(features)  # (Batch size, 1)
        self._validate_output_shape(logits, x.size(0))  
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
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
            input_size = self._input_size,
            mean = _RETFOUND_GREEN_NORM["mean"],
            std = _RETFOUND_GREEN_NORM["std"],
            crop_strategy="pad_to_square"  # review: was "pad_resize" which isn't a valid strategy in transforms.py
        )
