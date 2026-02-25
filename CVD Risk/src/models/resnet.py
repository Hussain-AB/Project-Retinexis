from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM

# Variant configuration
VARIANT_CONFIG = {
    "resnet50": {"model_name": "resnet50", "feature_dim": 2048, "input_size": 224},
}