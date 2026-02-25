"""
seresnext — SE-ResNeXt-50 classifier for retinal fundus images.

This is the CIMT literature baseline (Dai et al.).
It is a CNN with squeeze-and-excitation blocks. 
Note: The original paper used a dual-eye bilateral input pipeline. 
This implementation uses a single-image pipeline to match our 
CNN baseline constraints, serving as a lower bound versus their 
published numbers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM

# Variant configuration
VARIANT_CONFIG = {
    "seresnext50": {"model_name": "seresnext50_32x4d", "feature_dim": 2048, "input_size": 224},
}