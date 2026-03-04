"""
retfound_mae - RETFound-MAE classifier.

This implements the original RETFound-MAE foundation model (ViT-Large/16, ~304M params).
It handles loading weights from the MAE pre-training, stripping out decoder/DDP layers,
and smoothly interpolating positional embeddings if the resolution differs from 224x224.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM

import logging
logger = logging.getLogger(__name__)


class RETFoundMAEClassifier(BaseClassifier):
    def __init__(
        self,
        pretrained: bool = True,  # review: added — test_models.py passes pretrained=False, factory expects it
        weights_path: str | None = None,
        input_size: int = 224,
        drop_path_rate: float = 0.1,
        global_pool: bool = False,
    ) -> None:
        super().__init__()
        
        self._input_size = input_size
        self._feature_dim = 1024  
        self._global_pool = global_pool
        

        self._backbone = timm.models.vision_transformer.VisionTransformer(
            img_size=224, 
            patch_size=16,
            embed_dim=self._feature_dim,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=drop_path_rate,
            num_classes=0, 
            global_pool="avg" if self._global_pool else "token",
        )
        
        self._head = nn.Linear(self._feature_dim, 1)
        
        if pretrained and weights_path is not None:  # review: gate on pretrained flag
            self._load_pretrained_weights(weights_path)
            
        if self._input_size != 224:
            self._interpolate_custom_pos_embed(self._input_size)

    def _load_pretrained_weights(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        logger.info(f"Loading RETFound-MAE weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        allowed_prefixes = ("cls_token", "pos_embed", "patch_embed.", "blocks.", "norm.")
        
        filtered_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace("module.", "")
            
            if clean_key.startswith(allowed_prefixes):
                filtered_dict[clean_key] = v
                
        msg = self._backbone.load_state_dict(filtered_dict, strict=False)
        logger.info(f"Loaded pretrained backbone. Missing keys: {msg.missing_keys}. Unexpected keys: {msg.unexpected_keys}")

    def _interpolate_custom_pos_embed(self, new_size: int):
        """
        Interpolates the 2D positional embedding for a new image size.
        Expects self._backbone.patch_embed and self._backbone.pos_embed to exist.
        """
        logger.info(f"Interpolating pos_embed from 224 to {new_size}")
        

        patch_size = self._backbone.patch_embed.patch_size[0]
        new_grid_size = new_size // patch_size
        num_new_patches = new_grid_size * new_grid_size
        
        self._backbone.patch_embed.img_size = (new_size, new_size)
        self._backbone.patch_embed.num_patches = num_new_patches

        pos_embed = self._backbone.pos_embed
        num_tokens = self._backbone.num_prefix_tokens
        
        token_pos = pos_embed[:, :num_tokens]
        patch_pos = pos_embed[:, num_tokens:]
        
        orig_grid_size = int((patch_pos.shape[1]) ** 0.5)
        
        if orig_grid_size != new_grid_size:
            dim = pos_embed.shape[-1]
            
            patch_pos = patch_pos.reshape(1, orig_grid_size, orig_grid_size, dim).permute(0, 3, 1, 2)
            
            patch_pos = F.interpolate(
                patch_pos, 
                size=(new_grid_size, new_grid_size), 
                mode="bicubic", 
                align_corners=False
            )
            
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
            
            new_pos_embed = torch.cat((token_pos, patch_pos), dim=1)
            
            self._backbone.pos_embed = nn.Parameter(new_pos_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        logits = self._head(features)
        self._validate_output_shape(logits, x.size(0))
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, feature_dim) using CLS token or global pooling based on init config."""
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
