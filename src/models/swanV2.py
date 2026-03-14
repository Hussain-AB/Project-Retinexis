
"""
swanV2 - SimMIM(SwinV2)-based model-only classifier.

Model flow:
    1. One image is encoded by a SwinV2 backbone (SimMIM-pretrained weights)
    2. Features are pooled
    3. Linear head returns logits (B, 1)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.models.swin_transformer_v2 import SwinTransformerV2, checkpoint_filter_fn

from .base import BaseClassifier, IMAGENET_NORM, PreprocessingConfig


_HF_REPO_ID = "zdaxie/SimMIM"

# Resolve the project root so we can find data/weights/ from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/models/../../ = project root

VARIANT_CONFIG = {
    "small": {
        "model_name": "swinv2_small_window16_256",
        "checkpoint_file": "simmim_swinv2_finetune_models/finetune_swinv2_small_1k_500k.pth",
        "feature_dim": 768,
        "input_size": 256,
    },
    "large": {
        "model_name": "swinv2_large_window16_256",
        "checkpoint_file": "simmim_swinv2_finetune_models/finetune_swinv2_large_1k_500k.pth",
        "feature_dim": 1536,
        "input_size": 256,
    },
    "huge": {
        "model_name": "swinv2_huge_window16_256",
        "checkpoint_file": "simmim_swinv2_finetune_models/finetune_swinv2_huge_1k_500k.pth",
        "feature_dim": 2816,
        "input_size": 256,
    },
    "giant": {
        # Not in timm's registry — architecture is constructed manually.
        "model_name": None,
        "checkpoint_file": "data/weights/swinv2_giant_22k_500k.pth",
        "feature_dim": 4096,
        "input_size": 256,
        "local_weights": True,
        # SwinV2-G architecture hyperparameters (from the SwinV2 paper).
        "arch_kwargs": {
            "img_size": 256,
            "patch_size": 4,
            "window_size": 16,
            "embed_dim": 512,
            "depths": (2, 2, 42, 4),
            "num_heads": (16, 32, 64, 128),
        },
    },
}


class SWANv2Classifier(BaseClassifier):
    def __init__(
        self,
        variant: str = "giant",
        num_classes: int = 1,
        pretrained: bool = True,
        pool: str = "mean",
        input_size_override: int | None = None,
    ) -> None:  
        super().__init__()

        if variant not in VARIANT_CONFIG:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(VARIANT_CONFIG.keys())}"
            )
        if pool != "mean":
            raise ValueError(
                "SWANv2/SimMIM uses mean pooling only. "
                "Set pool='mean'."
            )
        if num_classes != 1:
            raise ValueError(
                "BaseClassifier currently enforces binary logits with shape (B, 1); "
                f"received num_classes={num_classes}."
            )

        config = VARIANT_CONFIG[variant]
        self._input_size = input_size_override or config["input_size"]

        self._backbone = self._load_backbone(
            model_name=config["model_name"],
            checkpoint_file=config["checkpoint_file"],
            pretrained=pretrained,
            config=config,
        )
        self._feature_dim = int(getattr(self._backbone, "num_features", config["feature_dim"]))
        self._head = nn.Linear(self._feature_dim, 1)

    @staticmethod
    def _load_backbone(
        model_name: str | None,
        checkpoint_file: str,
        pretrained: bool,
        config: dict | None = None,
    ) -> nn.Module:
        # --- Giant variant: not in timm registry, build manually ---
        if model_name is None:
            if config is None or "arch_kwargs" not in config:
                raise RuntimeError(
                    "Giant variant requires 'arch_kwargs' in its config."
                )
            backbone = SwinTransformerV2(
                num_classes=0,
                **config["arch_kwargs"],
            )
        else:
            backbone = timm.create_model(model_name, pretrained=False, num_classes=0)

        if not pretrained:
            return backbone

        # --- Resolve checkpoint path ---
        is_local = config is not None and config.get("local_weights", False)
        if is_local:
            checkpoint_path = str(_PROJECT_ROOT / checkpoint_file)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Local weights not found: {checkpoint_path}"
                )
        else:
            checkpoint_path = hf_hub_download(
                repo_id=_HF_REPO_ID, filename=checkpoint_file
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("Unsupported SimMIM checkpoint format.")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state_dict, Mapping):
            raise RuntimeError("Unsupported SimMIM checkpoint format.")

        # Use timm's own filter for SwinV2 checkpoints (handles key remapping
        # and buffer skipping), then fall back to our normaliser.
        if model_name is None:
            state_dict = checkpoint_filter_fn(state_dict, backbone)
        else:
            state_dict = SWANv2Classifier._normalize_state_dict_keys(state_dict)

        if not state_dict:
            raise RuntimeError(f"No encoder weights found in '{checkpoint_file}'.")

        msg = backbone.load_state_dict(state_dict, strict=False)
        print(f"  SwinV2 weights loaded: missing={len(msg.missing_keys)}, "
              f"unexpected={len(msg.unexpected_keys)}")
        return backbone

    @staticmethod
    def _normalize_state_dict_keys(
        state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue

            new_key = key
            for prefix in ("module.", "model.", "encoder.", "backbone."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            # SimMIM pretraining includes decoder-only parameters.
            if new_key.startswith("decoder.") or new_key.startswith("mask_token"):
                continue

            cleaned[new_key] = value
        return cleaned

    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        features = self._backbone.forward_features(x)
        features = self._backbone.forward_head(features, pre_logits=True)
        if features.ndim != 2:
            raise RuntimeError(f"Unexpected feature shape from backbone: {tuple(features.shape)}")
        return features

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
        # review: removed manual no_grad guard — trainer handles eval context,
        # and this blocks gradient flow during Phase 2 fine-tuning
        return self._encode_single(x)

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
