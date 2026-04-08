"""
efficientnet_vessel — Dual-stream EfficientNet for retinal image + vessel mask fusion.

Architecture:
    Image branch:  EfficientNet-B2 (warm-started from trained image-only weights) → (B, 1408)
    Vessel branch: EfficientNet-B0 (ImageNet pretrained) → (B, 1280)
    Head:          Linear(2688, 1)

Intended training flow:
    1. Run image-only EfficientNet training (already done).
    2. Set backbone_checkpoint_path to the trained image-only best.pt.
       The trainer does a partial load: image backbone weights are copied,
       vessel backbone keeps ImageNet init, head is re-initialised (new size).
    3. Phase 1: freeze image backbone, train vessel backbone + head.
    4. Phase 2: fine-tune all with layer-wise LR decay on image backbone
       and full base_lr on vessel backbone (via extra_modules property).

The vessel branch uses B0 because vessel masks are sparse binary-ish maps
(artery/vein/vessel channels from RRWNet) and don't need B2 capacity.

Metadata fusion (use_metadata=True) works identically to EfficientNetClassifier:
    the trainer builds a metadata_head over (feature_dim + metadata_dim)
    and bypasses the model's own _head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM
from .efficientnet import VARIANT_CONFIG


# Vessel branch config
VESSEL_VARIANT = "b0"
VESSEL_FEATURE_DIM = VARIANT_CONFIG[VESSEL_VARIANT]["feature_dim"]   # 1280


class EfficientNetVesselClassifier(BaseClassifier):
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

        img_cfg = VARIANT_CONFIG[variant]
        self._img_feature_dim = img_cfg["feature_dim"]
        self._input_size = input_size_override or img_cfg["input_size"]
        self._feature_dim = self._img_feature_dim + VESSEL_FEATURE_DIM

        # Image branch — will be warm-started from trained image-only checkpoint
        self._image_backbone = timm.create_model(
            img_cfg["model_name"],
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )

        # Vessel branch — separate backbone for RRWNet artery/vein/vessel masks.
        # Starts from ImageNet init; trained from Phase 1 onwards.
        self._vessel_backbone = timm.create_model(
            VARIANT_CONFIG[VESSEL_VARIANT]["model_name"],
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )

        # Combined head (larger than image-only: 2688 vs 1408)
        self._head = nn.Linear(self._feature_dim, 1)

    # ── BaseClassifier interface ──────────────────────────────────────────────

    def forward(self, x: torch.Tensor, vessel_mask: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x, vessel_mask)
        logits = self._head(features)
        self._validate_output_shape(logits, x.size(0))
        return logits

    def get_features(self, x: torch.Tensor, vessel_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns (B, feature_dim) — concat of image and vessel features."""
        img_feat = self._image_backbone(x)
        if vessel_mask is not None:
            vessel_feat = self._vessel_backbone(vessel_mask)
            return torch.cat([img_feat, vessel_feat], dim=1)
        return img_feat

    @property
    def backbone(self) -> nn.Module:
        """Image backbone — used by optimiser for layer-wise LR decay."""
        return self._image_backbone

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

    @property
    def vessel_preprocessing_config(self) -> PreprocessingConfig:
        """Preprocessing config for vessel masks.

        Same spatial size as the image branch. Centred normalisation
        (mean=0.5, std=0.5) maps RRWNet's sigmoid outputs [0, 1] to [-1, 1].
        """
        return PreprocessingConfig(
            input_size=self._input_size,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            crop_strategy="pad_to_square",
        )

    @property
    def extra_modules(self) -> list[nn.Module]:
        """Vessel backbone gets full base_lr in Phase 2 via this property.

        The optimiser's build_layer_wise_param_groups() iterates this list
        and adds their params at base_lr (no layer-wise decay for the vessel
        branch since it's new and needs to learn freely).
        """
        return [self._vessel_backbone]

    # ── Freeze/unfreeze overrides ─────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze only the image backbone. Vessel backbone stays trainable.

        Called by set_linear_probe_mode() during Phase 1.
        Image backbone is already well-trained; vessel backbone needs to
        adapt from scratch, so it trains alongside the head in Phase 1.
        """
        for param in self._image_backbone.parameters():
            param.requires_grad = False
        for param in self._vessel_backbone.parameters():
            param.requires_grad = True
        for param in self._head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
