"""
dinov2 - DINOv2-based paired-image classifier and input utilities.

This module consolidates the previous `dinoV2_model.py` and
`dinoV2_data.py` into one implementation aligned with BaseClassifier.

Model flow:
    1. Two images (R/L) are encoded by a shared DINOv2 backbone
    2. Features are pooled (CLS token or mean patch tokens)
    3. Features are combined (concat/diff/sum)
    4. Linear head returns logits (B, 1)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
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


def _to_chw_float_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(image):
        tensor = image
    else:
        tensor = torch.from_numpy(np.asarray(image))

    if tensor.ndim == 3 and tensor.shape[0] not in {1, 3}:
        tensor = tensor.permute(2, 0, 1)

    is_float_input = torch.is_floating_point(tensor)
    tensor = tensor.float()
    if (not is_float_input) or (tensor.min() >= 0 and tensor.max() > 1.0):
        tensor = tensor / 255.0
    return tensor


class DINOv2Classifier(BaseClassifier):
    def __init__(
        self,
        variant: str = "giant",
        num_classes: int = 1,
        pretrained: bool = True,
        pool: str = "cls",
        combine: str = "concat",
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
        if combine not in {"concat", "diff", "sum"}:
            raise ValueError("combine must be 'concat', 'diff', or 'sum'")
        if num_classes != 1:
            raise ValueError(
                "BaseClassifier currently enforces binary logits with shape (B, 1); "
                f"received num_classes={num_classes}."
            )

        config = VARIANT_CONFIG[variant]
        self._feature_dim = config["feature_dim"]
        self._input_size = input_size_override or config["input_size"]
        self._pool = pool
        self._combine = combine

        self._backbone = self._load_backbone(
            model_name=config["model_name"],
            pretrained=pretrained,
        )

        head_in = 2 * self._feature_dim if combine == "concat" else self._feature_dim
        self._head = nn.Linear(head_in, 1)

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

    def _combine_features(self, features_r: torch.Tensor, features_l: torch.Tensor) -> torch.Tensor:
        if self._combine == "concat":
            return torch.cat([features_r, features_l], dim=1)
        if self._combine == "diff":
            return (features_r - features_l).abs()
        return features_r + features_l

    @staticmethod
    def _parse_paired_input(
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, Mapping):
            if "images_R" in x and "images_L" in x:
                return x["images_R"], x["images_L"]
            if "image_R" in x and "image_L" in x:
                return x["image_R"], x["image_L"]
            raise KeyError("Input dict must contain ('images_R', 'images_L') or ('image_R', 'image_L').")

        if isinstance(x, tuple) and len(x) == 2:
            return x[0], x[1]

        if not torch.is_tensor(x):
            raise TypeError("Input must be a tensor, tuple[Tensor, Tensor], or mapping with paired tensors.")

        if x.ndim == 5 and x.size(1) == 2:
            return x[:, 0], x[:, 1]

        raise ValueError(
            "Expected paired input as (images_R, images_L), {'images_R','images_L'}, "
            "or a stacked tensor with shape (B, 2, C, H, W)."
        )

    def forward(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        images_r, images_l = self._parse_paired_input(x)
        features = self.get_features((images_r, images_l))
        logits = self._head(features)
        self._validate_output_shape(logits, images_r.size(0))
        return logits

    def get_features(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        images_r, images_l = self._parse_paired_input(x)

        if all(not p.requires_grad for p in self._backbone.parameters()):
            with torch.no_grad():
                features_r = self._encode_single(images_r)
                features_l = self._encode_single(images_l)
        else:
            features_r = self._encode_single(images_r)
            features_l = self._encode_single(images_l)

        return self._combine_features(features_r, features_l)

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def feature_dim(self) -> int:
        return 2 * self._feature_dim if self._combine == "concat" else self._feature_dim

    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        return PreprocessingConfig(
            input_size=self._input_size,
            mean=IMAGENET_NORM["mean"],
            std=IMAGENET_NORM["std"],
            crop_strategy="resize",
        )


class DINOv2PairedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """
    Minimal paired-image dataset wrapper.

    Expected sample keys:
      - "image_R"
      - "image_L"
      - "label"
    """

    def __init__(
        self,
        dataset: Sequence[Mapping[str, Any]],
        transform: Callable[..., Any] | None = None,
        target_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_dtype = target_dtype

    def __len__(self) -> int:
        return len(self.dataset)

    def _apply_transform(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self.transform is None:
            return _to_chw_float_tensor(image)

        try:
            transformed = self.transform(image=image)
        except TypeError:
            transformed = self.transform(image)

        if isinstance(transformed, dict):
            transformed = transformed["image"]

        return _to_chw_float_tensor(transformed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        image_r = np.asarray(sample["image_R"])
        image_l = np.asarray(sample["image_L"])

        image_r_tensor = self._apply_transform(image_r)
        image_l_tensor = self._apply_transform(image_l)
        target = torch.as_tensor(sample["label"], dtype=self.target_dtype).reshape(1)

        return image_r_tensor, image_l_tensor, target


def build_dinov2_transform(
    input_size: int = 504,
    train: bool = True,
) -> transforms.Compose:
    transform_ops: list[Callable[[torch.Tensor], torch.Tensor]] = [
        transforms.Lambda(_to_chw_float_tensor),
        transforms.Resize((input_size, input_size)),
    ]
    if train:
        transform_ops.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_ops.extend(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=IMAGENET_NORM["mean"],
                std=IMAGENET_NORM["std"],
            ),
        ]
    )
    return transforms.Compose(transform_ops)


def dinov2_collate_fn(
    inputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    return {
        "images_R": torch.stack([sample[0] for sample in inputs], dim=0),
        "images_L": torch.stack([sample[1] for sample in inputs], dim=0),
        "labels": torch.stack([sample[2] for sample in inputs], dim=0),
    }
"""
resnet — ResNet-based classifier for retinal fundus images.

This serves as the universal baseline for our architecture comparison.
It anchors our results to the broader literature, as earlier BRSET 
work utilized ResNet-50 before transitioning to ConvNeXt V2.

Architecture flow:
    1. Input image (B, 3, input_size, input_size)
    2. timm backbone with num_classes=0 → pooled features (B, feature_dim)
    3. Linear head → logits (B, 1)
"""

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

        self._head = nn.Linear(self._feature_dim, 1)

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
