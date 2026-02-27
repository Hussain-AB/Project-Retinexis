"""
dinov2 - DINOv2-based single-image classifier and input utilities.

Model flow:
    1. One image is encoded by a DINOv2 backbone
    2. Features are pooled (CLS token or mean patch tokens)
    3. Linear head returns logits (B, 1)
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
        combine: str | None = None,
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

        # Retained for backward compatibility with existing configs.
        _ = combine

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


class DINOv2Dataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Minimal single-image dataset wrapper. Note that you need to ensure you are using the correct format of the dataset,
    specifically use the data set at load_dataset("abdalla390/CIMT_5806_Thickness_Classification"), which is private,
    contact abdalla390 to get access.

    Expected sample keys:
      - "image"
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        image = np.asarray(sample["image"])

        image_tensor = self._apply_transform(image)
        target = torch.as_tensor(sample["label"], dtype=self.target_dtype).reshape(1)

        return image_tensor, target



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
    inputs: Sequence[tuple[torch.Tensor, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    return {
        "images": torch.stack([sample[0] for sample in inputs], dim=0),
        "labels": torch.stack([sample[1] for sample in inputs], dim=0),
    }
