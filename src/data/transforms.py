"""
data.transforms — Build torchvision transform pipelines from model config.

Model-agnostic: reads input_size, mean, std, and crop_strategy from the
model's PreprocessingConfig. The pipeline never hardcodes normalisation
values or input sizes.

Three crop strategies:
    pad_to_square  — pad shorter side with black (preserves circular fundus)
    center_crop    — crop to square from centre
    resize         — naive resize (distorts aspect ratio)

Augmentation is applied only to training transforms and controlled
entirely by config YAML fields. Val/test transforms are deterministic.

Both CIMT and ODIR/BRSET images arrive as 512x512 squares, so
PadToSquare is a no-op for both. But the strategy exists for future
datasets with non-square images, and it does no harm on square inputs.

EDA-informed defaults (CIMT, see spike doc Section 7-8):
    - hflip: safe for image-only (reconsider if eye laterality metadata added)
    - rotation: 15 degrees (conservative)
    - color_jitter: 0.1 (keep hue low — fundus colour is clinically meaningful)
    - random_erasing: 0.0 (off — retinal vessel patterns are fine-grained)
    - no vertical flip (fundus orientation is meaningful)

Usage:
    from src.data.transforms import build_transforms

    train_tfm = build_transforms(model.preprocessing_config, is_train=True, config=config)
    val_tfm = build_transforms(model.preprocessing_config, is_train=False, config=config)
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image

from src.models.base import PreprocessingConfig


class PadToSquare:
    """Pad the shorter side of a PIL Image to make it square.

    Uses black (0, 0, 0) padding to preserve the circular fundus region.
    Black padding matches the natural background of fundus photographs
    and the letterbox padding used in ODIR preprocessing.

    No-op if the image is already square (both CIMT 512x512 and
    preprocessed ODIR 512x512).
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img

        max_side = max(w, h)
        # Black padding — matches fundus background
        padded = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        # Centre the original image
        padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        return padded


class CenterCropToSquare:
    """Crop the longer side to make the image square.

    Loses content — use only when the image has non-essential borders.
    Not recommended for fundus images (risks cutting the retinal field).
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img

        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side))


# strategy dispatch
_CROP_STRATEGIES = {
    "pad_to_square": PadToSquare,
    "center_crop": CenterCropToSquare,
    "resize": None,  # no pre-squaring — Resize handles it (distorts AR)
}


def build_transforms(
    preprocessing_config: PreprocessingConfig,
    is_train: bool = False,
    config: object | None = None,
) -> T.Compose:
    """Build a torchvision transform pipeline from model + training config.

    Args:
        preprocessing_config: From model.preprocessing_config. Declares
            input_size, mean, std, crop_strategy.
        is_train: If True, adds augmentation transforms.
        config: TrainConfig instance (optional). If None, augmentation
            uses conservative defaults. Only needed for is_train=True.

    Returns:
        A Compose pipeline: PIL Image → Tensor (C, H, W), normalised.
    """
    input_size = preprocessing_config.input_size
    mean = preprocessing_config.mean
    std = preprocessing_config.std
    crop_strategy = preprocessing_config.crop_strategy

    transforms = []

    # Make square (if needed)
    if crop_strategy in _CROP_STRATEGIES and _CROP_STRATEGIES[crop_strategy] is not None:
        transforms.append(_CROP_STRATEGIES[crop_strategy]())

    # Resize to model's expected input size
    # Both CIMT (512x512) and ODIR (512x512) get resized here to
    # whatever the model needs: 260px for EfficientNet-B2, 224px for ViTs.
    transforms.append(T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC))

    # Training augmentation (if is_train)
    if is_train:
        transforms.extend(_build_augmentation(config))

    # Convert to tensor + normalise
    transforms.append(T.ToTensor())  # [0, 255] uint8 → [0.0, 1.0] float32
    transforms.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transforms)


def _build_augmentation(config: object | None) -> list:
    """Build augmentation transforms from config fields.

    All augmentation parameters come from the config YAML. If config is
    None, returns conservative defaults suitable for retinal images.

    Augmentation is applied AFTER squaring and resizing but BEFORE
    ToTensor/Normalize, so all operations work on PIL Images.
    """
    augs = []

    # Read config fields with safe defaults
    hflip = getattr(config, "aug_hflip", True)
    rotation = getattr(config, "aug_rotation", 15)
    color_jitter = getattr(config, "aug_color_jitter", 0.1)
    random_erasing = getattr(config, "aug_random_erasing", 0.0)

    # Horizontal flip
    # Safe for both CIMT and BRSET/ODIR in image-only mode.
    # If eye laterality is later added as metadata, either disable
    # this or add logic to flip the laterality label consistently.
    if hflip:
        augs.append(T.RandomHorizontalFlip(p=0.5))

    # Rotation
    # Mild rotation simulates camera angle variation. Black fill
    # matches fundus background. 15° is conservative per CIMT EDA.
    if rotation > 0:
        augs.append(T.RandomRotation(
            degrees=rotation,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,  # black fill — matches fundus background
        ))

    # Colour jitter
    # Brightness + contrast + saturation vary with acquisition setup.
    # Hue jitter is halved — fundus colour carries clinical signal
    # (vessel colour, optic disc hue, haemorrhage redness).
    if color_jitter > 0:
        augs.append(T.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=color_jitter * 0.5,  # half strength for hue
        ))

    # Random erasing
    # Applied AFTER ToTensor (operates on tensors, not PIL).
    # Off by default — retinal vessel patterns are fine-grained and
    # erasing can remove clinically meaningful structure.
    # Returned separately since it must go after ToTensor.
    # NOTE: we handle this in build_transforms directly since it
    # needs to be post-ToTensor. See below.

    # No vertical flip — fundus orientation is meaningful (superior/
    # inferior retinal fields are anatomically distinct).

    return augs


def _get_random_erasing_prob(config: object | None) -> float:
    """Get random erasing probability from config. Separated because
    RandomErasing operates on tensors, not PIL, so it must go after
    ToTensor in the pipeline."""
    return getattr(config, "aug_random_erasing", 0.0)


# Override build_transforms to handle RandomErasing placement correctly
_original_build = build_transforms


def build_transforms(
    preprocessing_config: PreprocessingConfig,
    is_train: bool = False,
    config: object | None = None,
) -> T.Compose:
    """Build transforms with correct RandomErasing placement."""
    input_size = preprocessing_config.input_size
    mean = preprocessing_config.mean
    std = preprocessing_config.std
    crop_strategy = preprocessing_config.crop_strategy

    transforms = []

    # Make square
    if crop_strategy in _CROP_STRATEGIES and _CROP_STRATEGIES[crop_strategy] is not None:
        transforms.append(_CROP_STRATEGIES[crop_strategy]())

    # Resize
    transforms.append(T.Resize(
        (input_size, input_size),
        interpolation=T.InterpolationMode.BICUBIC,
    ))

    # Augmentation (PIL-based, before ToTensor)
    if is_train:
        transforms.extend(_build_augmentation(config))

    # To tensor + normalise
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=mean, std=std))

    # Random erasing (tensor-based, after ToTensor+Normalize)
    if is_train:
        erasing_prob = _get_random_erasing_prob(config)
        if erasing_prob > 0:
            transforms.append(T.RandomErasing(
                p=erasing_prob,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0,  # black — matches fundus background
            ))

    return T.Compose(transforms)
