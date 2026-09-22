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

Augmentation design aligned with CIMT data prep notebook
(technique_randaug) which was medically validated:
    - hflip: safe for image-only (reconsider if eye laterality metadata added)
    - rotation: ±10° (conservative, matches notebook)
    - scale: 0.98–1.02 (simulates slight distance variation)
    - brightness/contrast: ±0.05 (low to prevent burnout, matches notebook)
    - saturation/hue: off by default (fundus colour is clinically meaningful)
    - blur: motion + gaussian at p=0.4 (simulates camera shake / defocus)
    - no vertical flip (fundus orientation is meaningful)
    - random_erasing: off (retinal vessel patterns are fine-grained)

Usage:
    from src.data.transforms import build_transforms

    train_tfm = build_transforms(model.preprocessing_config, is_train=True, config=config)
    val_tfm = build_transforms(model.preprocessing_config, is_train=False, config=config)
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

from src.models.base import PreprocessingConfig


# Crop strategies

class PadToSquare:
    """Pad the shorter side of a PIL Image to make it square.

    Uses black (0, 0, 0) padding to preserve the circular fundus region.
    No-op if the image is already square (both CIMT 512x512 and
    preprocessed ODIR 512x512).
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img

        max_side = max(w, h)
        padded = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        return padded


class CenterCropToSquare:
    """Crop the longer side to make the image square.

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


_CROP_STRATEGIES = {
    "pad_to_square": PadToSquare,
    "center_crop": CenterCropToSquare,
    "resize": None,
}


# Custom augmentation transforms

class RandomScale:
    """Random scale jitter via affine transform.

    Simulates slight distance variation between camera and eye.
    Black fill matches fundus background. Aligned with notebook's
    A.Affine(scale=(0.98, 1.02)).
    """

    def __init__(self, scale_min: float = 0.98, scale_max: float = 1.02):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, img: Image.Image) -> Image.Image:
        scale = random.uniform(self.scale_min, self.scale_max)
        return TF.affine(
            img,
            angle=0,
            translate=[0, 0],
            scale=scale,
            shear=[0, 0],
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        )


class RandomBlur:
    """Randomly apply either motion-style or gaussian blur.

    Simulates camera shake (motion blur) and defocus (gaussian blur).
    Aligned with notebook's OneOf([MotionBlur, GaussianBlur], p=0.4).

    torchvision doesn't have MotionBlur, so we approximate it with
    a directional kernel via PIL. The effect at blur_limit=3 is
    visually similar.
    """

    def __init__(self, blur_limit: int = 3, p: float = 0.4):
        self.blur_limit = blur_limit
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        if random.random() < 0.5:
            # Gaussian blur
            kernel_size = self.blur_limit if self.blur_limit % 2 == 1 else self.blur_limit + 1
            return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size])
        else:
            # Motion blur approximation via PIL directional filter
            radius = max(1, self.blur_limit // 2)
            return img.filter(ImageFilter.BoxBlur(radius))


# Pipeline builder

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
        config: TrainConfig instance. Only needed for is_train=True.

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
    transforms.append(T.Resize(
        (input_size, input_size),
        interpolation=T.InterpolationMode.BICUBIC,
    ))

    # Augmentation (PIL-based, before ToTensor)
    if is_train:
        transforms.extend(_build_augmentation(config))

    # Convert to tensor + normalise
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=mean, std=std))

    # Random erasing (tensor-based, must be after ToTensor)
    if is_train:
        erasing_prob = getattr(config, "aug_random_erasing", 0.0)
        if erasing_prob > 0:
            transforms.append(T.RandomErasing(
                p=erasing_prob,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0,
            ))

    return T.Compose(transforms)


# Paired (image + vessel mask) pipeline

class GeometricDraw:
    """One sampled geometric augmentation, reusable across branches.

    The image and the vessel mask must undergo the SAME spatial transform.
    Previously each branch ran its own Compose with its own random draws, so an
    hflip on the image (p=0.5) left the mask unflipped half the time -- putting
    the optic disc and the arteriovenous crossings on the opposite side of the
    retina from the vessels the mask describes. Rotation and scale decorrelated
    the same way.

    Reseeding the RNG identically before each pipeline does NOT fix this: the
    image pipeline also draws for ColorJitter and blur, so the two Composes
    consume different numbers of random values and the draws do not line up.
    The geometry has to be sampled once and passed to both, which is what this
    class is for.
    """

    __slots__ = ("hflip", "angle", "scale")

    def __init__(self, hflip: bool = False, angle: float = 0.0, scale: float = 1.0):
        self.hflip = hflip
        self.angle = angle
        self.scale = scale

    @classmethod
    def sample(cls, config: object | None) -> "GeometricDraw":
        """Draw from the same distributions _build_augmentation uses."""
        hflip = bool(getattr(config, "aug_hflip", True)) and random.random() < 0.5

        rotation = float(getattr(config, "aug_rotation", 10.0))
        angle = random.uniform(-rotation, rotation) if rotation > 0 else 0.0

        scale_min = float(getattr(config, "aug_scale_min", 0.98))
        scale_max = float(getattr(config, "aug_scale_max", 1.02))
        scale = random.uniform(scale_min, scale_max) if scale_max > scale_min else 1.0

        return cls(hflip=hflip, angle=angle, scale=scale)

    def is_identity(self) -> bool:
        return not self.hflip and self.angle == 0.0 and self.scale == 1.0

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.hflip:
            img = TF.hflip(img)
        if self.angle != 0.0:
            img = TF.rotate(
                img, self.angle,
                interpolation=T.InterpolationMode.BILINEAR, fill=0,
            )
        if self.scale != 1.0:
            img = TF.affine(
                img, angle=0, translate=[0, 0], scale=self.scale, shear=[0, 0],
                interpolation=T.InterpolationMode.BILINEAR, fill=0,
            )
        return img

    def __repr__(self) -> str:
        return f"GeometricDraw(hflip={self.hflip}, angle={self.angle:.3f}, scale={self.scale:.4f})"


def _build_photometric(config: object | None) -> list:
    """Colour and blur augmentation -- image branch only.

    These do not move pixels, so they never need to be mirrored onto the mask,
    and applying them to a mask would corrupt what it encodes.
    """
    augs = []
    brightness = getattr(config, "aug_brightness", 0.05)
    contrast = getattr(config, "aug_contrast", 0.05)
    saturation = getattr(config, "aug_saturation", 0.0)
    hue = getattr(config, "aug_hue", 0.0)
    if any(v > 0 for v in [brightness, contrast, saturation, hue]):
        augs.append(T.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        ))

    blur_prob = getattr(config, "aug_blur_prob", 0.4)
    blur_limit = getattr(config, "aug_blur_limit", 3)
    if blur_prob > 0:
        augs.append(RandomBlur(blur_limit=blur_limit, p=blur_prob))
    return augs


class PairedTransform:
    """Transform an image and its vessel mask under one shared geometry.

    Op order matches build_transforms exactly -- crop, resize, geometric,
    photometric, ToTensor, Normalize, erasing -- so an image produced here is
    distributed identically to one from the single-branch pipeline. Only the
    source of the geometric randomness differs.

    Call with mask=None to transform an image alone.
    """

    def __init__(
        self,
        image_config: PreprocessingConfig,
        vessel_config: PreprocessingConfig | None = None,
        config: object | None = None,
        is_train: bool = False,
    ) -> None:
        self.image_config = image_config
        self.vessel_config = vessel_config
        self.config = config
        self.is_train = is_train
        self._photometric = _build_photometric(config) if is_train else []
        self._erasing = None
        if is_train:
            prob = getattr(config, "aug_random_erasing", 0.0)
            if prob and prob > 0:
                self._erasing = T.RandomErasing(
                    p=prob, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0,
                )

    def _branch(
        self,
        img: Image.Image,
        cfg: PreprocessingConfig,
        draw: GeometricDraw,
        photometric: bool,
    ) -> torch.Tensor:
        crop = _CROP_STRATEGIES.get(cfg.crop_strategy)
        if crop is not None:
            img = crop()(img)
        img = TF.resize(
            img, [cfg.input_size, cfg.input_size],
            interpolation=T.InterpolationMode.BICUBIC,
        )
        if not draw.is_identity():
            img = draw(img)
        if photometric:
            for op in self._photometric:
                img = op(img)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=list(cfg.mean), std=list(cfg.std))
        if photometric and self._erasing is not None:
            tensor = self._erasing(tensor)
        return tensor

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        draw = GeometricDraw.sample(self.config) if self.is_train else GeometricDraw()
        image_t = self._branch(image, self.image_config, draw, photometric=True)
        if mask is None or self.vessel_config is None:
            return image_t, None
        mask_t = self._branch(mask, self.vessel_config, draw, photometric=False)
        return image_t, mask_t


def build_vessel_transforms(
    preprocessing_config: PreprocessingConfig,
) -> T.Compose:
    """Build the DETERMINISTIC pipeline for RRWNet vessel masks.

    Mirrors build_transforms' spatial handling -- same crop strategy, same
    input_size, same interpolation -- but never augments, because the mask is
    applied independently of the image and any random geometry here would
    decorrelate the two branches outright.

    WARNING -- known limitation of the vessel arm, not of this function:
    the image branch DOES augment (rotation, scale, hflip), and the mask does
    not receive those same draws. A rotated image is therefore paired with an
    unrotated mask. Before the vessel arm runs for the paper, either drive both
    branches from one sampled geometric transform, or disable geometric
    augmentation for vessel runs. The core grid is unaffected: it never loads
    masks.

    Masks are single-channel sigmoid outputs in [0, 1]; mean=std=0.5 in the
    model's vessel_preprocessing_config maps them to [-1, 1].
    """
    input_size = preprocessing_config.input_size
    crop_strategy = preprocessing_config.crop_strategy

    transforms = []
    if crop_strategy in _CROP_STRATEGIES and _CROP_STRATEGIES[crop_strategy] is not None:
        transforms.append(_CROP_STRATEGIES[crop_strategy]())
    transforms.append(T.Resize(
        (input_size, input_size),
        interpolation=T.InterpolationMode.BICUBIC,
    ))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean=preprocessing_config.mean,
        std=preprocessing_config.std,
    ))
    return T.Compose(transforms)


def _build_augmentation(config: object | None) -> list:
    """Build augmentation transforms from config fields.

    Aligned with CIMT data prep notebook's technique_randaug:
        - HorizontalFlip(p=0.5)
        - Affine(rotate=±10, scale=0.98–1.02)
        - RandomBrightnessContrast(0.05, 0.05)
        - OneOf([MotionBlur(3), GaussianBlur(3)], p=0.4)
        - No VerticalFlip (fundus orientation meaningful)

    All parameters come from config YAML. Defaults match the notebook.
    """
    augs = []

    # Horizontal flip (p=0.5)
    # Safe for image-only mode. If eye laterality is later added as
    # metadata, either disable or flip the laterality label too.
    if getattr(config, "aug_hflip", True):
        augs.append(T.RandomHorizontalFlip(p=0.5))

    # Rotation (±10° default, black fill matches fundus background)
    rotation = getattr(config, "aug_rotation", 10.0)
    if rotation > 0:
        augs.append(T.RandomRotation(
            degrees=rotation,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        ))

    # Scale jitter (0.98–1.02 default)
    # Simulates slight camera distance variation.
    scale_min = getattr(config, "aug_scale_min", 0.98)
    scale_max = getattr(config, "aug_scale_max", 1.02)
    if scale_max > scale_min:
        augs.append(RandomScale(scale_min, scale_max))

    # Brightness and contrast (±0.05 default)
    # Kept low to prevent burnout per notebook comments.
    # Saturation and hue off by default — fundus colour is
    # clinically meaningful (vessel colour, haemorrhage redness).
    brightness = getattr(config, "aug_brightness", 0.05)
    contrast = getattr(config, "aug_contrast", 0.05)
    saturation = getattr(config, "aug_saturation", 0.0)
    hue = getattr(config, "aug_hue", 0.0)
    if any(v > 0 for v in [brightness, contrast, saturation, hue]):
        augs.append(T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ))

    # Blur (motion + gaussian, p=0.4 default)
    # Simulates camera shake and defocus during acquisition.
    blur_prob = getattr(config, "aug_blur_prob", 0.4)
    blur_limit = getattr(config, "aug_blur_limit", 3)
    if blur_prob > 0:
        augs.append(RandomBlur(blur_limit=blur_limit, p=blur_prob))

    # No vertical flip — superior/inferior retinal fields are
    # anatomically distinct. Flipping would create unrealistic images.

    return augs
