"""
xai.frames — the canonical analysis frame.

WHY THIS EXISTS
---------------
The six models do NOT see the same geometry:

    model              crop_strategy    input_size
    ---------------------------------------------
    retfound_dinov2    pad_to_square    518
    retfound_mae       pad_to_square    224
    dinov2 (large)     resize           504
    dinov3 (large)     resize           224*
    convnextv2         pad_to_square    224
    efficientnet (b2)  pad_to_square    260

    * whatever the HF processor reports

`resize` squashes a non-square fundus; `pad_to_square` letterboxes it. A pixel
at (x, y) in the original image therefore lands at DIFFERENT locations in
different models' input tensors. Comparing saliency maps across models without
fixing this measures the preprocessing, not the model.

The fix: every model is evaluated on an identical square CANONICAL FRAME.

    canonical = resize(pad_to_square(original), (SIZE, SIZE))

Because the canonical frame is already square, both crop strategies become
no-ops, and each model's transform reduces to `Resize(input_size) -> Normalize`.
Geometry is now shared, attribution lives natively in canonical coordinates,
and anatomical masks only ever need to be built once per image.

This is a deliberate, reportable methodological choice. `check_frame.py`
verifies that canonical-frame AUROC reproduces each model's recorded
val_predictions.csv AUROC before any attribution is computed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

CANON_SIZE = 512


# --------------------------------------------------------------------------
# canonical frame construction
# --------------------------------------------------------------------------
def pad_to_square(img: np.ndarray, fill: int = 0) -> np.ndarray:
    """Letterbox to square with zero padding. Mirrors transforms.PadToSquare."""
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    if img.ndim == 3:
        out = np.full((side, side, img.shape[2]), fill, dtype=img.dtype)
        out[top:top + h, left:left + w] = img
    else:
        out = np.full((side, side), fill, dtype=img.dtype)
        out[top:top + h, left:left + w] = img
    return out


def to_canonical(
    img: np.ndarray,
    size: int = CANON_SIZE,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Original image (H, W, 3) uint8 -> canonical (size, size, 3) uint8."""
    squared = pad_to_square(img)
    if squared.shape[0] == size:
        return squared
    interp = interpolation if squared.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(squared, (size, size), interpolation=interp)


def load_canonical(path: str, size: int = CANON_SIZE) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    return to_canonical(img, size)


def load_canonical_mask(path: str, size: int = CANON_SIZE) -> np.ndarray:
    """Load an auxiliary map (e.g. an RRWNet vessel PNG) into the canonical frame.

    Uses the SAME pad+resize geometry as the image, so masks stay registered.
    """
    m = np.array(Image.open(path).convert("RGB"))
    return to_canonical(m, size)


# --------------------------------------------------------------------------
# field of view
# --------------------------------------------------------------------------
def fov_mask(img: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Binary circular field-of-view mask from the black surround.

    Same procedure as scripts/generate_vessel_masks.generate_roi_mask, so the
    FOV used here matches the one RRWNet saw.
    """
    gray = img.max(axis=2)
    binary = (gray > threshold).astype(np.uint8)
    filled = ndimage.binary_fill_holes(binary).astype(np.uint8)

    labeled, n = ndimage.label(filled)
    if n == 0:
        return np.ones(gray.shape, dtype=bool)

    sizes = ndimage.sum(filled, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


@dataclass(frozen=True)
class FovGeometry:
    cx: float
    cy: float
    radius: float

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius


def fov_geometry(mask: np.ndarray) -> FovGeometry:
    """Centroid and equivalent-circle radius of the FOV."""
    ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    radius = float(np.sqrt(mask.sum() / np.pi))
    return FovGeometry(cx=cx, cy=cy, radius=radius)


# --------------------------------------------------------------------------
# perturbation baseline
# --------------------------------------------------------------------------
def blurred_baseline(img: np.ndarray, sigma_frac: float = 0.05) -> np.ndarray:
    """The reference image that occluded regions are blended toward.

    DO NOT occlude fundus images to black. A black region is indistinguishable
    from the FOV surround, which is in-distribution for every one of these
    models -- the model shrugs and the attribution collapses toward zero
    everywhere. Blending toward a heavily blurred copy preserves local colour,
    illumination and vignetting while destroying the structure (vessel calibre,
    disc margin, texture) that the ablation is meant to remove.
    """
    sigma = max(1.0, sigma_frac * img.shape[0])
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def blend(img: np.ndarray, baseline: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """keep in [0,1], shape (H, W) or (N, H, W). 1 = show image, 0 = show baseline."""
    if keep.ndim == 2:
        k = keep[..., None]
        return (img * k + baseline * (1.0 - k)).astype(np.float32)
    k = keep[..., None]
    return (img[None] * k + baseline[None] * (1.0 - k)).astype(np.float32)


# --------------------------------------------------------------------------
# laterality
# --------------------------------------------------------------------------
def canonicalise_laterality(img: np.ndarray, disc_x: float) -> tuple[np.ndarray, bool]:
    """Flip so the optic disc always sits on the same side.

    Radial and zonal statistics smear if left and right eyes are pooled without
    this. Returns (possibly flipped image, was_flipped).
    """
    if disc_x < img.shape[1] / 2.0:
        return img[:, ::-1].copy(), True
    return img, False


def normalise_map(sal: np.ndarray, fov: np.ndarray, mode: str = "relu_sum1") -> np.ndarray:
    """Turn a signed saliency map into a comparable distribution inside the FOV.

    relu_sum1 : clip negatives to zero and normalise to sum 1 ("positive
                evidence mass"). Required for mass/enrichment metrics.
    absolute  : absolute value, sum 1.
    signed    : leave as-is, FOV-masked (used for rank correlation).
    """
    s = np.where(fov, sal, 0.0).astype(np.float64)
    if mode == "signed":
        return s
    if mode == "absolute":
        s = np.abs(s)
    else:
        s = np.clip(s, 0.0, None)
    total = s.sum()
    if total <= 0:
        return np.zeros_like(s)
    return s / total
