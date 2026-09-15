"""
xai.anatomy — retinal regions of interest in the canonical frame.

Vessels and artery/vein come from RRWNet, which this repo already runs
(scripts/generate_vessel_masks.py -> RGB PNG, R=artery, G=vein, B=vessel).
No AutoMorph install is needed for the vascular masks.

  !! CRITICAL !!
  generate_vessel_masks.py defaults to the RAW images directory, but the
  models were trained on images_preprocessed/. If those two differ
  geometrically (crop, resize, circular re-masking) the vessel masks will NOT
  register to the images the models actually see, and every enrichment number
  will be silently wrong. Regenerate the masks from the SAME directory as
  cfg.data_dir before running any of this. check_frame.py asserts this.

The optic disc is estimated (this repo has no disc segmenter) by combining two
cues that both peak on the disc: local brightness, and the convergence of the
major vessel arcades. It is a heuristic -- validate it on ~30 images with
figures.py::plot_disc_qc and report the hit rate in the paper. Swapping in a
real disc segmenter later only requires replacing estimate_disc().
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .frames import FovGeometry, fov_geometry

# Optic disc diameter as a fraction of the FOV diameter. A 45-degree fundus
# spans roughly 13.5 mm and the disc is roughly 1.8 mm, so ~0.13.
DISC_DIAM_FRAC = 0.13
# Fovea sits ~2.5 disc diameters temporal to the disc centre.
FOVEA_OFFSET_DD = 2.5


@dataclass(frozen=True)
class DiscGeometry:
    cx: float
    cy: float
    radius: float          # disc radius, px

    @property
    def dd(self) -> float:  # one disc diameter, px -- the unit for all zones
        return 2.0 * self.radius


def load_vessel_channels(vessel_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """RRWNet RGB PNG -> per-structure probability maps in [0, 1]."""
    v = vessel_rgb.astype(np.float32) / 255.0
    return {"artery": v[..., 0], "vein": v[..., 1], "vessel": v[..., 2]}


def estimate_disc(
    image: np.ndarray,
    fov: np.ndarray,
    vessel_prob: np.ndarray,
    diam_frac: float = DISC_DIAM_FRAC,
) -> DiscGeometry:
    """Locate the optic disc from brightness + vessel convergence.

    Both cues are z-scored inside the FOV and summed, so neither dominates on
    an image where one fails (e.g. a bright reflex artefact, or a disc obscured
    by media opacity).
    """
    geom = fov_geometry(fov)
    radius = 0.5 * diam_frac * geom.diameter
    sigma = max(2.0, radius)
    k = int(2 * round(3 * sigma) + 1)

    bright = cv2.GaussianBlur(image.mean(axis=2).astype(np.float32), (k, k), sigma)
    vessels = cv2.GaussianBlur(vessel_prob.astype(np.float32), (k, k), sigma)

    # Keep the search away from the FOV rim, where blur pulls in the surround.
    inner = ndimage.binary_erosion(fov, iterations=int(max(1, 0.08 * geom.radius)))
    if inner.sum() < 100:
        inner = fov

    def z(a):
        vals = a[inner]
        sd = vals.std()
        return (a - vals.mean()) / (sd if sd > 1e-6 else 1.0)

    score = np.where(inner, z(bright) + z(vessels), -np.inf)
    cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
    return DiscGeometry(cx=float(cx), cy=float(cy), radius=float(radius))


def _disk(shape, cx, cy, r) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _annulus(shape, cx, cy, r_in, r_out) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 >= r_in * r_in) & (d2 <= r_out * r_out)


def build_rois(
    image: np.ndarray,
    fov: np.ndarray,
    vessels: dict[str, np.ndarray],
    disc: DiscGeometry,
    vessel_thresh: float = 0.5,
    adjacent_px: int = 8,
) -> dict[str, np.ndarray]:
    """All regions of interest, as boolean masks confined to the FOV.

    Zone B (0.5-1.0 DD from the disc MARGIN) is where CRAE/CRVE/AVR are
    clinically measured, so it carries the strongest hypertension prior
    available. Periphery is a failure detector, not a biological region:
    attribution concentrated there means the model is reading border or
    vignetting artefacts.
    """
    shape = image.shape[:2]
    geom = fov_geometry(fov)
    dd = disc.dd

    vessel_bin = vessels["vessel"] >= vessel_thresh
    artery_bin = (vessels["artery"] >= vessel_thresh) & vessel_bin
    vein_bin = (vessels["vein"] >= vessel_thresh) & vessel_bin

    skel = skeletonize(vessel_bin)
    dilated = ndimage.binary_dilation(vessel_bin, iterations=adjacent_px)
    adjacent = dilated & ~vessel_bin

    disc_mask = _disk(shape, disc.cx, disc.cy, disc.radius)
    zone_b = _annulus(shape, disc.cx, disc.cy, disc.radius + 0.5 * dd, disc.radius + 1.0 * dd)
    zone_c = _annulus(shape, disc.cx, disc.cy, disc.radius + 0.5 * dd, disc.radius + 2.0 * dd)

    # Fovea is temporal to the disc: toward the FOV centre.
    direction = 1.0 if geom.cx >= disc.cx else -1.0
    fov_x = disc.cx + direction * FOVEA_OFFSET_DD * dd
    macula = _disk(shape, fov_x, disc.cy, dd)

    periphery = _annulus(shape, geom.cx, geom.cy, 0.90 * geom.radius, geom.radius)

    rois = {
        "disc": disc_mask,
        "zone_b": zone_b,
        "zone_c": zone_c,
        "artery": artery_bin,
        "vein": vein_bin,
        "vessel": vessel_bin,
        "vessel_skeleton": skel,
        "vessel_adjacent": adjacent,
        "macula": macula,
        "periphery": periphery,
    }
    rois["background"] = fov & ~(dilated | disc_mask)
    return {k: (v & fov) for k, v in rois.items()}


ROI_ORDER = [
    "disc", "zone_b", "zone_c",
    "artery", "vein", "vessel", "vessel_skeleton", "vessel_adjacent",
    "macula", "background", "periphery",
]

# ROIs used for the causal masking ablation. Masking overlapping regions
# separately is fine -- each is its own independent intervention.
ABLATION_ROIS = ["vessel", "disc", "zone_b", "macula", "background"]
