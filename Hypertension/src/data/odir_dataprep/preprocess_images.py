import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


"""
Implements Project Retinexis Data Preparation

This script performs dataset-wide standardization for retinal fundus images AFTER:
- metadata processing
- quality control (QC)
- duplicate detection

Order (must not be changed):
1) Mask top-left overlay text region (standardization step; masking is destructive)
2) Aspect-ratio-preserving resize to target resolution (512x512)
3) CLAHE (LAB, L channel only)
4) Normalize pixel values to [0, 1]

Notes:
- No augmentation here (training only)
- No train/val/test splits here (training phase)
- Raw files are NEVER overwritten (outputs go to processed/images/)
"""


#  -------------------------
# Config
#  -------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration for preprocessing.

    target_size:
        We use 512x512 to preserve retinal detail.
    overlay_mask:
        Controls masking of the top-left overlay region.
    clahe:
        CLAHE parameters for LAB/L channel enhancement.
    """
    target_size: Tuple[int, int] = (512, 512)

    # Masking region: top-left patch proportion to black out
    # Spec mentions top-left overlay and "31床" text; ODIR may not have that text,
    # but we still mask the region to eliminate overlay information.
    mask_enabled: bool = True
    mask_patch_frac: float = 0.20  # mask 20% width/height in top-left

    # CLAHE parameters
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)


#  -------------------------
# Paths
#  -------------------------
DATA_DIR = Path(__file__).resolve().parents[1]  # -> data/
ODIR_DIR = DATA_DIR / "ODIR-5K"
TRAIN_IMAGES_DIR = ODIR_DIR / "Training Images"
TEST_IMAGES_DIR = ODIR_DIR / "Testing Images"

MODULE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = MODULE_DIR / "processed"
LOGS_DIR = MODULE_DIR / "logs"
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Input lists produced earlier in pipeline
KEEP_LIST_CSV = LOGS_DIR / "dedup_keep_list.csv"

# Outputs
PREPROCESS_LOG_JSON = LOGS_DIR / "preprocess_log.json"
PREPROCESS_SUMMARY_TXT = LOGS_DIR / "preprocess_summary.txt"


#  -------------------------
# Utility functions
#  -------------------------
def resolve_image_path(image_filename: str) -> Optional[Path]:
    """Search Training Images and Testing Images dirs for the file."""
    for d in (TRAIN_IMAGES_DIR, TEST_IMAGES_DIR):
        path = d / image_filename
        if path.exists():
            return path
        # fallback: add .jpg if missing
        if Path(image_filename).suffix == "":
            path2 = d / f"{image_filename}.jpg"
            if path2.exists():
                return path2
    return None


def load_image_rgb(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image as RGB.

    Purpose:
    - OpenCV loads BGR by default, but our pipeline standardizes on RGB.
    - Returns None if unreadable/invalid (keeps pipeline robust).
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    if len(bgr.shape) != 3 or bgr.shape[2] != 3:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_top_left_overlay(image_rgb: np.ndarray, patch_frac: float) -> np.ndarray:
    """
    Mask the top-left region by blacking it out.

    Purpose:
    - Remove any clinical text overlays in the top-left corner.
    - Prevent leakage from non-retinal information.

    This is applied BEFORE resizing/CLAHE to ensure overlay pixels do not influence
    contrast enhancement.

    Args:
        patch_frac: fraction of width and height to mask in top-left.
    """
    h, w = image_rgb.shape[:2]
    ph = max(1, int(h * patch_frac))
    pw = max(1, int(w * patch_frac))

    masked = image_rgb.copy()
    masked[:ph, :pw] = 0
    return masked


def resize_aspect_ratio_preserving(
    image_rgb: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Resize with aspect ratio preservation using padding (letterbox).

    - maintain aspect ratio with center cropping or padding
    - interpolation:
        bicubic for downsampling, bilinear for upsampling

    We implement padding because it never crops retinal content.

    Steps:
    1) scale image so longer side fits target
    2) pad remaining space evenly to reach exact target size
    """
    target_w, target_h = target_size
    h, w = image_rgb.shape[:2]

    # Compute scale factor to fit within target box
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Choose interpolation based on whether we are upsampling or downsampling
    if scale < 1.0:
        interp = cv2.INTER_CUBIC  # downsampling
    else:
        interp = cv2.INTER_LINEAR  # upsampling

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)

    # Create padded canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)

    # Center paste
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


def apply_clahe_lab_l(image_rgb: np.ndarray, clip_limit: float, tile_grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Apply CLAHE in LAB color space on the L channel only.

    Purpose:
    - Standardize contrast and illumination across dataset.
    - CLAHE improves vessel and lesion visibility without distorting colors too much.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def normalize_to_unit(image_rgb: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] float32.

    Purpose:
    - provides standardized numeric scale for training models later.
    """
    return image_rgb.astype(np.float32) / 255.0


def save_image_uint8_rgb(image_rgb: np.ndarray, out_path: Path) -> None:
    """
    Save an RGB uint8 image to disk as JPG.

    Note:
    - OpenCV saves in BGR, so we convert back.
    - We save processed images as uint8 for storage efficiency.
      Normalized [0,1] arrays are primarily for model ingestion later.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if image_rgb is None or image_rgb.size == 0:
        raise ValueError(f"Empty image passed to save_image_uint8_rgb for: {out_path}")

    if image_rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image for saving, got {image_rgb.dtype} for: {out_path}")

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(str(out_path), bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for: {out_path}")


#  -------------------------
# Main pipeline
#  -------------------------
def preprocess_single_image(
    image_path: Path,
    image_filename: str,
    cfg: PreprocessConfig
) -> Dict[str, object]:
    """
    Run preprocessing pipeline for a single image and return log record.

    Log record includes:
    - input path
    - output path
    - whether each step was applied
    - any failure reason
    """
    record: Dict[str, object] = {
        "image_filename": image_filename,
        "input_path": str(image_path),
        "success": False,
        "failure_reason": "",
        "mask_applied": False,
        "clahe_applied": False,
        "target_size": cfg.target_size,
        "output_path": "",
    }

    img = load_image_rgb(image_path)
    if img is None:
        record["failure_reason"] = "load_failed_or_invalid_format"
        return record

    # Step 1: Mask top-left overlays
    if cfg.mask_enabled:
        img = mask_top_left_overlay(img, cfg.mask_patch_frac)
        record["mask_applied"] = True

    # Step 2: Resize with aspect ratio preserved
    img = resize_aspect_ratio_preserving(img, cfg.target_size)

    # Step 3: CLAHE
    if cfg.clahe_enabled:
        img = apply_clahe_lab_l(img, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
        record["clahe_applied"] = True

    # Step 4: Normalize to [0,1]
    img_norm = normalize_to_unit(img)

    # Save processed image (uint8)
    out_path = PROCESSED_IMAGES_DIR / image_filename
    save_image_uint8_rgb((img_norm * 255.0).astype(np.uint8), out_path)

    record["success"] = True
    record["output_path"] = str(out_path)
    return record


def run_preprocessing(cfg: PreprocessConfig):
    """
    Run preprocessing on all images in dedup_keep_list.csv.

    Critical design choice:
    - We process only the 'keep list' to ensure deduplication decisions
      are respected in downstream steps.
    """
    if not KEEP_LIST_CSV.exists():
        raise FileNotFoundError(
            f"Missing keep list: {KEEP_LIST_CSV}\n"
            "Run deduplicate.py first."
        )

    keep_df = pd.read_csv(KEEP_LIST_CSV)
    if "image_filename" not in keep_df.columns:
        raise ValueError("dedup_keep_list.csv must contain 'image_filename' column")

    logs: List[Dict[str, object]] = []

    filenames = keep_df["image_filename"].astype(str).tolist()
    total = len(filenames)

    for i, fname in enumerate(filenames, 1):
        # Skip already-processed images
        out_path = PROCESSED_IMAGES_DIR / fname
        if out_path.exists():
            continue

        if i == 1 or i % 200 == 0:
            print(f"[{i}/{total}] Processing {fname}...")

        img_path = resolve_image_path(fname)
        if img_path is None:
            logs.append({
                "image_filename": fname,
                "input_path": "",
                "success": False,
                "failure_reason": "file_not_found",
                "mask_applied": False,
                "clahe_applied": False,
                "target_size": cfg.target_size,
                "output_path": "",
            })
            continue

        logs.append(preprocess_single_image(img_path, fname, cfg))

    # Save full preprocessing log
    PREPROCESS_LOG_JSON.write_text(json.dumps(logs, indent=2), encoding="utf-8")

    # Summary
    total = len(logs)
    ok = sum(1 for r in logs if r.get("success"))
    fail = total - ok

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("PREPROCESSING SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append(f"Total images requested (keep list): {total}")
    summary_lines.append(f"Successfully processed: {ok} ({ok / total * 100:.2f}%)" if total else "Successfully processed: 0")
    summary_lines.append(f"Failed: {fail} ({fail / total * 100:.2f}%)" if total else "Failed: 0")
    summary_lines.append("")
    summary_lines.append("Config used:")
    summary_lines.append(json.dumps(asdict(cfg), indent=2))
    summary_lines.append("")

    PREPROCESS_SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Preprocessing complete.")
    print(f"Processed images directory: {PROCESSED_IMAGES_DIR}")
    print(f"Log: {PREPROCESS_LOG_JSON}")
    print(f"Summary: {PREPROCESS_SUMMARY_TXT}")


if __name__ == "__main__":
    run_preprocessing(PreprocessConfig())
