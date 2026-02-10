import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

# This file implements (Quality Control & Filtering)

# -------------------------
# Config (All thresholds are centralized here)
# -------------------------

@dataclass(frozen=True)
class QCConfig:
    """
    Configuration for quality control thresholds.

    Keeping these thresholds in one place:
    - makes the pipeline auditable (important for reporting),
    - allows easy tuning without changing logic,
    - ensures consistency across runs.
    """
    # Resolution requirement
    # We check raw images have at least this size to support later 512x512 standardization.
    min_width: int = 512
    min_height: int = 512

    # Blur detection
    # Typical guidance:
    #   < 100: very blurry
    #   100-500: acceptable
    #   > 500: sharp
    blur_min_variance: float = 100.0

    # Illumination
    # Typical good fundus brightness: ~80-180
    brightness_min: float = 80.0
    brightness_max: float = 180.0

    # Artifacts/occlusion proxy
    # We implement a measurable proxy:
    #   If too much of the image is very dark (near-black), it may indicate
    #   heavy border/occlusion/capture failure.
    #
    # This is not perfect clinically, but it is:
    # - reproducible
    # - measurable
    # - logged
    dark_pixel_threshold: int = 15         # pixel intensity considered "near black"
    max_dark_fraction: float = 0.60        # if > 60% near-black pixels, flag

    # Text overlay detection 
    # For BRSET images, the specific "31床" may not exist,
    #
    # We implement a heuristic detector:
    # - look at a top-left patch
    # - detect "high-frequency edges" (text tends to create strong edges)
    # - flag if edges exceed a threshold
    overlay_patch_frac: float = 0.20       # top-left patch = 20% width/height
    overlay_edge_threshold: float = 0.08   # fraction of edge pixels in patch to flag



# -------------------------
# Paths
# -------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "raw" / "ODIR-5K" / "brazilian-ophthalmological" / "1.0.1"
IMAGES_DIR = RAW_DIR / "fundus_photos"

MODULE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = MODULE_DIR / "processed"
LOGS_DIR = MODULE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

METADATA_CSV = PROCESSED_DIR / "metadata_processed.csv"
QC_RESULTS_CSV = LOGS_DIR / "qc_results.csv"
QC_REPORT_JSON = LOGS_DIR / "qc_report.json"
QC_SUMMARY_TXT = LOGS_DIR / "qc_summary.txt"

# -------------------------
# Data structures
# -------------------------
@dataclass
class QCMetrics:
    """
    All QC metrics we compute per image.

    These metrics are the *evidence* for why an image was filtered or kept.
    Keeping them per-image is required for:
    - transparency
    - debugging
    - reporting filtered percentages and reasons
    """
    image_filename: str
    image_path: str

    width: int
    height: int

    blur_score: float
    brightness: float

    # Artifact/occlusion proxy
    dark_fraction: float

    # Overlay detection heuristic
    overlay_flag: bool

    # Status fields
    load_success: bool
    reasons_failed: List[str]



# -------------------------
# Core QC functions (each maps to one requirement)
# -------------------------
def load_image_rgb(image_path: Path) -> Optional[np.ndarray]:
    """
    Load an image and convert it to RGB.

    Purpose:
    - We explicitly control color conversion.
    - If loading fails or image is malformed, we treat it as a QC failure.

    Returns:
        RGB image as numpy array (H, W, 3) or None if load fails.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    if len(bgr.shape) != 3 or bgr.shape[2] != 3:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def compute_resolution(image: np.ndarray) -> Tuple[int, int]:
    """
    Compute image resolution (width, height).

    Purpose:
    - Ensure images can support standardization to 512x512 later.
    """
    h, w = image.shape[:2]
    return w, h


def compute_blur_score(image_rgb: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance on grayscale.

    Purpose:
    - Laplacian variance is a standard focus/blur heuristic.
    - Lower values mean blurrier images.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image_rgb: np.ndarray) -> float:
    """
    Compute mean pixel intensity across all channels.

    Purpose:
    - Very low brightness suggests underexposed images.
    - Very high brightness suggests overexposed images.
    """
    return float(np.mean(image_rgb))


def compute_dark_fraction(image_rgb: np.ndarray, threshold: int) -> float:
    """
    Compute fraction of near-black pixels.

    Purpose:
    - We implement a measurable proxy:
      large near-black region often indicates heavy borders, occlusion,
      capture failure, or missing retinal content.

    Args:
        threshold: intensity below which a pixel counts as near-black

    Returns:
        fraction of near-black pixels in the image.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray <= threshold).astype(np.float32)
    return float(dark.mean())


def detect_top_left_overlay(image_rgb: np.ndarray, patch_frac: float, edge_thresh: float) -> bool:
    """
    Heuristically detect presence of a text overlay in the top-left region.

    Purpose:
    - Some datasets stamp clinical info (IDs, camera text, etc.) in the top-left.
    - The spec requires detecting such overlays.

    Method (simple + reproducible):
    - Take a top-left patch (patch_frac of width/height).
    - Compute edges via Canny.
    - If the edge density is high, we flag it as likely overlay.

    Note:
    - This is a detection step only. Masking/blackout happens later in
      preprocess_images.py.
    """
    h, w = image_rgb.shape[:2]
    ph = max(1, int(h * patch_frac))
    pw = max(1, int(w * patch_frac))

    patch = image_rgb[:ph, :pw]
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0  # normalize to [0,1]
    return edge_density > edge_thresh


def evaluate_qc(image_rgb: np.ndarray, cfg: QCConfig) -> Tuple[Dict[str, float], List[str], bool]:
    """
    Evaluate quality metrics and determine fail reasons.

    Purpose:
    - centralizes the QC decision logic
    - returns both metrics and reasons for transparency/logging.

    Returns:
        metrics dict, list of reasons_failed, overlay_flag
    """
    w, h = compute_resolution(image_rgb)
    blur = compute_blur_score(image_rgb)
    bright = compute_brightness(image_rgb)
    dark_frac = compute_dark_fraction(image_rgb, cfg.dark_pixel_threshold)
    overlay = detect_top_left_overlay(image_rgb, cfg.overlay_patch_frac, cfg.overlay_edge_threshold)

    reasons = []

    # Resolution checks
    if w < cfg.min_width or h < cfg.min_height:
        reasons.append("low_resolution")

    # Blur checks
    if blur < cfg.blur_min_variance:
        reasons.append("excessive_blur")

    # Illumination checks
    if bright < cfg.brightness_min:
        reasons.append("too_dark")
    if bright > cfg.brightness_max:
        reasons.append("too_bright")

    # Artifact/occlusion proxy
    if dark_frac > cfg.max_dark_fraction:
        reasons.append("possible_occlusion_or_artifact")

    # Overlay flag is required to detect and report.
    if overlay:
        reasons.append("top_left_overlay_detected")

    metrics = {
        "width": w,
        "height": h,
        "blur_score": blur,
        "brightness": bright,
        "dark_fraction": dark_frac,
    }
    return metrics, reasons, overlay




# -------------------------
# Reporting helpers
# -------------------------
def summarize_fail_reasons(results: List[QCMetrics]) -> Dict[str, int]:
    """
    Count how many times each fail reason appears.

    Purpose:
    - supports requirement: "Document percentage of images filtered and reasons"
    """
    counts: Dict[str, int] = {}
    for r in results:
        for reason in r.reasons_failed:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def write_summary(results: List[QCMetrics], cfg: QCConfig):
    """
    Write a human-readable summary txt + a detailed json report.

    Purpose:
    - makes it easy to paste into slides / meeting notes
    - keeps machine-readable report for reproducibility
    """
    total = len(results)
    passed = sum(1 for r in results if len(r.reasons_failed) == 0 and r.load_success)
    failed = total - passed

    reason_counts = summarize_fail_reasons(results)

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("QUALITY CONTROL SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append(f"Total images evaluated: {total}")
    summary_lines.append(f"Passed QC: {passed} ({passed / total * 100:.2f}%)" if total else "Passed QC: 0")
    summary_lines.append(f"Filtered/flagged: {failed} ({failed / total * 100:.2f}%)" if total else "Filtered/flagged: 0")
    summary_lines.append("")
    summary_lines.append("Fail/flag reasons (counts):")
    for k in sorted(reason_counts.keys()):
        summary_lines.append(f"  - {k}: {reason_counts[k]}")
    summary_lines.append("")
    summary_lines.append("QC thresholds/config used:")
    summary_lines.append(json.dumps(asdict(cfg), indent=2))
    summary_lines.append("")

    QC_SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    # Detailed JSON report
    json_report = {
        "config": asdict(cfg),
        "total_images": total,
        "passed": passed,
        "failed_or_flagged": failed,
        "fail_reason_counts": reason_counts,
        "per_image": [asdict(r) for r in results],
    }
    QC_REPORT_JSON.write_text(json.dumps(json_report, indent=2), encoding="utf-8")


# -------------------------
# Main pipeline
# -------------------------
def run_quality_control(cfg: QCConfig):
    """
    Run quality control for all images referenced by metadata_processed.csv.

    Purpose:
    - ensures QC aligns with metadata (not random files in folder)
    - keeps a single source of truth for which images are considered in the dataset
    """
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {METADATA_CSV}\n"
            "Run process_metadata.py first."
        )

    df = pd.read_csv(METADATA_CSV)

    results: List[QCMetrics] = []

    for _, row in df.iterrows():
        image_filename = str(row["image_filename"])
        image_path = IMAGES_DIR / image_filename

        # Attempt load
        img = load_image_rgb(image_path)
        if img is None:
            results.append(QCMetrics(
                image_filename=image_filename,
                image_path=str(image_path),
                width=0,
                height=0,
                blur_score=0.0,
                brightness=0.0,
                dark_fraction=1.0,
                overlay_flag=False,
                load_success=False,
                reasons_failed=["load_failed_or_invalid_format"]
            ))
            continue

        metrics, reasons, overlay_flag = evaluate_qc(img, cfg)

        results.append(QCMetrics(
            image_filename=image_filename,
            image_path=str(image_path),
            width=int(metrics["width"]),
            height=int(metrics["height"]),
            blur_score=float(metrics["blur_score"]),
            brightness=float(metrics["brightness"]),
            dark_fraction=float(metrics["dark_fraction"]),
            overlay_flag=bool(overlay_flag),
            load_success=True,
            reasons_failed=reasons
        ))

    # Save per-image QC results as CSV (easy for quick inspection)
    out_df = pd.DataFrame([{
        "image_filename": r.image_filename,
        "load_success": r.load_success,
        "width": r.width,
        "height": r.height,
        "blur_score": r.blur_score,
        "brightness": r.brightness,
        "dark_fraction": r.dark_fraction,
        "overlay_flag": r.overlay_flag,
        "reasons_failed": ";".join(r.reasons_failed),
        "num_reasons": len(r.reasons_failed),
    } for r in results])

    out_df.to_csv(QC_RESULTS_CSV, index=False)

    # Write reports
    write_summary(results, cfg)

    print("Quality control complete.")
    print(f"Per-image results: {QC_RESULTS_CSV}")
    print(f"Summary: {QC_SUMMARY_TXT}")
    print(f"Detailed report: {QC_REPORT_JSON}")


if __name__ == "__main__":
    run_quality_control(QCConfig())