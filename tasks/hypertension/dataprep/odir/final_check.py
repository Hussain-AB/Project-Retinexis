import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class FinalCheckConfig:
    # Paths are repo-root anchored for portability
    processed_dir_name: str = "processed"
    logs_dir_name: str = "logs"

    # Expected filenames produced by your pipeline
    metadata_csv_name: str = "metadata_processed.csv"
    keep_list_name: str = "dedup_keep_list.csv"
    preprocess_summary_name: str = "preprocess_summary.txt"
    preprocess_log_name: str = "preprocess_log.json"

    # Processed image output folder (created by preprocess_images.py)
    processed_images_subdir: str = "images"

    # Output report
    report_name: str = "final_check_report.txt"


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {name} at {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"{name} is empty: {path}")
    return df


def _check_binary_labels(df: pd.DataFrame, col: str) -> List[str]:
    issues = []
    if col not in df.columns:
        return [f"Missing required column: {col}"]
    vals = set(df[col].dropna().unique().tolist())
    if not vals.issubset({0, 1}):
        issues.append(f"Non-binary labels in '{col}': {sorted(list(vals))}")
    return issues


def _exists_count(base: Path, filenames: List[str]) -> Tuple[int, List[str]]:
    missing = []
    for f in filenames:
        p = base / f
        if not p.exists():
            missing.append(f)
    return len(filenames) - len(missing), missing


def _raw_image_exists(train_dir: Path, test_dir: Path, filename: str) -> bool:
    return (train_dir / filename).exists() or (test_dir / filename).exists()


def processed_exists(processed_dir: Path, raw_filename: str) -> bool:
    # Try exact name first
    if (processed_dir / raw_filename).exists():
        return True

    stem = Path(raw_filename).stem
    for ext in [".jpg", ".jpeg", ".png"]:
        if (processed_dir / f"{stem}{ext}").exists():
            return True
    return False

def final_check(cfg: FinalCheckConfig = FinalCheckConfig()) -> None:
    module_dir = _module_dir()

    data_dir = module_dir.parents[0]  # -> data/
    odir_dir = data_dir / "ODIR-5K"
    train_images_dir = odir_dir / "Training Images"
    test_images_dir = odir_dir / "Testing Images"

    processed_dir = module_dir / cfg.processed_dir_name
    logs_dir = module_dir / cfg.logs_dir_name

    metadata_csv = processed_dir / cfg.metadata_csv_name
    keep_csv = logs_dir / cfg.keep_list_name
    preprocess_summary = logs_dir / cfg.preprocess_summary_name
    preprocess_log = logs_dir / cfg.preprocess_log_name

    processed_images_dir = processed_dir / cfg.processed_images_subdir
    report_path = logs_dir / cfg.report_name

    # -------------------------
    # Load inputs
    # -------------------------
    metadata_df = _read_csv_required(metadata_csv, cfg.metadata_csv_name)
    keep_df = _read_csv_required(keep_csv, cfg.keep_list_name)

    # -------------------------
    # Collect issues
    # -------------------------
    issues: List[str] = []

    # Path existence checks
    has_train = train_images_dir.exists()
    has_test = test_images_dir.exists()
    if not has_train and not has_test:
        issues.append(f"Neither image directory found: {train_images_dir} / {test_images_dir}")
    if not processed_images_dir.exists():
        issues.append(f"Processed images directory not found: {processed_images_dir}")
    if not preprocess_summary.exists():
        issues.append(f"Missing preprocessing summary: {preprocess_summary}")
    if not preprocess_log.exists():
        issues.append(f"Missing preprocessing log: {preprocess_log}")

    # Required columns
    for col in ["image_filename", "patient_id", "label"]:
        if col not in keep_df.columns:
            issues.append(f"Keep list missing column: {col}")
    for col in ["image_filename", "patient_id", "label"]:
        if col not in metadata_df.columns:
            issues.append(f"Metadata missing column: {col}")

    # Label validity
    issues.extend(_check_binary_labels(metadata_df, "label"))
    issues.extend(_check_binary_labels(keep_df, "label"))

    # No duplicate filenames in keep list
    if "image_filename" in keep_df.columns:
        dup_count = int(keep_df["image_filename"].duplicated().sum())
        if dup_count > 0:
            issues.append(f"Keep list has duplicate image_filename entries: {dup_count}")

    # Keep list must be subset of metadata (by image_filename)
    if "image_filename" in keep_df.columns and "image_filename" in metadata_df.columns:
        meta_set = set(metadata_df["image_filename"].astype(str).tolist())
        keep_set = set(keep_df["image_filename"].astype(str).tolist())
        extra_in_keep = sorted(list(keep_set - meta_set))
        if extra_in_keep:
            issues.append(
                f"Keep list contains {len(extra_in_keep)} filenames not present in metadata_processed.csv "
                f"(showing up to 10): {extra_in_keep[:10]}"
            )

    # Raw existence for all kept files (search both dirs)
    missing_raw: List[str] = []
    if (has_train or has_test) and "image_filename" in keep_df.columns:
        kept_files = keep_df["image_filename"].astype(str).tolist()
        for f in kept_files:
            if not _raw_image_exists(train_images_dir, test_images_dir, f):
                missing_raw.append(f)
        if missing_raw:
            issues.append(
                f"Missing raw images for {len(missing_raw)} kept entries "
                f"(showing up to 10): {missing_raw[:10]}"
            )

    # Processed existence for all kept files
    missing_processed: List[str] = []
    if processed_images_dir.exists() and "image_filename" in keep_df.columns:
        kept_files = keep_df["image_filename"].astype(str).tolist()

        for f in kept_files:
            if not processed_exists(processed_images_dir, f):
                missing_processed.append(f)

        if missing_processed:
            issues.append(
                f"Missing processed images for {len(missing_processed)} kept entries "
                f"(showing up to 10): {missing_processed[:10]}"
            )


    # -------------------------
    # Summary stats
    # -------------------------
    total_metadata = len(metadata_df)
    total_keep = len(keep_df)

    raw_count = 0
    if has_train:
        raw_count += len(list(train_images_dir.glob("*.jpg")))
    if has_test:
        raw_count += len(list(test_images_dir.glob("*.jpg")))

    processed_count = 0
    if processed_images_dir.exists():
        processed_count = len(list(processed_images_dir.glob("*.*")))

    # Patient-level integrity readiness: can group by patient_id
    patient_ready = ("patient_id" in keep_df.columns) and (keep_df["patient_id"].isna().sum() == 0)

    # -------------------------
    # Write report
    # -------------------------
    logs_dir.mkdir(parents=True, exist_ok=True)

    status = "PASS" if len(issues) == 0 else "FAIL"

    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("FINAL DATA PREP CHECK REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"STATUS: {status}")
    lines.append("")
    lines.append("Counts:")
    lines.append(f"- metadata_processed.csv rows: {total_metadata}")
    lines.append(f"- dedup_keep_list.csv rows: {total_keep}")
    lines.append(f"- raw jpg files (Training + Testing Images): {raw_count}")
    lines.append(f"- processed jpg files (processed/images): {processed_count}")
    lines.append("")
    lines.append("Key validations:")
    lines.append(f"- Patient grouping ready (patient_id non-missing in keep list): {patient_ready}")
    lines.append("")

    if issues:
        lines.append("Issues found:")
        for i, msg in enumerate(issues, start=1):
            lines.append(f"{i}. {msg}")
        lines.append("")
    else:
        lines.append("No issues found. Dataset artifacts are consistent.")
        lines.append("")

    # Include a small machine-readable JSON snippet for debugging
    debug = {
        "status": status,
        "total_metadata_rows": total_metadata,
        "total_keep_rows": total_keep,
        "raw_jpg_count": raw_count,
        "processed_jpg_count": processed_count,
        "missing_raw_count": len(missing_raw),
        "missing_processed_count": len(missing_processed),
    }
    lines.append("Debug summary (JSON):")
    lines.append(json.dumps(debug, indent=2))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final check complete. Report: {report_path}")
    print(f"STATUS: {status}")




if __name__ == "__main__":
    final_check()
