"""
merge_datasets.py

Combines ODIR and BRSET into one CSV and generates 5-fold stratified splits.
BRSET patient IDs are shifted by +5000 to avoid collisions with ODIR.

Usage: python tasks/hypertension/dataprep/merge_datasets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# -------------------------
# Repo root (mirrors pattern in scripts/generate_splits.py)
# -------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# -------------------------
# CONFIGURATION 
# -------------------------

ODIR_METADATA_CSV = (
    REPO_ROOT / "Hypertension/src/data/odir_dataprep/processed/metadata_processed.csv"
)

#TODO Set this to where metadata_processing.py saved the BRSET output.
BRSET_METADATA_CSV = REPO_ROOT / "path/to/brset/processed_metadata.csv"

OUT_METADATA_CSV = REPO_ROOT / "Hypertension/data/metadata_combined.csv"
OUT_SPLITS_JSON  = REPO_ROOT / "tasks/hypertension/splits/hypertension_combined_5fold_splits.json"

# Numeric offset added to every BRSET patient_id to form patient_uid.
BRSET_PATIENT_OFFSET = 5000

N_SPLITS = 5
SEED     = 42



# -------------------------
# Column mapping constants
# -------------------------

# Unified column names
COL_PATIENT_ID  = "patient_id"    # original int per dataset
COL_PATIENT_UID = "patient_uid"   # unique int across both datasets (used for splits)
COL_IMAGE_ID    = "image_id"      # globally unique string
COL_FILENAME    = "image_filename"
COL_AGE         = "patient_age"
COL_LABEL       = "hypertension"  # binary 0/1
COL_DATASET     = "dataset"

UNIFIED_COLS = [
    COL_PATIENT_ID,
    COL_PATIENT_UID,
    COL_IMAGE_ID,
    COL_FILENAME,
    COL_AGE,
    COL_LABEL,
    COL_DATASET,
]


# -------------------------
# Loaders
# -------------------------

def load_odir(path: Path) -> pd.DataFrame:
    """Load ODIR metadata and rename columns to match the unified schema."""
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out[COL_PATIENT_ID]  = pd.to_numeric(df["patient_id"], errors="raise").astype(int)
    out[COL_PATIENT_UID] = out[COL_PATIENT_ID]          # no offset for ODIR
    out[COL_IMAGE_ID]    = df["image_id"].astype(str)
    out[COL_FILENAME]    = df["image_filename"].astype(str)
    out[COL_AGE]         = pd.to_numeric(df["age"], errors="raise").astype(int)
    out[COL_LABEL]       = pd.to_numeric(df["label"], errors="raise").astype(int)
    out[COL_DATASET]     = "odir"

    return out[UNIFIED_COLS]


def load_brset(path: Path) -> pd.DataFrame:
    """Load BRSET metadata and rename columns to match the unified schema."""
    df = pd.read_csv(path)

    pid = pd.to_numeric(df["patient_id"], errors="raise").astype(int)

    out = pd.DataFrame()
    out[COL_PATIENT_ID]  = pid
    out[COL_PATIENT_UID] = pid + BRSET_PATIENT_OFFSET
    out[COL_IMAGE_ID]    = df["image_id"].astype(str)
    out[COL_FILENAME]    = df["image_filename"].astype(str)
    out[COL_AGE]         = pd.to_numeric(df["age"], errors="raise").astype(int)
    out[COL_LABEL]       = pd.to_numeric(df["hypertension"], errors="raise").astype(int)
    out[COL_DATASET]     = "brset"

    return out[UNIFIED_COLS]


# -------------------------
# Validation
# -------------------------

def validate_dataset(df: pd.DataFrame, name: str) -> None:
    """Per-dataset checks before merging. Raises on failure."""
    print(f"\n  Validating {name}...")

    for col in UNIFIED_COLS:
        n_null = df[col].isna().sum()
        if n_null:
            raise ValueError(f"[{name}] '{col}' has {n_null} null value(s).")

    invalid = ~df[COL_LABEL].isin([0, 1])
    if invalid.any():
        bad = df.loc[invalid, COL_LABEL].unique().tolist()
        raise ValueError(f"[{name}] '{COL_LABEL}' contains non-binary values: {bad}")

    dup_images = df[COL_IMAGE_ID].duplicated().sum()
    if dup_images:
        raise ValueError(f"[{name}] {dup_images} duplicate image_id value(s) found.")

    inconsistent = df.groupby(COL_PATIENT_UID)[COL_LABEL].nunique()
    bad_patients = inconsistent[inconsistent > 1].index.tolist()
    if bad_patients:
        raise ValueError(f"[{name}] {len(bad_patients)} patient(s) have inconsistent labels: {bad_patients[:5]}")

    pos_rate = df[COL_LABEL].mean()
    print(
        f"    rows={len(df):,}  patients={df[COL_PATIENT_ID].nunique():,}"
        f"  patient_uid range=[{df[COL_PATIENT_UID].min()}, {df[COL_PATIENT_UID].max()}]"
        f"  hypertension_pos_rate={pos_rate:.4f}"
    )


def validate_no_patient_uid_overlap(odir_df: pd.DataFrame, brset_df: pd.DataFrame) -> None:
    """Assert the numeric offset is large enough that patient_uid never collides."""
    odir_max  = int(odir_df[COL_PATIENT_UID].max())
    brset_min = int(brset_df[COL_PATIENT_UID].min())

    if odir_max >= brset_min:
        raise ValueError(
            f"patient_uid collision: ODIR max={odir_max} >= BRSET min={brset_min}. "
            f"Increase BRSET_PATIENT_OFFSET (currently {BRSET_PATIENT_OFFSET})."
        )
    print(f"\n  patient_uid ranges do not overlap: ODIR [0–{odir_max}], BRSET [{brset_min}+]  ✓")



def validate_combined(df: pd.DataFrame) -> None:
    dup = df[COL_IMAGE_ID].duplicated().sum()
    if dup:
        raise ValueError(
            f"{dup} duplicate image_id value(s) found in combined table. "
            "ODIR and BRSET image_id formats should be disjoint ('0_left' vs 'img00001')."
        )
    print(f"\n  Combined image_id values are globally unique  ✓")
    print(f"  Combined rows : {len(df):,}")
    print(f"  Label distribution:")
    print(df.groupby([COL_DATASET, COL_LABEL]).size().rename("count").to_string())


# -------------------------
# Split generation
# -------------------------

def generate_splits(df: pd.DataFrame) -> dict:
    """K-fold splits grouped by patient, stratified by dataset+label."""
    work = df[[COL_IMAGE_ID, COL_PATIENT_UID, COL_LABEL, COL_DATASET]].dropna().copy()
    work[COL_LABEL] = work[COL_LABEL].astype(int)

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    X      = work[[COL_IMAGE_ID]]
    y      = work[COL_DATASET].astype(str) + "_" + work[COL_LABEL].astype(str)  # stratify on dataset+label
    groups = work[COL_PATIENT_UID]

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
        train_df = work.iloc[train_idx]
        val_df   = work.iloc[val_idx]

        folds.append({
            "fold":            fold_idx,
            "train_sample_ids": train_df[COL_IMAGE_ID].tolist(),
            "val_sample_ids":   val_df[COL_IMAGE_ID].tolist(),
        })

    return {
        "version":       1,
        "task_name":     "hypertension_combined",
        "n_splits":      N_SPLITS,
        "seed":          SEED,
        "group_col":     COL_PATIENT_UID,
        "label_col":     COL_LABEL,
        "sample_id_col": COL_IMAGE_ID,
        "n_rows":        int(len(work)),
        "n_patients":    int(work[COL_PATIENT_UID].nunique()),
        "folds":         folds,
    }


def validate_splits(split_dict: dict, df: pd.DataFrame) -> None:
    """Checks leakage, dataset presence, and label balance across all folds."""
    id_to_uid   = dict(zip(df[COL_IMAGE_ID], df[COL_PATIENT_UID]))
    id_to_ds    = dict(zip(df[COL_IMAGE_ID], df[COL_DATASET]))
    id_to_label = dict(zip(df[COL_IMAGE_ID], df[COL_LABEL]))

    overall_pos = float(df[COL_LABEL].mean())
    tolerance   = 0.05

    all_val_patients: set = set()
    errors: list[str]    = []

    print("\n  Fold summary:")
    for fold in split_dict["folds"]:
        fidx      = fold["fold"]
        train_ids = fold["train_sample_ids"]
        val_ids   = fold["val_sample_ids"]

        train_pats = {id_to_uid[s] for s in train_ids if s in id_to_uid}
        val_pats   = {id_to_uid[s] for s in val_ids   if s in id_to_uid}

        # Leakage within fold
        overlap = train_pats & val_pats
        if overlap:
            errors.append(f"Fold {fidx}: patient leakage — {len(overlap)} shared patients.")

        # Leakage across val folds
        repeated = all_val_patients & val_pats
        if repeated:
            errors.append(f"Fold {fidx}: {len(repeated)} val patients already in earlier fold.")
        all_val_patients |= val_pats

        # Dataset mixing
        for split_name, ids in [("train", train_ids), ("val", val_ids)]:
            ds_present = {id_to_ds[s] for s in ids if s in id_to_ds}
            for ds in ("odir", "brset"):
                if ds not in ds_present:
                    errors.append(f"Fold {fidx} {split_name}: '{ds}' images missing.")

        # Positive rate balance
        val_labels  = [id_to_label[s] for s in val_ids if s in id_to_label]
        val_pos     = sum(val_labels) / len(val_labels) if val_labels else float("nan")
        if abs(val_pos - overall_pos) > tolerance:
            errors.append(
                f"Fold {fidx}: val pos_rate={val_pos:.4f} deviates from overall "
                f"{overall_pos:.4f} by more than tolerance {tolerance}."
            )

        # Counts by dataset in val
        val_odir  = sum(1 for s in val_ids if id_to_ds.get(s) == "odir")
        val_brset = sum(1 for s in val_ids if id_to_ds.get(s) == "brset")
        print(
            f"    Fold {fidx}: train={len(train_ids):,}  val={len(val_ids):,}"
            f"  val_pos={val_pos:.4f}  val_odir={val_odir:,}  val_brset={val_brset:,}"
        )

    if errors:
        print("\n  Errors found:")
        for err in errors:
            print(f"    [ERROR] {err}")
        raise SystemExit("Split validation failed. Splits JSON was NOT saved.")

    print("  All folds passed validation  ✓")


# -------------------------
# Entry point
# -------------------------

def main() -> None:
    # Check inputs exist
    missing = [p for p in (ODIR_METADATA_CSV, BRSET_METADATA_CSV) if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] Input not found: {p}")
        raise SystemExit(
            "Update the CONFIGURATION block at the top of this script with correct paths."
        )

    # Load
    print("Loading datasets...")
    odir_df  = load_odir(ODIR_METADATA_CSV)
    brset_df = load_brset(BRSET_METADATA_CSV)

    # Validate each dataset individually
    print("\nValidating individual datasets...")
    validate_dataset(odir_df,  "ODIR")
    validate_dataset(brset_df, "BRSET")

    # Check patient_uid offset is sufficient
    validate_no_patient_uid_overlap(odir_df, brset_df)

    # Merge
    print("\nMerging...")
    combined = pd.concat([odir_df, brset_df], ignore_index=True)
    validate_combined(combined)

    # Save metadata
    OUT_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_METADATA_CSV, index=False)
    print(f"\n  Saved metadata  →  {OUT_METADATA_CSV}")

    # Generate and validate splits
    print("\nGenerating splits...")
    split_dict = generate_splits(combined)
    validate_splits(split_dict, combined)

    # Save splits
    OUT_SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SPLITS_JSON.open("w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)
    print(f"\n  Saved splits    →  {OUT_SPLITS_JSON}")

    print("\nDone.")


if __name__ == "__main__":
    main()
