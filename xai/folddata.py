"""
xai.folddata — resolve a fold's HELD-OUT rows to image paths on disk.

The attribution study is fold-matched and strictly out-of-fold:

    for each fold k, take the images held out in fold k, and score every
    model's fold-k checkpoint on exactly those images.

That way all six models are compared on identical images that none of them
trained on. Pooling across k = 0..4 then covers the whole dataset once.

Anything else leaks: comparing model A's fold-0 checkpoint with model B's
fold-2 checkpoint on a shared image means at least one of them memorised it,
and memorised images produce visibly different (sharper, more confident)
saliency, which would masquerade as an architecture difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import PROJECT_ROOT


def _resolve(p: str | Path, project_root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else project_root / p


def held_out_rows(cfg, fold: int, project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Rows of the labels CSV that are in fold `fold`'s validation split."""
    splits_file = _resolve(cfg.splits_file, project_root)
    with open(splits_file) as f:
        splits = json.load(f)

    entry = splits["folds"][fold]
    # "val_sample_ids" is what src/data/splits.py actually writes; the other
    # spellings are kept for any hand-made splits file. Without the first key
    # every xai entry point raised KeyError, so nothing in this package had ever
    # run against a real splits file.
    val_ids = (
        entry.get("val_sample_ids")
        or entry.get("val")
        or entry.get("val_ids")
        or entry.get("validation")
    )
    if val_ids is None:
        raise KeyError(f"No val id list in fold {fold} of {splits_file}; keys={list(entry)}")

    meta = splits.get("metadata", splits)
    sample_id_col = meta.get("sample_id_col", cfg.sample_id_col)

    df = pd.read_csv(_resolve(cfg.labels_csv, project_root))
    if sample_id_col not in df.columns:
        raise ValueError(f"'{sample_id_col}' not in labels CSV; have {list(df.columns)[:12]}")

    out = df[df[sample_id_col].astype(str).isin({str(v) for v in val_ids})].copy()
    out["_sample_id"] = out[sample_id_col].astype(str)
    return out.reset_index(drop=True)


def image_paths(df: pd.DataFrame, cfg, project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Attach `_image_path` and `_rel` using the same rule as the Dataset classes."""
    image_dir = _resolve(cfg.data_dir, project_root)

    def rel_for(row) -> str:
        for col in ("relative_image_path", "filename", "image_filename"):
            v = row.get(col)
            if isinstance(v, str) and v:
                return v
        raise KeyError(
            "No image path column found. Expected one of "
            "relative_image_path / filename / image_filename."
        )

    df = df.copy()
    df["_rel"] = df.apply(rel_for, axis=1)
    df["_image_path"] = df["_rel"].map(lambda r: str(image_dir / r))
    return df


def vessel_mask_path(rel: str, vessel_dir: Path) -> Path:
    """RRWNet writes one PNG per input image, keeping the input filename."""
    return Path(vessel_dir) / Path(rel).name


def prepare_fold(
    cfg, fold: int, vessel_dir: str | Path | None, project_root: Path = PROJECT_ROOT
) -> pd.DataFrame:
    df = image_paths(held_out_rows(cfg, fold, project_root), cfg, project_root)
    if vessel_dir is not None:
        vd = _resolve(vessel_dir, project_root)
        df["_vessel_path"] = df["_rel"].map(lambda r: str(vessel_mask_path(r, vd)))
    df["_label"] = df[cfg.label_col].astype(float)
    df["_patient_id"] = df[cfg.patient_id_col].astype(str)
    df["_fold"] = fold
    return df
