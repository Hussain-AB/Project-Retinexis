from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _to_path(path_like: Union[str, Path]) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def generate_splits(
    df: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    n_splits: int = 5,
    seed: int = 42,
    sample_id_col: str = "image_id",
    task_name: str = "brset",
) -> Dict[str, Any]:
    """
    Generate patient-grouped, stratified K-fold splits and return a JSON-serialisable dict.

    The output stores row-level sample IDs for train/val plus summary metadata.
    Grouping is done by `group_col` (e.g., patient_id), stratification by `label_col`.
    """
    required = {group_col, label_col, sample_id_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[[sample_id_col, group_col, label_col]].copy()

    # Clean / validate target
    work = work.dropna(subset=[group_col, label_col, sample_id_col]).copy()
    # Cast binary labels robustly
    work[label_col] = pd.to_numeric(work[label_col], errors="raise").astype(int)

    uniq_labels = sorted(work[label_col].unique().tolist())
    if len(uniq_labels) != 2:
        raise ValueError(
            f"`{label_col}` must be binary for stratification. Found values: {uniq_labels}"
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X = work[[sample_id_col]]
    y = work[label_col]
    groups = work[group_col]

    folds: List[Dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
        train_df = work.iloc[train_idx]
        val_df = work.iloc[val_idx]

        folds.append(
            {
                "fold": fold_idx,
                "train_sample_ids": train_df[sample_id_col].astype(str).tolist(),
                "val_sample_ids": val_df[sample_id_col].astype(str).tolist(),
                "train_patient_ids": sorted(train_df[group_col].astype(str).unique().tolist()),
                "val_patient_ids": sorted(val_df[group_col].astype(str).unique().tolist()),
            }
        )

    split_dict: Dict[str, Any] = {
        "version": 1,
        "task_name": task_name,
        "n_splits": n_splits,
        "seed": seed,
        "group_col": group_col,
        "label_col": label_col,
        "sample_id_col": sample_id_col,
        "n_rows": int(len(work)),
        "n_patients": int(work[group_col].nunique()),
        "folds": folds,
    }
    return split_dict


def validate_splits(
    split_dict: Dict[str, Any],
    df: pd.DataFrame,
    *,
    group_col: Optional[str] = None,
    label_col: Optional[str] = None,
    sample_id_col: Optional[str] = None,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Validate no patient leakage and report fold balance.
    tolerance: allowed absolute deviation in positive rate from overall.
    """
    group_col = group_col or split_dict["group_col"]
    label_col = label_col or split_dict["label_col"]
    sample_id_col = sample_id_col or split_dict["sample_id_col"]

    required = {group_col, label_col, sample_id_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[[sample_id_col, group_col, label_col]].copy()
    work = work.dropna(subset=[group_col, label_col, sample_id_col]).copy()
    work[label_col] = pd.to_numeric(work[label_col], errors="raise").astype(int)
    work[sample_id_col] = work[sample_id_col].astype(str)
    work[group_col] = work[group_col].astype(str)

    sample_to_patient = dict(zip(work[sample_id_col], work[group_col]))
    sample_to_label = dict(zip(work[sample_id_col], work[label_col]))

    overall_pos_rate = float(work[label_col].mean())
    all_val_patients = set()
    all_val_samples = set()

    fold_summaries: List[Dict[str, Any]] = []
    errors: List[str] = []

    for fold in split_dict["folds"]:
        fidx = int(fold["fold"])
        train_samples = [str(x) for x in fold["train_sample_ids"]]
        val_samples = [str(x) for x in fold["val_sample_ids"]]

        # Missing samples check
        missing_train = [s for s in train_samples if s not in sample_to_patient]
        missing_val = [s for s in val_samples if s not in sample_to_patient]
        if missing_train:
            errors.append(f"Fold {fidx}: {len(missing_train)} train sample IDs not found in dataframe.")
        if missing_val:
            errors.append(f"Fold {fidx}: {len(missing_val)} val sample IDs not found in dataframe.")

        train_patients = {sample_to_patient[s] for s in train_samples if s in sample_to_patient}
        val_patients = {sample_to_patient[s] for s in val_samples if s in sample_to_patient}

        overlap = train_patients & val_patients
        if overlap:
            errors.append(f"Fold {fidx}: patient leakage detected ({len(overlap)} overlapping patients).")

        repeated_val_patients = all_val_patients & val_patients
        if repeated_val_patients:
            errors.append(f"Fold {fidx}: patients repeated across validation folds ({len(repeated_val_patients)}).")
        all_val_patients |= val_patients

        repeated_val_samples = all_val_samples & set(val_samples)
        if repeated_val_samples:
            errors.append(f"Fold {fidx}: samples repeated across validation folds ({len(repeated_val_samples)}).")
        all_val_samples |= set(val_samples)

        val_labels = [sample_to_label[s] for s in val_samples if s in sample_to_label]
        train_labels = [sample_to_label[s] for s in train_samples if s in sample_to_label]

        val_pos_rate = float(sum(val_labels) / len(val_labels)) if val_labels else None
        train_pos_rate = float(sum(train_labels) / len(train_labels)) if train_labels else None

        if val_pos_rate is not None and abs(val_pos_rate - overall_pos_rate) > tolerance:
            errors.append(
                f"Fold {fidx}: val positive rate {val_pos_rate:.4f} differs from overall {overall_pos_rate:.4f} "
                f"by more than tolerance {tolerance:.2f}."
            )

        fold_summaries.append(
            {
                "fold": fidx,
                "n_train_samples": len(train_samples),
                "n_val_samples": len(val_samples),
                "n_train_patients": len(train_patients),
                "n_val_patients": len(val_patients),
                "train_pos_rate": train_pos_rate,
                "val_pos_rate": val_pos_rate,
            }
        )

    summary = {
        "ok": len(errors) == 0,
        "overall": {
            "n_rows": int(len(work)),
            "n_patients": int(work[group_col].nunique()),
            "positive_rate": overall_pos_rate,
        },
        "folds": fold_summaries,
        "errors": errors,
    }
    return summary


def save_splits(split_dict: Dict[str, Any], path_like: Union[str, Path]) -> None:
    path = _to_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)


def load_splits(path_like: Union[str, Path]) -> Dict[str, Any]:
    path = _to_path(path_like)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
