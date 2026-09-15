from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# Splits schema version.
#   1 = train/val only (pre-ISBI).
#   2 = train/inner_val/val, where `val` keeps its original meaning (the
#       held-out outer fold that gets reported) and `train` is now the outer
#       training set MINUS the inner-validation slice.
SPLITS_VERSION = 2


def _to_path(path_like: Union[str, Path]) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _strat_labels(work: pd.DataFrame, label_col: str, strat_extra_col: Optional[str]) -> pd.Series:
    """Stratification target: the label, optionally crossed with a second column.

    Crossing with `source_dataset` keeps each fold's cohort mix stable. Without
    it, StratifiedGroupKFold balances only the label, so the BRSET/ODIR ratio
    drifts fold to fold and adds variance that has nothing to do with the model.
    """
    strat = work[label_col].astype(str)
    if strat_extra_col:
        strat = strat + "_" + work[strat_extra_col].astype(str)
    return strat


def _inner_split(
    outer_train: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    inner_val_frac: float,
    seed: int,
    n_tries: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `inner_val_frac` of the outer-training PATIENTS as inner validation.

    GroupShuffleSplit is group-aware but not stratified, so the inner split's
    positive rate can drift. We draw `n_tries` candidate splits and keep the one
    whose inner-val positive rate is closest to the outer-training rate. This is
    a deterministic function of `seed`.
    """
    target = float(outer_train[label_col].mean())
    best = None
    best_gap = None

    for t in range(n_tries):
        gss = GroupShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=seed + t)
        in_idx, iv_idx = next(gss.split(outer_train, groups=outer_train[group_col]))
        inner_train = outer_train.iloc[in_idx]
        inner_val = outer_train.iloc[iv_idx]
        if len(inner_val) == 0 or len(inner_train) == 0:
            continue
        gap = abs(float(inner_val[label_col].mean()) - target)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = (inner_train, inner_val)

    if best is None:
        raise ValueError(
            f"Could not build an inner validation split at frac={inner_val_frac} "
            f"from {outer_train[group_col].nunique()} patients."
        )
    return best


def generate_splits(
    df: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    n_splits: int = 5,
    seed: int = 42,
    sample_id_col: str = "image_id",
    task_name: str = "brset",
    inner_val_frac: float = 0.15,
    strat_extra_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate patient-grouped, stratified K-fold splits and return a JSON-serialisable dict.

    Each fold carries THREE disjoint sample-ID lists:

        val_sample_ids        the held-out outer fold. This is what gets reported.
                              Same meaning as in schema version 1.
        inner_val_sample_ids  a patient-disjoint slice of the outer training set,
                              used for early stopping, best-checkpoint selection,
                              temperature fitting and threshold choice.
        train_sample_ids      the remainder of the outer training set.

    Splitting the outer training set rather than the outer fold is what makes the
    reported number unbiased: the outer fold is never used to choose anything.

    Set `inner_val_frac=0.0` to reproduce version-1 behaviour exactly (no inner
    split; `inner_val_sample_ids` is written as an empty list).

    Grouping is by `group_col` (e.g. patient_id); stratification by `label_col`,
    optionally crossed with `strat_extra_col` (e.g. source_dataset).
    """
    required = {group_col, label_col, sample_id_col}
    if strat_extra_col:
        required.add(strat_extra_col)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not 0.0 <= inner_val_frac < 1.0:
        raise ValueError(f"inner_val_frac must be in [0, 1), got {inner_val_frac}")

    cols = [sample_id_col, group_col, label_col] + ([strat_extra_col] if strat_extra_col else [])
    work = df[cols].copy()

    # Clean / validate target
    work = work.dropna(subset=[group_col, label_col, sample_id_col]).copy()
    # Cast binary labels robustly
    work[label_col] = pd.to_numeric(work[label_col], errors="raise").astype(int)

    uniq_labels = sorted(work[label_col].unique().tolist())
    if len(uniq_labels) != 2:
        raise ValueError(
            f"`{label_col}` must be binary for stratification. Found values: {uniq_labels}"
        )

    # Duplicate sample IDs silently inflate every split and every metric.
    dupes = int(work[sample_id_col].duplicated().sum())
    if dupes:
        raise ValueError(
            f"`{sample_id_col}` contains {dupes} duplicate values. De-duplicate the "
            f"labels CSV before generating splits (see scripts/fix_combined_metadata.py)."
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X = work[[sample_id_col]]
    y = _strat_labels(work, label_col, strat_extra_col)
    groups = work[group_col]

    folds: List[Dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
        outer_train = work.iloc[train_idx]
        val_df = work.iloc[val_idx]

        if inner_val_frac > 0.0:
            train_df, inner_val_df = _inner_split(
                outer_train,
                group_col=group_col,
                label_col=label_col,
                inner_val_frac=inner_val_frac,
                seed=seed + 1000 * fold_idx,
            )
        else:
            train_df = outer_train
            inner_val_df = outer_train.iloc[0:0]

        folds.append(
            {
                "fold": fold_idx,
                "train_sample_ids": train_df[sample_id_col].astype(str).tolist(),
                "inner_val_sample_ids": inner_val_df[sample_id_col].astype(str).tolist(),
                "val_sample_ids": val_df[sample_id_col].astype(str).tolist(),
                "train_patient_ids": sorted(train_df[group_col].astype(str).unique().tolist()),
                "inner_val_patient_ids": sorted(inner_val_df[group_col].astype(str).unique().tolist()),
                "val_patient_ids": sorted(val_df[group_col].astype(str).unique().tolist()),
            }
        )

    split_dict: Dict[str, Any] = {
        "version": SPLITS_VERSION,
        "task_name": task_name,
        "n_splits": n_splits,
        "seed": seed,
        "group_col": group_col,
        "label_col": label_col,
        "sample_id_col": sample_id_col,
        "inner_val_frac": inner_val_frac,
        "strat_extra_col": strat_extra_col,
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
    inner_tolerance: float = 0.05,
    strat_extra_col: Optional[str] = None,
    strat_tolerance: float = 0.02,
) -> Dict[str, Any]:
    """
    Validate no patient leakage and report fold balance.

    Checks, in order:
      1. every split sample ID exists in the dataframe
      2. train / inner_val / val are pairwise patient-disjoint within a fold
      3. no patient and no sample appears in more than one fold's val split
      4. val positive rate is within `tolerance` of overall
      5. inner_val positive rate is within `inner_tolerance` of the fold's train rate
      6. if `strat_extra_col` is given, each val fold's cohort mix is within
         `strat_tolerance` of the overall mix

    tolerance: allowed absolute deviation in positive rate from overall.
    """
    group_col = group_col or split_dict["group_col"]
    label_col = label_col or split_dict["label_col"]
    sample_id_col = sample_id_col or split_dict["sample_id_col"]
    strat_extra_col = strat_extra_col or split_dict.get("strat_extra_col")

    required = {group_col, label_col, sample_id_col}
    if strat_extra_col:
        required.add(strat_extra_col)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cols = [sample_id_col, group_col, label_col] + ([strat_extra_col] if strat_extra_col else [])
    work = df[cols].copy()
    work = work.dropna(subset=[group_col, label_col, sample_id_col]).copy()
    work[label_col] = pd.to_numeric(work[label_col], errors="raise").astype(int)
    work[sample_id_col] = work[sample_id_col].astype(str)
    work[group_col] = work[group_col].astype(str)

    sample_to_patient = dict(zip(work[sample_id_col], work[group_col]))
    sample_to_label = dict(zip(work[sample_id_col], work[label_col]))
    sample_to_strat = (
        dict(zip(work[sample_id_col], work[strat_extra_col].astype(str)))
        if strat_extra_col else {}
    )

    overall_pos_rate = float(work[label_col].mean())
    overall_mix = (
        work[strat_extra_col].astype(str).value_counts(normalize=True).to_dict()
        if strat_extra_col else {}
    )

    all_val_patients = set()
    all_val_samples = set()

    fold_summaries: List[Dict[str, Any]] = []
    errors: List[str] = []

    for fold in split_dict["folds"]:
        fidx = int(fold["fold"])
        train_samples = [str(x) for x in fold["train_sample_ids"]]
        inner_samples = [str(x) for x in fold.get("inner_val_sample_ids", [])]
        val_samples = [str(x) for x in fold["val_sample_ids"]]

        # 1. Missing samples check
        for name, samples in (("train", train_samples), ("inner_val", inner_samples), ("val", val_samples)):
            unknown = [s for s in samples if s not in sample_to_patient]
            if unknown:
                errors.append(f"Fold {fidx}: {len(unknown)} {name} sample IDs not found in dataframe.")

        train_patients = {sample_to_patient[s] for s in train_samples if s in sample_to_patient}
        inner_patients = {sample_to_patient[s] for s in inner_samples if s in sample_to_patient}
        val_patients = {sample_to_patient[s] for s in val_samples if s in sample_to_patient}

        # 2. Pairwise patient disjointness within the fold
        for a_name, a, b_name, b in (
            ("train", train_patients, "val", val_patients),
            ("train", train_patients, "inner_val", inner_patients),
            ("inner_val", inner_patients, "val", val_patients),
        ):
            overlap = a & b
            if overlap:
                errors.append(
                    f"Fold {fidx}: patient leakage between {a_name} and {b_name} "
                    f"({len(overlap)} overlapping patients)."
                )

        # samples must not repeat across the three lists either
        counts = len(train_samples) + len(inner_samples) + len(val_samples)
        if counts != len(set(train_samples) | set(inner_samples) | set(val_samples)):
            errors.append(f"Fold {fidx}: a sample ID appears in more than one split.")

        # 3. Cross-fold repetition of the reported split
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
        inner_labels = [sample_to_label[s] for s in inner_samples if s in sample_to_label]

        val_pos_rate = float(sum(val_labels) / len(val_labels)) if val_labels else None
        train_pos_rate = float(sum(train_labels) / len(train_labels)) if train_labels else None
        inner_pos_rate = float(sum(inner_labels) / len(inner_labels)) if inner_labels else None

        # 4. Reported-split balance
        if val_pos_rate is not None and abs(val_pos_rate - overall_pos_rate) > tolerance:
            errors.append(
                f"Fold {fidx}: val positive rate {val_pos_rate:.4f} differs from overall {overall_pos_rate:.4f} "
                f"by more than tolerance {tolerance:.2f}."
            )

        # 5. Inner-val balance against the fold's own training rate
        if inner_pos_rate is not None and train_pos_rate is not None:
            if abs(inner_pos_rate - train_pos_rate) > inner_tolerance:
                errors.append(
                    f"Fold {fidx}: inner_val positive rate {inner_pos_rate:.4f} differs from "
                    f"train {train_pos_rate:.4f} by more than {inner_tolerance:.2f}."
                )

        # 6. Cohort mix of the reported split
        fold_mix = {}
        if strat_extra_col and val_samples:
            vals = [sample_to_strat[s] for s in val_samples if s in sample_to_strat]
            if vals:
                s = pd.Series(vals).value_counts(normalize=True)
                fold_mix = s.to_dict()
                for key, overall_frac in overall_mix.items():
                    got = float(fold_mix.get(key, 0.0))
                    if abs(got - float(overall_frac)) > strat_tolerance:
                        errors.append(
                            f"Fold {fidx}: val '{strat_extra_col}={key}' fraction {got:.4f} differs "
                            f"from overall {float(overall_frac):.4f} by more than {strat_tolerance:.2f}."
                        )

        fold_summaries.append(
            {
                "fold": fidx,
                "n_train_samples": len(train_samples),
                "n_inner_val_samples": len(inner_samples),
                "n_val_samples": len(val_samples),
                "n_train_patients": len(train_patients),
                "n_inner_val_patients": len(inner_patients),
                "n_val_patients": len(val_patients),
                "train_pos_rate": train_pos_rate,
                "inner_val_pos_rate": inner_pos_rate,
                "val_pos_rate": val_pos_rate,
                "val_mix": fold_mix,
            }
        )

    summary = {
        "ok": len(errors) == 0,
        "overall": {
            "n_rows": int(len(work)),
            "n_patients": int(work[group_col].nunique()),
            "positive_rate": overall_pos_rate,
            "mix": overall_mix,
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
