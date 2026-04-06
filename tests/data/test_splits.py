from __future__ import annotations

import pandas as pd
import pytest

from src.data.splits import generate_splits, load_splits, save_splits, validate_splits


def _sample_df() -> pd.DataFrame:
    rows = []
    for patient_idx in range(10):
        patient_id = f"patient_{patient_idx}"
        label = patient_idx % 2
        for image_idx in range(2):
            rows.append(
                {
                    "image_id": f"{patient_id}_image_{image_idx}",
                    "patient_id": patient_id,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_generate_splits_builds_expected_metadata_and_folds():
    df = _sample_df()

    split_dict = generate_splits(
        df,
        group_col="patient_id",
        label_col="label",
        n_splits=5,
        seed=123,
        sample_id_col="image_id",
        task_name="cimt",
    )

    assert split_dict["task_name"] == "cimt"
    assert split_dict["n_splits"] == 5
    assert split_dict["n_rows"] == len(df)
    assert split_dict["n_patients"] == df["patient_id"].nunique()
    assert len(split_dict["folds"]) == 5
    assert all(len(fold["val_patient_ids"]) == 2 for fold in split_dict["folds"])


def test_generate_splits_rejects_missing_columns():
    df = _sample_df().drop(columns=["label"])

    with pytest.raises(ValueError, match="Missing required columns"):
        generate_splits(df, group_col="patient_id", label_col="label")


def test_validate_splits_accepts_generated_grouped_splits():
    df = _sample_df()
    split_dict = generate_splits(
        df,
        group_col="patient_id",
        label_col="label",
        n_splits=5,
        seed=123,
        sample_id_col="image_id",
    )

    summary = validate_splits(split_dict, df, tolerance=0.6)

    assert summary["ok"] is True
    assert summary["errors"] == []
    assert len(summary["folds"]) == 5


def test_validate_splits_detects_patient_leakage():
    df = _sample_df()
    split_dict = generate_splits(
        df,
        group_col="patient_id",
        label_col="label",
        n_splits=5,
        seed=123,
        sample_id_col="image_id",
    )

    leaked_sample = split_dict["folds"][0]["val_sample_ids"][0]
    split_dict["folds"][0]["train_sample_ids"].append(leaked_sample)

    summary = validate_splits(split_dict, df, tolerance=0.6)

    assert summary["ok"] is False
    assert any("patient leakage detected" in error for error in summary["errors"])


def test_save_and_load_splits_round_trip(tmp_path):
    df = _sample_df()
    split_dict = generate_splits(
        df,
        group_col="patient_id",
        label_col="label",
        n_splits=5,
        seed=321,
        sample_id_col="image_id",
    )

    path = tmp_path / "splits" / "folds.json"
    save_splits(split_dict, path)

    loaded = load_splits(path)

    assert loaded == split_dict
