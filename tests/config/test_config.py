from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.loader import _cast_override, _expand_env_vars, load_config
from src.config.schema import TrainConfig


def _base_config(output_dir: Path | None = None) -> dict:
    config = {
        "dataset": "cimt",
        "architecture": "efficientnet",
        "fold": 0,
        "phase1_epochs": 5,
        "use_wandb": False,
    }
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    return config


def test_expand_env_vars_recurses_through_nested_values(monkeypatch, tmp_path):
    monkeypatch.setenv("RETINEXIS_TEST_ROOT", str(tmp_path))

    value = {
        "data_dir": "$RETINEXIS_TEST_ROOT/data",
        "nested": ["${RETINEXIS_TEST_ROOT}/images", {"path": "$RETINEXIS_TEST_ROOT/logs"}],
    }

    expanded = _expand_env_vars(value)

    assert expanded["data_dir"] == str(tmp_path / "data")
    assert expanded["nested"][0] == str(tmp_path / "images")
    assert expanded["nested"][1]["path"] == str(tmp_path / "logs")


@pytest.mark.parametrize(
    ("key", "value", "reference", "expected"),
    [
        ("phase1_epochs", "7", {"phase1_epochs": 5}, 7),
        ("phase2_lr", "0.001", {"phase2_lr": 1e-4}, 0.001),
        ("use_wandb", "true", {"use_wandb": False}, True),
        ("architecture", "resnet", {"architecture": "efficientnet"}, "resnet"),
    ],
)
def test_cast_override_matches_reference_type(key, value, reference, expected):
    assert _cast_override(key, value, reference) == expected


def test_load_config_expands_env_vars_and_applies_typed_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("RETINEXIS_DATA_ROOT", str(tmp_path / "dataset_root"))

    config_path = tmp_path / "config.yaml"
    raw = _base_config()
    raw["data_dir"] = "$RETINEXIS_DATA_ROOT/images"
    raw["phase2_lr"] = 1e-4

    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(
        config_path,
        overrides={
            "fold": "2",
            "phase1_epochs": "9",
            "use_wandb": "true",
            "phase2_lr": "0.002",
        },
    )

    assert isinstance(config, TrainConfig)
    assert config.data_dir == str(tmp_path / "dataset_root" / "images")
    assert config.fold == 2
    assert config.phase1_epochs == 9
    assert config.use_wandb is True
    assert config.phase2_lr == pytest.approx(0.002)


def test_load_config_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing)


def test_train_config_from_dict_rejects_unknown_keys():
    raw = _base_config()
    raw["not_a_real_field"] = 123

    with pytest.raises(ValueError, match="Unknown config keys"):
        TrainConfig.from_dict(raw)


def test_train_config_validate_rejects_invalid_metadata_combinations():
    missing_columns = TrainConfig(
        dataset="cimt",
        architecture="efficientnet",
        use_metadata=True,
        metadata_columns=[],
    )

    with pytest.raises(ValueError, match="metadata_columns is empty"):
        missing_columns.validate()

    unexpected_columns = TrainConfig(
        dataset="cimt",
        architecture="efficientnet",
        use_metadata=False,
        metadata_columns=["age"],
    )

    with pytest.raises(ValueError, match="metadata_columns is set but use_metadata=False"):
        unexpected_columns.validate()


def test_train_config_validate_rejects_invalid_fold_and_patience():
    invalid_fold = TrainConfig(dataset="cimt", architecture="efficientnet", fold=6)

    with pytest.raises(ValueError, match="fold must be 0–4"):
        invalid_fold.validate()

    invalid_patience = TrainConfig(dataset="cimt", architecture="efficientnet", patience=0)

    with pytest.raises(ValueError, match="patience must be >= 1"):
        invalid_patience.validate()


def test_train_config_path_properties_and_save_roundtrip(tmp_path):
    config = TrainConfig(
        dataset="cimt",
        architecture="efficientnet",
        fold=3,
        output_dir=str(tmp_path / "outputs"),
    )

    assert config.run_dir == tmp_path / "outputs" / "cimt" / "efficientnet" / "fold_3"
    assert config.checkpoint_dir == config.run_dir / "checkpoints"
    assert config.predictions_dir == config.run_dir / "predictions"
    assert config.log_dir == config.run_dir / "logs"

    save_path = tmp_path / "resolved" / "config.yaml"
    config.save(save_path)

    saved = yaml.safe_load(save_path.read_text(encoding="utf-8"))

    assert saved["dataset"] == "cimt"
    assert saved["architecture"] == "efficientnet"
    assert saved["fold"] == 3
