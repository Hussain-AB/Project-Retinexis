"""
config.loader — YAML config loading with CLI overrides.

Loads a YAML file, expands environment variables in string values,
applies CLI overrides, and returns a validated TrainConfig.

Usage:
    config = load_config("path/to/config.yaml", overrides={"fold": 2})
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from src.config.schema import TrainConfig


def _expand_env_vars(value: Any) -> Any:
    """Expand $USER, $HOME, ${VAR} in string values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _cast_override(key: str, value: str, reference: dict) -> Any:
    """Cast a CLI override string to the type of the existing config value."""
    if key in reference and reference[key] is not None:
        target_type = type(reference[key])
        if target_type is bool:
            return value.lower() in ("true", "1", "yes")
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
    # Fall through: keep as string
    return value


def load_config(
    config_path: str | Path,
    overrides: dict | None = None,
) -> TrainConfig:
    """Load YAML config, apply env expansion and CLI overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: Dict of key-value overrides from CLI.

    Returns:
        Validated TrainConfig instance.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Expand environment variables ($USER, $HOME, etc.)
    raw = _expand_env_vars(raw)

    # Apply CLI overrides
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, str):
                raw[key] = _cast_override(key, value, raw)
            else:
                raw[key] = value

    # Build TrainConfig — this validates all fields
    config = TrainConfig.from_dict(raw)

    return config
