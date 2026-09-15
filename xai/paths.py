"""
xai.paths — repo layout and the model roster.

Deliberately free of torch and of `src.*` imports, so figures.py, metrics.py
and analyze.py can run anywhere the CSVs are (a laptop, a notebook) without
the training environment installed.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Display key -> phase-2 config basename. The config file carries the
# architecture and model_kwargs, so no model detail is hard-coded here.
MODEL_KEYS: dict[str, str] = {
    "retfound_dinov2": "retfound_dinov2_phase2.yaml",
    "retfound_mae": "retfound_mae_phase2.yaml",
    "dinov2_large": "dinov2_large_phase2.yaml",
    "dinov3_large": "dinov3_large_phase2.yaml",
    "convnextv2": "convnextv2_phase2.yaml",
    "efficientnet_b2": "efficientnet_b2_phase2.yaml",
}

# The comparison axes the paper actually argues about: transformer vs
# convolutional, and retina-specific vs generic pretraining.
MODEL_FAMILY: dict[str, str] = {
    "retfound_dinov2": "vit-retinal",
    "retfound_mae": "vit-retinal",
    "dinov2_large": "vit-generic",
    "dinov3_large": "vit-generic",
    "convnextv2": "cnn-generic",
    "efficientnet_b2": "cnn-generic",
}

TASK_DIR = {"cimt": "cimt", "hypertension": "hypertension"}


def config_path(task: str, model_key: str, project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / "tasks" / TASK_DIR[task] / "configs" / "phase2" / MODEL_KEYS[model_key]
