"""
config.schema — TrainConfig dataclass for a single training run.

Every hyperparameter lives here. No hyperparameters live in code.
A resolved config is saved alongside every checkpoint so any run
can be reproduced exactly.

Usage:
    config = TrainConfig(
        dataset="cimt",
        architecture="efficientnet",
        model_kwargs={"variant": "b2", "pretrained": True},
        fold=0,
        ...
    )

    # The architecture + model_kwargs map directly to the factory:
    model = create_model(config.architecture, **config.model_kwargs)

    # The model then declares its own preprocessing:
    transforms = build_transforms(model.preprocessing_config, is_train=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """Complete specification for a single training run (1 model × 1 fold)."""

    dataset: str
    architecture: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    fold: int = 0                       # 0–4 for 5-fold CV

    # Paths - Relative to project root unless absolute. SLURM env_setup.sh
    # exports PROJECT_ROOT and DATA_DIR; configs can use these.
    data_dir: str = ""                  # root directory containing images or if we're doing this another way need to discuss this
    labels_csv: str = ""                # path to labels CSV
    splits_file: str = ""               # path to saved fold splits JSON
    output_dir: str = "outputs"         # base output directory

    # Column mapping
    sample_id_col: str = "image_id"
    label_col: str = "label"
    patient_id_col: str = "patient_id"

    # Phase 1: Linear Probe
    # Frozen backbone, head-only training. Quick screening.
    phase1_epochs: int = 5
    phase1_lr: float = 1e-3
    phase1_batch_size: int = 64

    # Phase 2: Fine-Tune
    # Full model training with layer-wise LR decay and early stopping.
    phase2_epochs: int = 50
    phase2_lr: float = 1e-4
    phase2_batch_size: int = 32
    lr_decay_factor: float = 0.65       # layer-wise: lr_i = base_lr * (factor ^ (n_layers - i))
    warmup_epochs: int = 5              # linear warmup before cosine decay
    patience: int = 7                   # early stopping on val AUROC
    weight_decay: float = 0.05
    min_lr: float = 1e-6                # floor for cosine schedule

    # Loss
    use_weighted_bce: bool = False
    pos_weight: float | None = None

    # Metadata Fusion
    # When enabled, the trainer replaces the model's head with one
    # that accepts (feature_dim + metadata_dim) inputs. The backbone
    # and get_features() are untouched.
    use_metadata: bool = False
    metadata_columns: list[str] = field(default_factory=list)
    metadata_head_hidden: int = 0

    # Data Loading
    num_workers: int = 6
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Resumption
    # Critical for Ibex where SLURM jobs may be killed before completion.
    # The job template auto-detects last.pt and passes this flag.
    checkpoint_path: str | None = None
    skip_phase1: bool = False

    # Augmentation
    aug_hflip: bool = True
    aug_rotation: float = 15.0
    aug_color_jitter: float = 0.1
    aug_random_erasing: float = 0.0

    # Logging
    use_wandb: bool = False
    wandb_project: str = "retinexis"
    wandb_entity: str | None = None

    def __post_init__(self):
        """Coerce fields to correct types (guards against string values
        from CLI overrides or YAML parsing edge cases)."""
        self.fold = int(self.fold)
        self.phase1_epochs = int(self.phase1_epochs)
        self.phase1_lr = float(self.phase1_lr)
        self.phase1_batch_size = int(self.phase1_batch_size)
        self.phase2_epochs = int(self.phase2_epochs)
        self.phase2_lr = float(self.phase2_lr)
        self.phase2_batch_size = int(self.phase2_batch_size)
        self.lr_decay_factor = float(self.lr_decay_factor)
        self.warmup_epochs = int(self.warmup_epochs)
        self.patience = int(self.patience)
        self.weight_decay = float(self.weight_decay)
        self.min_lr = float(self.min_lr)
        self.num_workers = int(self.num_workers)
        self.seed = int(self.seed)
        self.aug_rotation = float(self.aug_rotation)
        self.aug_color_jitter = float(self.aug_color_jitter)
        self.aug_random_erasing = float(self.aug_random_erasing)

    # Derived / Computed
    @property
    def run_dir(self) -> Path:
        """Deterministic output path: {output_dir}/{dataset}/{architecture}/fold_{fold}"""
        return Path(self.output_dir) / self.dataset / self.architecture / f"fold_{self.fold}"

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    # Serialisation
    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for YAML serialisation."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save resolved config as YAML alongside checkpoints."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        """Construct from a plain dict (e.g. loaded from YAML).

        Rejects unknown keys to catch typos in config files.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d.keys()) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown config keys: {unknown}. "
                f"Valid keys: {sorted(valid_fields)}"
            )
        return cls(**d)

    def validate(self) -> None:
        """Run basic sanity checks. Call after construction.

        Does NOT validate model_kwargs — that's the factory's job.
        Does NOT validate paths exist — that happens at runtime when
        the trainer tries to load data, since paths may be relative
        to PROJECT_ROOT which is set by the SLURM environment.
        """
        if not self.dataset:
            raise ValueError("dataset must not be empty")

        if not self.architecture:
            raise ValueError("architecture must not be empty")

        if not 0 <= self.fold <= 4:
            raise ValueError(f"fold must be 0–4, got {self.fold}")

        if self.phase1_epochs < 0 or self.phase2_epochs < 0:
            raise ValueError("epoch counts must be non-negative")

        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")

        if self.use_metadata and not self.metadata_columns:
            raise ValueError("use_metadata=True but metadata_columns is empty")

        if not self.use_metadata and self.metadata_columns:
            raise ValueError(
                "metadata_columns is set but use_metadata=False. "
                "Set use_metadata=True or clear metadata_columns."
            )

        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be 0–6 (Ibex limit: 6 CPUs per GPU), got {self.num_workers}"
            )
