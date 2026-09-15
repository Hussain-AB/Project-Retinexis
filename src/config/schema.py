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
    # Origin-cohort column, written to every predictions row so within-site
    # metrics need no positional join. "source_dataset" for hypertension;
    # None for single-cohort tasks like CIMT, which fall back to config.dataset.
    source_col: str | None = None

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
    patience: int = 7                   # early stopping on inner-val AUROC
    weight_decay: float = 0.05
    min_lr: float = 1e-6                # floor for cosine schedule
    min_delta: float = 0.001            # improvement needed to reset patience
                                        # (was hardcoded in trainer.py)
    # Horizon the cosine schedule anneals over. phase2_epochs is the hard CAP;
    # early stopping normally fires at 10-33. Building the cosine for 50 epochs
    # meant the LR never left 97% of base and no model ever annealed. Set this
    # to the realistic budget instead. None => fall back to phase2_epochs
    # (exactly the old behaviour).
    scheduler_total_epochs: int | None = 30

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

    # Nested cross-validation
    # Fraction of TRAINING patients held out as inner validation, used for early
    # stopping, best-checkpoint selection, temperature fitting and threshold
    # choice. 0.0 reproduces the old two-way behaviour, in which the reported
    # fold was also the selection fold and the reported AUROC was therefore
    # optimistically biased. Consumed by the split generators, not at runtime.
    inner_val_frac: float = 0.15

    # Raise (rather than warn) when split IDs are absent from the labels CSV.
    strict_missing_ids: bool = True

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Resumption
    # Critical for Ibex where SLURM jobs may be killed before completion.
    # The job template auto-detects last.pt and passes this flag.
    checkpoint_path: str | None = None
    resume_weights_only: bool = False
    skip_phase1: bool = False

    # Augmentation
    aug_hflip: bool = True
    aug_rotation: float = 10.0
    aug_scale_min: float = 0.98
    aug_scale_max: float = 1.02
    aug_brightness: float = 0.05
    aug_contrast: float = 0.05
    aug_saturation: float = 0.0
    aug_hue: float = 0.0
    aug_blur_prob: float = 0.4
    aug_blur_limit: int = 3
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
        self.min_delta = float(self.min_delta)
        self.inner_val_frac = float(self.inner_val_frac)
        if self.scheduler_total_epochs is not None:
            self.scheduler_total_epochs = int(self.scheduler_total_epochs)
        self.num_workers = int(self.num_workers)
        self.seed = int(self.seed)
        self.aug_rotation = float(self.aug_rotation)
        self.aug_scale_min = float(self.aug_scale_min)
        self.aug_scale_max = float(self.aug_scale_max)
        self.aug_brightness = float(self.aug_brightness)
        self.aug_contrast = float(self.aug_contrast)
        self.aug_saturation = float(self.aug_saturation)
        self.aug_hue = float(self.aug_hue)
        self.aug_blur_prob = float(self.aug_blur_prob)
        self.aug_blur_limit = int(self.aug_blur_limit)
        self.aug_random_erasing = float(self.aug_random_erasing)

    # Derived / Computed
    # The reference seed. Runs at this seed keep the historical directory layout
    # so existing paths, scripts and the xai module are unaffected.
    REFERENCE_SEED = 42

    @property
    def run_dir(self) -> Path:
        """Deterministic output path.

            seed == 42 : {output_dir}/{dataset}/{architecture}/fold_{fold}
            otherwise  : {output_dir}/{dataset}/{architecture}/fold_{fold}_seed{seed}

        The seed MUST be part of the path. Without it a second seed writes to the
        same directory and overwrites the first run's best.pt -- which would
        silently destroy the seed study that the whole variance analysis rests on.
        """
        tag = f"fold_{self.fold}"
        if int(self.seed) != self.REFERENCE_SEED:
            tag = f"{tag}_seed{int(self.seed)}"
        return Path(self.output_dir) / self.dataset / self.architecture / tag

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def val_predictions_path(self) -> Path:
        """Per-image predictions on the held-out outer fold. The reported file.

        Name unchanged from the pre-ISBI pipeline so aggregate_folds.py,
        calibrate.py, summarize_all_models.py and xai/check_frame.py keep working.
        """
        return self.predictions_dir / "val_predictions.csv"

    @property
    def inner_val_predictions_path(self) -> Path:
        """Per-image predictions on the inner-validation split.

        Fit temperature and choose the operating threshold here, never on
        val_predictions.csv -- that is what makes the reported calibration and
        operating point honest.
        """
        return self.predictions_dir / "inner_val_predictions.csv"

    @property
    def manifest_path(self) -> Path:
        """Provenance record written at the end of every run."""
        return self.run_dir / "run_manifest.json"

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

        if not 0.0 <= self.inner_val_frac < 1.0:
            raise ValueError(
                f"inner_val_frac must be in [0, 1), got {self.inner_val_frac}"
            )

        if self.scheduler_total_epochs is not None:
            if not 1 <= self.scheduler_total_epochs <= max(1, self.phase2_epochs):
                raise ValueError(
                    f"scheduler_total_epochs must be between 1 and phase2_epochs "
                    f"({self.phase2_epochs}), got {self.scheduler_total_epochs}"
                )

        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")

        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be 0–6 (Ibex limit: 6 CPUs per GPU), got {self.num_workers}"
            )
