"""
data.loader — DataLoader factory.

Single integration point between config, model, splits, and datasets.
The trainer calls create_dataloaders() once at init and never touches
datasets or transforms directly.

Usage:
    train_loader, val_loader, encoder = create_dataloaders(
        config, model, batch_size=config.phase1_batch_size)

    # Phase 2: swap batch size without re-reading CSV/splits
    train_loader, val_loader = rebuild_loaders(
        train_loader, val_loader,
        batch_size=config.phase2_batch_size, config=config)
"""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.base_dataset import BaseRetinalDataset, MetadataEncoder
# build_vessel_transforms is imported lazily at its single call site below.
# It is needed only when use_vessel_masks=True, and a top-level import of a
# vessel-only helper once took down every job in the grid -- EfficientNet-B2
# on CIMT does not load masks, yet could not import this module.
from src.data.transforms import build_transforms
from src.models.base import BaseClassifier


#  DATASET REGISTRY
# Parallel to MODEL_REGISTRY. Each team registers their class at
# import time in src/data/datasets/__init__.py:
#
#   from src.data.loader import DATASET_REGISTRY
#   from src.data.datasets.cimt import CIMTDataset
#   DATASET_REGISTRY["cimt"] = CIMTDataset

DATASET_REGISTRY: dict[str, type[BaseRetinalDataset]] = {}


def _seed_worker(worker_id: int) -> None:
    """Seed Python's `random` and NumPy inside each DataLoader worker.

    PyTorch seeds only its own RNG per worker. src/data/transforms.py augments
    with Python's `random` module (RandomScale uses random.uniform, RandomBlur
    uses random.random), and NumPy is used elsewhere. On Linux the workers fork
    with an identical `random` state, so without this every worker produced the
    SAME augmentation sequence -- cutting effective augmentation diversity by
    num_workers, and leaving augmentation randomness outside the control of
    config.seed. That matters directly for the seed-variance study: sigma_seed
    has to measure training stochasticity we actually control.
    """
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)

#  MAIN API
def create_dataloaders(
    config: Any,
    model: BaseClassifier,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, MetadataEncoder | None]:
    """Build train, inner-validation and val DataLoaders from config and model.

    Handles: CSV loading, splits subsetting, transform construction,
    MetadataEncoder fitting, dataset construction, DataLoader wrapping.

    THREE loaders, not two:
        train      what the optimiser sees
        inner_val  early stopping, best-checkpoint selection, temperature
                   fitting, threshold choice. Patient-disjoint from both others.
        val        the held-out outer fold. Scored ONCE, at the end. Same
                   meaning as before, so val_predictions.csv is unchanged.

    If the splits file carries no inner_val_sample_ids (schema version 1, or
    inner_val_frac=0), inner_val falls back to the val loader and a warning is
    emitted -- that reproduces the old behaviour exactly, including its
    selection bias.

    Args:
        config: TrainConfig instance.
        model: Model instance (used for preprocessing_config only).
        batch_size: Explicit batch size. Defaults to phase1_batch_size.

    Returns:
        (train_loader, inner_val_loader, val_loader, metadata_encoder_or_None)
    """
    if batch_size is None:
        batch_size = config.phase1_batch_size

    # Validate dataset class exists
    if config.dataset not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys()) if DATASET_REGISTRY else ["(none registered)"]
        raise ValueError(
            f"Unknown dataset '{config.dataset}'. "
            f"Registered: {available}. "
            f"Check src/data/datasets/__init__.py imports."
        )
    DatasetClass = DATASET_REGISTRY[config.dataset]

    # Load labels CSV
    df = pd.read_csv(config.labels_csv)

    # Load splits and extract this fold
    train_ids, inner_val_ids, val_ids, splits_meta = _load_fold(config.splits_file, config.fold)

    # sample_id_col: authoritative source is splits metadata,
    # fall back to config if not present in the JSON
    sample_id_col = splits_meta.get("sample_id_col", config.sample_id_col)

    if sample_id_col not in df.columns:
        raise ValueError(
            f"sample_id_col '{sample_id_col}' not found in {config.labels_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # Subset DataFrame by sample IDs
    strict = getattr(config, "strict_missing_ids", True)
    train_df = _subset_df(df, train_ids, sample_id_col, split_name="train", strict=strict)
    val_df = _subset_df(df, val_ids, sample_id_col, split_name="val", strict=strict)
    if inner_val_ids:
        inner_val_df = _subset_df(
            df, inner_val_ids, sample_id_col, split_name="inner_val", strict=strict)
    else:
        warnings.warn(
            "Splits file has no inner_val_sample_ids: falling back to the val fold "
            "for early stopping and checkpoint selection. This reproduces the "
            "pre-ISBI behaviour, in which the reported split is also the selection "
            "split, so the reported AUROC is optimistically biased. Regenerate the "
            "splits with inner_val_frac > 0 to remove that bias."
        )
        inner_val_df = val_df

    # Build transforms from model's preprocessing config
    train_transform = build_transforms(
        model.preprocessing_config, is_train=True, config=config)
    val_transform = build_transforms(
        model.preprocessing_config, is_train=False, config=config)

    # Fit MetadataEncoder on train split (if enabled)
    encoder = None
    if config.use_metadata and config.metadata_columns:
        encoder = MetadataEncoder(config.metadata_columns)
        encoder.fit(train_df)

    # Build vessel mask transform (if enabled)
    vessel_mask_dir = None
    vessel_transform = None
    if getattr(config, "use_vessel_masks", False):
        vessel_cfg = getattr(model, "vessel_preprocessing_config", None)
        if vessel_cfg is None:
            raise ValueError(
                f"use_vessel_masks=True but model '{type(model).__name__}' has no "
                f"vessel_preprocessing_config property. Use architecture=efficientnet_vessel."
            )
        vessel_mask_dir = config.vessel_mask_dir
        from src.data.transforms import build_vessel_transforms
        vessel_transform = build_vessel_transforms(vessel_cfg)

    # Construct dataset instances
    train_dataset = DatasetClass(
        df=train_df,
        image_dir=config.data_dir,
        transform=train_transform,
        label_col=config.label_col,
        patient_id_col=config.patient_id_col,
        sample_id_col=sample_id_col,
        source_col=getattr(config, "source_col", None),
        source_default=config.dataset,
        metadata_encoder=encoder,
        vessel_mask_dir=vessel_mask_dir,
        vessel_transform=vessel_transform,
    )
    inner_val_dataset = DatasetClass(
        df=inner_val_df,
        image_dir=config.data_dir,
        transform=val_transform,
        label_col=config.label_col,
        patient_id_col=config.patient_id_col,
        sample_id_col=sample_id_col,
        source_col=getattr(config, "source_col", None),
        source_default=config.dataset,
        metadata_encoder=encoder,
        vessel_mask_dir=vessel_mask_dir,
        vessel_transform=vessel_transform,
    )
    val_dataset = DatasetClass(
        df=val_df,
        image_dir=config.data_dir,
        transform=val_transform,
        label_col=config.label_col,
        patient_id_col=config.patient_id_col,
        sample_id_col=sample_id_col,
        source_col=getattr(config, "source_col", None),
        source_default=config.dataset,
        metadata_encoder=encoder,
        vessel_mask_dir=vessel_mask_dir,
        vessel_transform=vessel_transform,
    )

    # Wrap in DataLoaders
    # persistent_workers requires num_workers > 0
    persistent = config.persistent_workers and config.num_workers > 0

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": persistent,
        "prefetch_factor": config.prefetch_factor if config.num_workers > 0 else None,
    }

    # Deterministic shuffling, and workers whose Python/NumPy RNGs are seeded.
    generator = torch.Generator()
    generator.manual_seed(int(getattr(config, "seed", 42)))

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True,
        worker_init_fn=_seed_worker, generator=generator, **loader_kwargs)
    inner_val_loader = DataLoader(
        inner_val_dataset, shuffle=False, drop_last=False,
        worker_init_fn=_seed_worker, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False,
        worker_init_fn=_seed_worker, **loader_kwargs)

    return train_loader, inner_val_loader, val_loader, encoder


def rebuild_loaders(
    train_loader: DataLoader,
    inner_val_loader: DataLoader,
    val_loader: DataLoader,
    batch_size: int,
    config: Any,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Rebuild DataLoaders with a new batch size, reusing existing datasets.

    Called when switching from Phase 1 (batch_size=64) to Phase 2
    (batch_size=32). Avoids re-reading CSV, re-parsing splits, and
    re-fitting MetadataEncoder.

    Re-applies worker_init_fn and the seeded generator. Omitting them here
    would silently drop the worker-RNG fix for the whole of Phase 2, which is
    where nearly all training happens.

    Args:
        train_loader: Existing train DataLoader (dataset is reused).
        inner_val_loader: Existing inner-validation DataLoader (dataset reused).
        val_loader: Existing val DataLoader (dataset is reused).
        batch_size: New batch size.
        config: TrainConfig (for num_workers, pin_memory, etc.).

    Returns:
        (new_train_loader, new_inner_val_loader, new_val_loader)
    """
    persistent = config.persistent_workers and config.num_workers > 0

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": persistent,
        "prefetch_factor": config.prefetch_factor if config.num_workers > 0 else None,
    }

    generator = torch.Generator()
    generator.manual_seed(int(getattr(config, "seed", 42)))

    new_train = DataLoader(
        train_loader.dataset, shuffle=True, drop_last=True,
        worker_init_fn=_seed_worker, generator=generator, **loader_kwargs)
    new_inner_val = DataLoader(
        inner_val_loader.dataset, shuffle=False, drop_last=False,
        worker_init_fn=_seed_worker, **loader_kwargs)
    new_val = DataLoader(
        val_loader.dataset, shuffle=False, drop_last=False,
        worker_init_fn=_seed_worker, **loader_kwargs)

    return new_train, new_inner_val, new_val

#  INTERNAL HELPERS
def _load_fold(
    splits_file: str | Path,
    fold: int,
) -> tuple[set[str], set[str], set[str], dict]:
    """Load a splits JSON and extract one fold's sample IDs.

    `inner_val_sample_ids` is optional: schema version 1 files do not have it,
    and version 2 files written with inner_val_frac=0 carry an empty list. In
    both cases an empty set is returned and the caller falls back to the val
    split (with a warning).

    Returns:
        (train_ids, inner_val_ids, val_ids, splits_metadata)
    """
    path = Path(splits_file)
    if not path.exists():
        raise FileNotFoundError(f"Splits file not found: {path}")

    with open(path) as f:
        splits = json.load(f)

    n_folds = len(splits["folds"])
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"Fold {fold} out of range (splits has {n_folds} folds)")

    fold_data = splits["folds"][fold]

    # Cast to str — splits store IDs as strings, but defensive
    train_ids = set(str(sid) for sid in fold_data["train_sample_ids"])
    val_ids = set(str(sid) for sid in fold_data["val_sample_ids"])
    inner_val_ids = set(str(sid) for sid in fold_data.get("inner_val_sample_ids", []))

    # A sample must never be in two splits of the same fold.
    for a_name, a_set, b_name, b_set in (
        ("train", train_ids, "val", val_ids),
        ("train", train_ids, "inner_val", inner_val_ids),
        ("inner_val", inner_val_ids, "val", val_ids),
    ):
        overlap = a_set & b_set
        if overlap:
            raise ValueError(
                f"Fold {fold}: {len(overlap)} sample IDs appear in both "
                f"'{a_name}' and '{b_name}' in {path}."
            )

    # Metadata from splits (sample_id_col, group_col, label_col, etc.)
    metadata = {k: v for k, v in splits.items() if k != "folds"}

    return train_ids, inner_val_ids, val_ids, metadata


def _subset_df(
    df: pd.DataFrame,
    sample_ids: set[str],
    sample_id_col: str,
    split_name: str = "",
    strict: bool = True,
) -> pd.DataFrame:
    """Filter DataFrame to rows matching sample IDs.

    Two failure modes this guards against, both of which used to pass silently:

    1. IDs in the splits that are absent from the CSV -- a column-name mismatch
       or a stale splits file. Previously a warning, so a run could quietly
       train on a fraction of its data.
    2. More rows than IDs -- duplicate sample IDs in the CSV. The hypertension
       metadata had 6,029 duplicate image_id rows, which inflated every split by
       22% and narrowed every confidence interval; this assertion would have
       caught it on the first run.

    Set strict=False (config.strict_missing_ids) to downgrade case 1 to a
    warning. Case 2 always raises: silently duplicated samples are never
    intended.
    """
    df_ids = df[sample_id_col].astype(str)
    mask = df_ids.isin(sample_ids)
    subset = df[mask].copy()

    # Check for missing IDs
    found_ids = set(df_ids[mask])
    missing = sample_ids - found_ids
    if missing:
        sample = sorted(missing)[:5]
        msg = (
            f"{split_name}: {len(missing)} of {len(sample_ids)} sample IDs from "
            f"splits not found in CSV column '{sample_id_col}'. First 5: {sample}. "
            f"Check that sample_id_col matches between splits and labels CSV."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)

    if len(subset) != len(found_ids):
        raise ValueError(
            f"{split_name}: {len(subset)} rows matched {len(found_ids)} distinct "
            f"sample IDs in column '{sample_id_col}'. The labels CSV contains "
            f"duplicate sample IDs; de-duplicate it first "
            f"(scripts/fix_combined_metadata.py)."
        )

    return subset
