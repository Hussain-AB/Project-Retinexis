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
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader

from src.data.base_dataset import BaseRetinalDataset, MetadataEncoder
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

#  MAIN API
def create_dataloaders(
    config: Any,
    model: BaseClassifier,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, MetadataEncoder | None]:
    """Build train and val DataLoaders from config and model.

    Handles: CSV loading, splits subsetting, transform construction,
    MetadataEncoder fitting, dataset construction, DataLoader wrapping.

    Args:
        config: TrainConfig instance.
        model: Model instance (used for preprocessing_config only).
        batch_size: Explicit batch size. Defaults to phase1_batch_size.

    Returns:
        (train_loader, val_loader, metadata_encoder_or_None)
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
    train_ids, val_ids, splits_meta = _load_fold(config.splits_file, config.fold)

    # sample_id_col: authoritative source is splits metadata,
    # fall back to config if not present in the JSON
    sample_id_col = splits_meta.get("sample_id_col", config.sample_id_col)

    if sample_id_col not in df.columns:
        raise ValueError(
            f"sample_id_col '{sample_id_col}' not found in {config.labels_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # Subset DataFrame by sample IDs
    train_df = _subset_df(df, train_ids, sample_id_col, split_name="train")
    val_df = _subset_df(df, val_ids, sample_id_col, split_name="val")

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

    # Construct dataset instances
    train_dataset = DatasetClass(
        df=train_df,
        image_dir=config.data_dir,
        transform=train_transform,
        label_col=config.label_col,
        patient_id_col=config.patient_id_col,
        metadata_encoder=encoder,
    )
    val_dataset = DatasetClass(
        df=val_df,
        image_dir=config.data_dir,
        transform=val_transform,
        label_col=config.label_col,
        patient_id_col=config.patient_id_col,
        metadata_encoder=encoder,
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

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, encoder


def rebuild_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_size: int,
    config: Any,
) -> tuple[DataLoader, DataLoader]:
    """Rebuild DataLoaders with a new batch size, reusing existing datasets.

    Called when switching from Phase 1 (batch_size=64) to Phase 2
    (batch_size=32). Avoids re-reading CSV, re-parsing splits, and
    re-fitting MetadataEncoder.

    Args:
        train_loader: Existing train DataLoader (dataset is reused).
        val_loader: Existing val DataLoader (dataset is reused).
        batch_size: New batch size.
        config: TrainConfig (for num_workers, pin_memory, etc.).

    Returns:
        (new_train_loader, new_val_loader)
    """
    persistent = config.persistent_workers and config.num_workers > 0

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": persistent,
        "prefetch_factor": config.prefetch_factor if config.num_workers > 0 else None,
    }

    new_train = DataLoader(
        train_loader.dataset, shuffle=True, drop_last=True, **loader_kwargs)
    new_val = DataLoader(
        val_loader.dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return new_train, new_val

#  INTERNAL HELPERS
def _load_fold(
    splits_file: str | Path,
    fold: int,
) -> tuple[set[str], set[str], dict]:
    """Load a splits JSON and extract one fold's sample IDs.

    Returns:
        (train_ids, val_ids, splits_metadata)
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

    # Metadata from splits (sample_id_col, group_col, label_col, etc.)
    metadata = {k: v for k, v in splits.items() if k != "folds"}

    return train_ids, val_ids, metadata


def _subset_df(
    df: pd.DataFrame,
    sample_ids: set[str],
    sample_id_col: str,
    split_name: str = "",
) -> pd.DataFrame:
    """Filter DataFrame to rows matching sample IDs.

    Warns if any sample IDs from the splits aren't found in the CSV
    (could indicate a column name mismatch or stale splits file).
    """
    df_ids = df[sample_id_col].astype(str)
    mask = df_ids.isin(sample_ids)
    subset = df[mask].copy()

    # Check for missing IDs
    found_ids = set(df_ids[mask])
    missing = sample_ids - found_ids
    if missing:
        sample = sorted(missing)[:5]
        warnings.warn(
            f"{split_name}: {len(missing)} sample IDs from splits not found "
            f"in CSV column '{sample_id_col}'. First 5: {sample}. "
            f"Check that sample_id_col matches between splits and labels CSV."
        )

    return subset
