"""
data.base_dataset — Abstract base class and return contract for all datasets.

Both CIMT and Hypertension inherit from this. The trainer and
DataLoader factory only interact with the base interface, so they never
need to know which dataset they're working with.

Return contract:
    __getitem__ always returns a SampleDict (TypedDict):

    Image-only mode (use_metadata=False):
        {
            "image":      Tensor (3, H, W)    — transformed image
            "label":      Tensor (1,)         — binary label, float for BCEWithLogitsLoss
            "patient_id": str                 — for patient-grouped CV and prediction saving
            "image_path": str                 — for debugging and error tracing
        }

    Metadata mode (use_metadata=True):
        {
            "image":      Tensor (3, H, W)
            "metadata":   Tensor (D,)         — encoded metadata vector
            "label":      Tensor (1,)
            "patient_id": str
            "image_path": str
        }

    Using a dict (not a positional tuple) because:
    - The trainer can check `if "metadata" in batch` instead of
      branching on tuple length
    - Adding fields later (e.g. laterality, exam_date) won't break
      existing unpacking code
    - Custom collate is not needed — default_collate handles dicts
      of tensors and strings natively

Integration with model pipeline:
    - Transforms are built from model.preprocessing_config via
      build_transforms() in transforms.py — datasets never hardcode
      image sizes or normalisation.
    - The trainer calls model(batch["image"]) in image-only mode, or
      model.get_features(batch["image"]) + concat batch["metadata"]
      in metadata mode. The dataset doesn't know or care about this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


# Return type
class SampleDict(TypedDict, total=False):
    """Return type for __getitem__. 'metadata' and 'vessel_mask' are optional."""
    image: torch.Tensor         # (3, H, W)
    vessel_mask: torch.Tensor   # (3, H, W) — only present when use_vessel_masks=True
    metadata: torch.Tensor      # (D,) — only present when use_metadata=True
    label: torch.Tensor         # (1,)
    sample_id: str              # image-level ID; written to every predictions CSV
    patient_id: str
    source_dataset: str         # origin cohort, e.g. "brset" / "odir" / "cimt"
    image_path: str


# Metadata encoding
class MetadataEncoder:
    """Encodes metadata columns into a flat tensor.

    Fitted on the training fold only. Applied identically to train and val
    splits (and external cohorts at inference). Saved alongside checkpoints
    so inference can reconstruct the same encoding.

    Usage:
        # Fit on training data
        encoder = MetadataEncoder(metadata_columns=["age", "gender"])
        encoder.fit(train_df)

        # Pass to both train and val datasets
        train_dataset = CIMTDataset(..., metadata_encoder=encoder)
        val_dataset   = CIMTDataset(..., metadata_encoder=encoder)

        # At inference / checkpoint loading
        encoder = MetadataEncoder.from_dict(checkpoint["metadata_encoder"])
    """

    def __init__(self, metadata_columns: list[str]) -> None:
        self.metadata_columns = metadata_columns
        self._fitted = False

        self._continuous_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._category_maps: dict[str, dict[str, int]] = {}
        self._output_dim: int = 0

    @property
    def output_dim(self) -> int:
        """Total dimensionality of the encoded metadata vector."""
        if not self._fitted:
            raise RuntimeError("MetadataEncoder has not been fitted yet.")
        return self._output_dim

    def fit(self, df: pd.DataFrame) -> MetadataEncoder:
        """Learn encoding parameters from training data.

        Continuous columns (numeric dtype): z-score normalisation.
        Categorical columns (object/category dtype): one-hot encoding.

        Args:
            df: Training fold DataFrame. Must contain all metadata_columns.

        Returns:
            self, for chaining.
        """
        for col in self.metadata_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Metadata column '{col}' not found in DataFrame. "
                    f"Available: {list(df.columns)}"
                )

        dim = 0
        for col in self.metadata_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self._continuous_cols.append(col)
                self._means[col] = float(df[col].mean())
                std = float(df[col].std())
                self._stds[col] = std if std > 1e-8 else 1.0  # guard against zero-variance
                dim += 1
            else:
                self._categorical_cols.append(col)
                categories = sorted(df[col].dropna().unique().tolist())
                self._category_maps[col] = {cat: i for i, cat in enumerate(categories)}
                dim += len(categories)

        self._output_dim = dim
        self._fitted = True
        return self

    def encode(self, row: pd.Series) -> torch.Tensor:
        """Encode a single row's metadata columns into a flat tensor.

        Args:
            row: A single row from the labels DataFrame.

        Returns:
            Tensor of shape (output_dim,).
        """
        if not self._fitted:
            raise RuntimeError("MetadataEncoder has not been fitted yet.")

        parts: list[float] = []

        for col in self._continuous_cols:
            val = float(row[col]) if pd.notna(row[col]) else 0.0
            parts.append((val - self._means[col]) / self._stds[col])

        for col in self._categorical_cols:
            cat_map = self._category_maps[col]
            one_hot = [0.0] * len(cat_map)
            val = row[col]
            if pd.notna(val) and val in cat_map:
                one_hot[cat_map[val]] = 1.0
            parts.extend(one_hot)

        return torch.tensor(parts, dtype=torch.float32)

    def to_dict(self) -> dict:
        """Serialise for checkpoint saving."""
        return {
            "metadata_columns": self.metadata_columns,
            "continuous_cols": self._continuous_cols,
            "categorical_cols": self._categorical_cols,
            "means": self._means,
            "stds": self._stds,
            "category_maps": self._category_maps,
            "output_dim": self._output_dim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MetadataEncoder:
        """Reconstruct from checkpoint dict."""
        encoder = cls(metadata_columns=d["metadata_columns"])
        encoder._continuous_cols = d["continuous_cols"]
        encoder._categorical_cols = d["categorical_cols"]
        encoder._means = d["means"]
        encoder._stds = d["stds"]
        encoder._category_maps = d["category_maps"]
        encoder._output_dim = d["output_dim"]
        encoder._fitted = True
        return encoder


# Abstract base dataset
class BaseRetinalDataset(Dataset, ABC):
    """Abstract base class for retinal image datasets.

    Subclasses (CIMTDataset, BRSETDataset) must implement:
        - __getitem__: load image, apply transform, return SampleDict
        - __len__: return number of samples

    Subclasses may override (but get working defaults from config column names):
        - patient_ids: returns list of all patient IDs (default reads patient_id_col)
        - labels: returns numpy array of all labels (default reads label_col)

    What the subclass does NOT need to handle:
        - Fold splitting (handled by splits.py, which stores sample IDs)
        - Fold subsetting (handled by loader.py, which filters the DataFrame
          by matching sample IDs before passing it here)
        - Transform construction (handled by transforms.py from model's PreprocessingConfig)
        - DataLoader wrapping (handled by loader.py)
        - Metadata encoding (handled by MetadataEncoder, passed in at construction)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: callable,
        label_col: str = "label",
        patient_id_col: str = "patient_id",
        sample_id_col: str = "image_id",
        source_col: str | None = None,
        source_default: str = "unknown",
        metadata_encoder: MetadataEncoder | None = None,
        vessel_mask_dir: str | None = None,
        vessel_transform: callable | None = None,
        paired_transform: callable | None = None,
        vessel_source: str = "mask",
        vessel_perm: object | None = None,
    ) -> None:
        """
        Args:
            df: DataFrame for this split (already subsetted by sample IDs
                from the splits JSON via the loader). Must contain at minimum:
                an image path/filename column, a label column, and a patient
                ID column.
            image_dir: Root directory containing images.
            transform: Callable (e.g. torchvision.transforms.Compose) to
                apply to each PIL Image. Built from model.preprocessing_config
                by transforms.py.
            label_col: Name of the binary label column in df. Read from
                config.label_col by the loader and passed through here.
            patient_id_col: Name of the patient ID column in df. Read from
                config.patient_id_col by the loader and passed through here.
            metadata_encoder: Fitted MetadataEncoder, or None if metadata
                is disabled. When provided, __getitem__ must include
                "metadata" in the returned SampleDict.
            vessel_mask_dir: Directory containing pre-generated RRWNet vessel
                masks. Filenames must match the image filenames. When set,
                __getitem__ must include "vessel_mask" in the returned SampleDict.
            vessel_transform: Transform to apply to vessel mask PIL Images.
                Built from model.vessel_preprocessing_config by transforms.py.
                Required when vessel_mask_dir is set.
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_col = label_col
        self.patient_id_col = patient_id_col
        self.sample_id_col = sample_id_col
        self.source_col = source_col
        self.source_default = source_default
        self.metadata_encoder = metadata_encoder
        self.vessel_mask_dir = Path(vessel_mask_dir) if vessel_mask_dir else None
        self.vessel_transform = vessel_transform
        self.paired_transform = paired_transform
        self.vessel_source = vessel_source
        self.vessel_perm = vessel_perm

        # Validate required columns exist
        if self.label_col not in self.df.columns:
            raise ValueError(
                f"label_col '{self.label_col}' not found in DataFrame. "
                f"Available: {list(self.df.columns)}"
            )
        if self.patient_id_col not in self.df.columns:
            raise ValueError(
                f"patient_id_col '{self.patient_id_col}' not found in DataFrame. "
                f"Available: {list(self.df.columns)}"
            )
        # sample_id is what links a prediction row back to an image, an
        # attribution map and a cohort. Without it val_predictions.csv cannot be
        # joined to anything.
        if self.sample_id_col not in self.df.columns:
            raise ValueError(
                f"sample_id_col '{self.sample_id_col}' not found in DataFrame. "
                f"Available: {list(self.df.columns)}"
            )
        if self.source_col is not None and self.source_col not in self.df.columns:
            raise ValueError(
                f"source_col '{self.source_col}' not found in DataFrame. "
                f"Available: {list(self.df.columns)}"
            )

    def _mask_relpath(self, row) -> str | None:
        """Path of this row's vessel mask, relative to vessel_mask_dir.

        Overridden per dataset because the column differs. Returning None
        disables vessel loading for that dataset.
        """
        return None

    def _vessel_companion(self, idx: int, image, row):
        """The PIL image the vessel branch receives, or None.

        Which one depends on vessel_source, and that is the whole point of the
        control: the model, its parameter count and its training are identical
        in all three cases, so a difference in score can only come from what
        this function returns.
        """
        if self.vessel_mask_dir is None:
            return None

        if self.vessel_source == "image":
            # Parameter-matched control: a second copy of the fundus. It goes
            # through the vessel branch's own preprocessing, exactly as a mask
            # would, so the branch itself is unchanged.
            return image.copy()

        if self.vessel_source == "shuffled":
            # Another patient's mask, from the same split, chosen by a seeded
            # derangement fixed at construction -- so it is identical on every
            # epoch and reproducible from the run's seed, and never this
            # image's own mask.
            if self.vessel_perm is None:
                raise ValueError(
                    "vessel_source='shuffled' but no permutation was supplied; "
                    "create_dataloaders builds it."
                )
            donor = self.df.iloc[int(self.vessel_perm[idx])]
            rel = self._mask_relpath(donor)
        else:
            rel = self._mask_relpath(row)

        if rel is None:
            return None
        return Image.open(self.vessel_mask_dir / rel).convert("RGB")

    def _transform_pair(self, image, mask_path=None):
        """Apply the image and vessel-mask transforms under ONE geometry.

        When paired_transform is set (vessel runs), the image and the mask
        receive the same sampled hflip / rotation / scale. Without it they were
        transformed by independent pipelines with independent draws, so a
        mirrored image was paired with an unmirrored mask roughly half the
        time -- the mask then described vessels on the wrong side of the retina.

        Falls back to the independent path when no mask is involved, so
        image-only runs are bit-identical to before.
        """
        def _as_image(src):
            if src is None or isinstance(src, Image.Image):
                return src
            return Image.open(src).convert("RGB")

        if self.paired_transform is not None:
            return self.paired_transform(image, _as_image(mask_path))

        image_t = self.transform(image)
        mask_t = None
        if mask_path is not None and self.vessel_transform is not None:
            mask_t = self.vessel_transform(_as_image(mask_path))
        return image_t, mask_t

    def _identity_fields(self, row) -> dict:
        """sample_id / patient_id / source_dataset for one row.

        Shared by every subclass so the predictions CSV schema is identical
        across datasets.
        """
        return {
            "sample_id": str(row[self.sample_id_col]),
            "patient_id": str(row[self.patient_id_col]),
            "source_dataset": (
                str(row[self.source_col]) if self.source_col else self.source_default
            ),
        }

    @abstractmethod
    def __getitem__(self, idx: int) -> SampleDict:
        """Load and return a single sample.

        Subclass implementation pattern:
            1. row = self.df.iloc[idx]
            2. image_path = self.image_dir / row["your_filename_col"]
            3. image = load as RGB PIL Image
            4. image = self.transform(image)
            5. label = torch.tensor([float(row[self.label_col])], dtype=torch.float32)
            6. sample = {"image": image, "label": label,
                         "patient_id": str(row[self.patient_id_col]),
                         "image_path": str(image_path)}
            7. if self.metadata_encoder is not None:
                   sample["metadata"] = self.metadata_encoder.encode(row)
            8. return sample

        The subclass owns:
            - Which column contains the image filename
            - How to construct the full image path (may vary by dataset)
            - Any dataset-specific image loading quirks

        The subclass gets for free via self.*:
            - self.label_col — which column has the label
            - self.patient_id_col — which column has the patient ID
            - self.metadata_encoder — encode(row) if not None
            - self.image_dir — root image directory
            - self.transform — the transform to apply
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    def patient_ids(self) -> list[str]:
        """All patient IDs in this dataset split.

        Used by validate_splits() to check for patient leakage across folds.
        Default reads from patient_id_col. Override if your dataset has
        a different structure.
        """
        return self.df[self.patient_id_col].astype(str).tolist()

    @property
    def labels(self) -> np.ndarray:
        """All labels as a 1D numpy array of 0s and 1s.

        Used by the trainer for computing pos_weight when use_weighted_bce=True.
        Default reads from label_col. Override if your dataset needs
        custom label extraction.
        """
        return self.df[self.label_col].values.astype(int)
