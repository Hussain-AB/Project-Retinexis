from pathlib import Path
from PIL import Image
import torch

from src.data.base_dataset import BaseRetinalDataset, SampleDict


class CIMTDataset(BaseRetinalDataset):
    def __len__(self) -> int:
        return len(self.df)

    def _mask_relpath(self, row) -> str | None:
        name = row.get("filename") or row.get("image_filename")
        return str(name) if name else None

    def __getitem__(self, idx: int) -> SampleDict:
        row = self.df.iloc[idx]

        filename = row.get("filename") or row.get("image_filename")
        image_path = self.image_dir / filename
        image = Image.open(image_path).convert("RGB")

        companion = self._vessel_companion(idx, image, row)
        image_t, vessel_t = self._transform_pair(image, companion)

        sample: SampleDict = {
            "image": image_t,
            "label": torch.tensor([float(row[self.label_col])], dtype=torch.float32),
            "image_path": str(image_path),
            **self._identity_fields(row),
        }

        if vessel_t is not None:
            sample["vessel_mask"] = vessel_t

        if self.metadata_encoder is not None:
            sample["metadata"] = self.metadata_encoder.encode(row)

        return sample
