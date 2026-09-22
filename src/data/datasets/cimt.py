from pathlib import Path
from PIL import Image
import torch

from src.data.base_dataset import BaseRetinalDataset, SampleDict


class CIMTDataset(BaseRetinalDataset):
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> SampleDict:
        row = self.df.iloc[idx]

        filename = row.get("filename") or row.get("image_filename")
        image_path = self.image_dir / filename
        image = Image.open(image_path).convert("RGB")

        vessel_path = None
        if self.vessel_mask_dir is not None:
            vessel_path = self.vessel_mask_dir / filename
        image_t, vessel_t = self._transform_pair(image, vessel_path)

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
