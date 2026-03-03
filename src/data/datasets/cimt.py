from pathlib import Path
from PIL import Image
import torch

from src.data.base_dataset import BaseRetinalDataset, SampleDict


class CIMTDataset(BaseRetinalDataset):
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> SampleDict:
        row = self.df.iloc[idx]

        image_path = self.image_dir / row["image_path"]  # change column name as needed
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        sample: SampleDict = {
            "image": image,
            "label": torch.tensor([float(row[self.label_col])], dtype=torch.float32),
            "patient_id": str(row[self.patient_id_col]),
            "image_path": str(image_path),
        }

        if self.metadata_encoder is not None:
            sample["metadata"] = self.metadata_encoder.encode(row)

        return sample
