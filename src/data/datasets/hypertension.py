from pathlib import Path
from PIL import Image
import torch

from src.data.base_dataset import BaseRetinalDataset, SampleDict

class HypertensionDataset(BaseRetinalDataset):
    """
    Hypertension combined dataset (BRSET + ODIR).
    Expects metadata columns:
      - relative_image_path
      - label
      - patient_id
    """
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> SampleDict:
        row = self.df.iloc[idx]

        # Use the universally constructed relative path
        rel_path = row.get("relative_image_path")
        if not rel_path:
            raise KeyError("Column 'relative_image_path' not found in split DataFrame.")
            
        image_path = self.image_dir / rel_path
        
        # In case some images don't exist during dry runs or buggy symlinks, 
        # it will throw FileNotFoundError here cleanly.
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Base dataset guarantees self.label_col and self.patient_id_col exist and default to "label" and "patient_id"
        sample: SampleDict = {
            "image": image,
            "label": torch.tensor([float(row[self.label_col])], dtype=torch.float32),
            "image_path": str(image_path),
            **self._identity_fields(row),
        }

        # RRWNet masks mirror the image layout (brset/<f>.jpg, odir/<f>.jpg),
        # so the same relative path resolves under vessel_mask_dir.
        if self.vessel_mask_dir is not None and self.vessel_transform is not None:
            vessel_path = self.vessel_mask_dir / rel_path
            vessel_mask = Image.open(vessel_path).convert("RGB")
            sample["vessel_mask"] = self.vessel_transform(vessel_mask)

        if self.metadata_encoder is not None:
            sample["metadata"] = self.metadata_encoder.encode(row)

        return sample
