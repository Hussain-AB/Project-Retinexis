from src.data.loader import DATASET_REGISTRY
from src.data.datasets.cimt import CIMTDataset
from src.data.datasets.hypertension import HypertensionDataset

DATASET_REGISTRY["cimt"] = CIMTDataset
DATASET_REGISTRY["hypertension_combined"] = HypertensionDataset
