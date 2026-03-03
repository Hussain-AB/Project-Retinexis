from src.data.loader import DATASET_REGISTRY
from src.data.datasets.cimt import CIMTDataset

DATASET_REGISTRY["cimt"] = CIMTDataset
