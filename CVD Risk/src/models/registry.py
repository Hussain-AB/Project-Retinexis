from .efficientnet import EfficientNetClassifier
from .dinov2 import DINOv2Classifier

MODEL_REGISTRY: dict[str, type] = {
    "efficientnet": EfficientNetClassifier,
    "dinov2": DINOv2Classifier,
}
