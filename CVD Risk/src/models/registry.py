from .efficientnet import EfficientNetClassifier
from .dinov2 import DINOv2Classifier
from .resnet import ResNetClassifier
from .seresnext import SEResNeXtClassifier

MODEL_REGISTRY: dict[str, type] = {
    "efficientnet": EfficientNetClassifier,
    "dinov2": DINOv2Classifier,
    "resnet": ResNetClassifier,
    "seresnext": SEResNeXtClassifier,
}
