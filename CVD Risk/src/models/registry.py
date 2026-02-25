from .efficientnet import EfficientNetClassifier
from .dinov2 import DINOv2Classifier
from .resnet import ResNetClassifier
from .seresnext import SEResNeXtClassifier
from .convnextv2 import ConvNeXtV2Classifier
from .retfound_dinov2 import RETFoundDINOv2

MODEL_REGISTRY: dict[str, type] = {
    "efficientnet": EfficientNetClassifier,
    "dinov2": DINOv2Classifier,
    "resnet": ResNetClassifier,
    "seresnext": SEResNeXtClassifier,
    "convnextv2": ConvNeXtV2Classifier,
    "retfound_dinov2": RETFoundDINOv2,
}
