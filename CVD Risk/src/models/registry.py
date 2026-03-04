from .efficientnet import EfficientNetClassifier
from .dinov2 import DINOv2Classifier
from .dinov3 import DINOv3Classifier
from .resnet import ResNetClassifier
from .seresnext import SEResNeXtClassifier
from .convnextv2 import ConvNeXtV2Classifier
from .retfound_green import RETFoundGreenClassifier

MODEL_REGISTRY: dict[str, type] = {
    "efficientnet": EfficientNetClassifier,
    "dinov2": DINOv2Classifier,
    "dinov3": DINOv3Classifier,
    "resnet": ResNetClassifier,
    "seresnext": SEResNeXtClassifier,
    "convnextv2": ConvNeXtV2Classifier,
    "retfound_green": RETFoundGreenClassifier,
}
