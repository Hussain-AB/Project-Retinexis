from .efficientnet import EfficientNetClassifier
from .efficientnet_vessel import EfficientNetVesselClassifier
from .dinov2 import DINOv2Classifier
from .dinov3 import DINOv3Classifier
from .resnet import ResNetClassifier
from .seresnext import SEResNeXtClassifier
from .convnextv2 import ConvNeXtV2Classifier
from .retfound_dinov2 import RETFoundDINOv2
from .retfound_green import RETFoundGreenClassifier
from .retfound_mae import RETFoundMAEClassifier
from .swanV2 import SWANv2Classifier

MODEL_REGISTRY: dict[str, type] = {
    "efficientnet": EfficientNetClassifier,
    "efficientnet_vessel": EfficientNetVesselClassifier,
    "dinov2": DINOv2Classifier,
    "dinov3": DINOv3Classifier,
    "swinv2": SWANv2Classifier,
    "swinV2": SWANv2Classifier,
    "resnet": ResNetClassifier,
    "seresnext": SEResNeXtClassifier,
    "convnextv2": ConvNeXtV2Classifier,
    "retfound_dinov2": RETFoundDINOv2,
    "retfound_green": RETFoundGreenClassifier,
    "retfound_mae": RETFoundMAEClassifier,
}
