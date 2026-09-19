from __future__ import annotations

from importlib import import_module

from .base import BaseClassifier, PreprocessingConfig

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EfficientNetClassifier": (".efficientnet", "EfficientNetClassifier"),
    "DINOv2Classifier": (".dinov2", "DINOv2Classifier"),
    "DINOv3Classifier": (".dinov3", "DINOv3Classifier"),
    "ResNetClassifier": (".resnet", "ResNetClassifier"),
    "SEResNeXtClassifier": (".seresnext", "SEResNeXtClassifier"),
    "ConvNeXtV2Classifier": (".convnextv2", "ConvNeXtV2Classifier"),
    "RETFoundDINOv2": (".retfound_dinov2", "RETFoundDINOv2"),
    "RETFoundMAEClassifier": (".retfound_mae", "RETFoundMAEClassifier"),
    "create_model": (".factory", "create_model"),
    "MODEL_REGISTRY": (".registry", "MODEL_REGISTRY"),
}

__all__ = [
    "BaseClassifier",
    "PreprocessingConfig",
    "EfficientNetClassifier",
    "DINOv2Classifier",
    "DINOv3Classifier",
    "ResNetClassifier",
    "SEResNeXtClassifier",
    "ConvNeXtV2Classifier",
    "RETFoundDINOv2",
    "RETFoundMAEClassifier",
    "create_model",
    "MODEL_REGISTRY",
]


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
