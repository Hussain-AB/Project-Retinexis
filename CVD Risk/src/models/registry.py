from .efficientnet import EfficientNetClassifier

MODEL_REGISTRY: dict[str, type] = {
    "efficientnet": EfficientNetClassifier,
}
