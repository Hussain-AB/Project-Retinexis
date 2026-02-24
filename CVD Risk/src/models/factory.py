from __future__ import annotations

from .base import BaseClassifier
from .registry import MODEL_REGISTRY

_EXPECTED_KWARGS: dict[str, set[str]] = {
    "efficientnet": {
        "variant", "num_classes", "pretrained", "drop_rate", "input_size_override",
    },
    "dinov2": {
        "variant",
        "num_classes",
        "pretrained",
        "pool",
        "combine",
        "input_size_override",
        "linear_probe",
    },
}

def create_model(architecture: str, **kwargs) -> BaseClassifier:
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    expected = _EXPECTED_KWARGS[architecture]
    unknown = set(kwargs.keys()) - expected
    if unknown:
        raise ValueError(
            f"Unexpected kwargs for '{architecture}': {unknown}. "
            f"Expected: {expected}"
        )

    return MODEL_REGISTRY[architecture](**kwargs)
