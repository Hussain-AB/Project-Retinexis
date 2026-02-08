"""
Standalone tests for EfficientNetClassifier.

This has no dependency on the factory, registry, or other model implementations.
When the full suite (test_models.py) is ready, this file becomes redundant. But
I will use it to derive the full test suite.

Run:
    pytest tests/test_efficientnet.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM
from src.models.efficientnet import EfficientNetClassifier, VARIANT_CONFIG


# helpers
INPUT_SIZES = {
    "b0": 224,
    "b2": 260,
    "b4": 380,
}


def _make_model(variant: str = "b2") -> EfficientNetClassifier:
    """Create an unpretrained EfficientNet for testing."""
    return EfficientNetClassifier(variant=variant, pretrained=False)


def _dummy_input(variant: str = "b2", batch_size: int = 2) -> torch.Tensor:
    size = INPUT_SIZES[variant]
    return torch.randn(batch_size, 3, size, size)


def _dummy_labels(batch_size: int = 2) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, 1)).float()


# Instantiation
class TestInstantiation:

    @pytest.mark.parametrize("variant", ["b0", "b2", "b4"])
    def test_all_variants_load(self, variant):
        model = _make_model(variant)
        assert isinstance(model, BaseClassifier)

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            _make_model("b99")

    def test_input_size_override(self):
        """input_size_override changes preprocessing config, not crash."""
        model = EfficientNetClassifier(
            variant="b2", pretrained=False, input_size_override=224
        )
        assert model.preprocessing_config.input_size == 224

    def test_default_input_size_is_native(self):
        """Without override, B2 uses its native 260."""
        model = _make_model("b2")
        assert model.preprocessing_config.input_size == 260

# Output shape
class TestOutputShape:

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_output_is_b1(self, batch_size):
        model = _make_model()
        model.eval()
        x = _dummy_input(batch_size=batch_size)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (batch_size, 1)

    @pytest.mark.parametrize("variant", ["b0", "b2", "b4"])
    def test_output_shape_all_variants(self, variant):
        model = _make_model(variant)
        model.eval()
        x = _dummy_input(variant=variant, batch_size=2)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 1)

    def test_output_is_raw_logits(self):
        """Output should be unbounded (not sigmoid-clamped to 0–1)."""
        model = _make_model()
        model.eval()
        x = torch.randn(8, 3, 260, 260) * 10
        with torch.no_grad():
            logits = model(x)
        assert logits.min() < 0 or logits.max() > 1, (
            "Logits suspiciously bounded — might be applying sigmoid"
        )

# Backward pass
class TestBackwardPass:

    def test_head_receives_finite_gradients(self):
        model = _make_model()
        model.train()
        x = _dummy_input()
        y = _dummy_labels()

        logits = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()

        head_grads = [
            p.grad for p in model.head.parameters()
            if p.grad is not None
        ]
        assert len(head_grads) > 0, "No gradients reached the head"
        assert all(torch.isfinite(g).all() for g in head_grads)

    def test_backbone_receives_gradients_in_finetune(self):
        """In finetune mode, backbone should also get gradients."""
        model = _make_model()
        model.set_finetune_mode()
        x = _dummy_input()
        y = _dummy_labels()

        logits = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()

        backbone_grads = [
            p.grad for p in model.backbone.parameters()
            if p.grad is not None
        ]
        assert len(backbone_grads) > 0, "No gradients reached the backbone"

    def test_backbone_no_gradients_when_frozen(self):
        """After freeze_backbone(), backbone params have no gradient."""
        model = _make_model()
        model.freeze_backbone()
        model.backbone.eval()
        model.head.train()
        x = _dummy_input()
        y = _dummy_labels()

        logits = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()

        for name, p in model.backbone.named_parameters():
            assert p.grad is None, f"Frozen param {name} received gradient"


# mode invariants — (linear probe lock)
def _assert_linear_probe_invariants(model: BaseClassifier) -> None:
    assert model.training is True, "model.training should be True"
    assert model.backbone.training is False, "backbone must be eval (locked)"
    assert model.head.training is True, "head must be train"


class TestModeInvariants:

    def test_linear_probe_sets_correct_state(self):
        model = _make_model()
        model.set_linear_probe_mode()
        _assert_linear_probe_invariants(model)

    def test_lock_survives_model_train(self):
        """model.train() must not revert backbone to train mode."""
        model = _make_model()
        model.set_linear_probe_mode()
        model.train()
        _assert_linear_probe_invariants(model)

    def test_lock_survives_eval_train_cycle(self):
        """eval() → train() must re-enter locked state."""
        model = _make_model()
        model.set_linear_probe_mode()
        model.eval()
        assert model.backbone.training is False
        assert model.head.training is False
        model.train()
        _assert_linear_probe_invariants(model)

    def test_finetune_releases_lock(self):
        model = _make_model()
        model.set_linear_probe_mode()
        model.set_finetune_mode()
        assert model.backbone.training is True
        assert model.head.training is True
        assert model._linear_probe_locked is False

    def test_freeze_counts(self):
        """Frozen backbone has 0 trainable params; head has >0."""
        model = _make_model()
        model.freeze_backbone()

        backbone_trainable = sum(
            1 for p in model.backbone.parameters() if p.requires_grad
        )
        head_trainable = sum(
            1 for p in model.head.parameters() if p.requires_grad
        )

        assert backbone_trainable == 0
        assert head_trainable > 0

    def test_param_counts_change_with_freeze(self):
        """num_trainable_params() should drop after freezing backbone."""
        model = _make_model()
        total_before = model.num_trainable_params()

        model.freeze_backbone()
        total_after = model.num_trainable_params()

        assert total_after < total_before

    def test_linear_probe_deterministic_backbone(self):
        """Backbone output is identical across two calls when locked."""
        model = _make_model()
        model.set_linear_probe_mode()
        x = _dummy_input()

        with torch.no_grad():
            f1 = model.get_features(x)
            f2 = model.get_features(x)

        assert torch.allclose(f1, f2)


# Feature extraction
class TestFeatureExtraction:

    def test_feature_dim_matches_output(self):
        model = _make_model()
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            features = model.get_features(x)
        assert features.shape == (2, model.feature_dim)

    def test_features_are_2d(self):
        model = _make_model()
        model.eval()
        x = _dummy_input()
        with torch.no_grad():
            features = model.get_features(x)
        assert features.dim() == 2

    @pytest.mark.parametrize("variant,expected_dim", [
        ("b0", 1280),
        ("b2", 1408),
        ("b4", 1792),
    ])
    def test_feature_dim_per_variant(self, variant, expected_dim):
        model = _make_model(variant)
        assert model.feature_dim == expected_dim


# Preprocessing config
class TestPreprocessing:

    def test_imagenet_normalisation(self):
        model = _make_model()
        config = model.preprocessing_config
        assert config.mean == IMAGENET_NORM["mean"]
        assert config.std == IMAGENET_NORM["std"]

    def test_pad_to_square(self):
        model = _make_model()
        assert model.preprocessing_config.crop_strategy == "pad_to_square"

    def test_config_is_frozen(self):
        model = _make_model()
        config = model.preprocessing_config
        with pytest.raises(AttributeError):
            config.input_size = 999

    @pytest.mark.parametrize("variant,expected_size", [
        ("b0", 224),
        ("b2", 260),
        ("b4", 380),
    ])
    def test_native_input_size(self, variant, expected_size):
        model = _make_model(variant)
        assert model.preprocessing_config.input_size == expected_size

# Determinism
class TestDeterminism:

    def test_eval_mode_deterministic(self):
        model = _make_model()
        model.cpu().eval()
        x = _dummy_input()

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)
