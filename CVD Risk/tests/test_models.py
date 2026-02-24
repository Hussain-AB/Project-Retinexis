"""
Universal contract tests for all registered models.

Every model in MODEL_REGISTRY is automatically discovered and tested against
the BaseClassifier interface contract. If your model passes these tests, it
will work with the training pipeline.

Run (all models):
    pytest tests/test_models.py -v

Run for youra specific model only:
    pytest tests/test_models.py -v -k "efficientnet"

This file tests the CONTRACT, not model-specific behaviour. Keep your
model-specific tests (edge cases, variant configs, weight loading) in a
separate file like test_efficientnet.py.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models import create_model, MODEL_REGISTRY, BaseClassifier, PreprocessingConfig

TEST_OVERRIDES: dict[str, dict] = {
    # add model-specific test kwargs here if the defaults don't work.
    # Example:
    # "retfound_mae": {"weights_path": None},
}


def _discover_models() -> list[tuple[str, BaseClassifier]]:
    """Instantiate every registered model with pretrained=False."""
    models = []
    for name in MODEL_REGISTRY:
        kwargs = {"pretrained": False}
        kwargs.update(TEST_OVERRIDES.get(name, {}))
        try:
            model = create_model(name, **kwargs)
            models.append((name, model))
        except Exception as e:
            pytest.fail(
                f"Failed to instantiate '{name}' with kwargs {kwargs}: {e}"
            )
    return models


def _dummy_input(model: BaseClassifier, batch_size: int = 2) -> torch.Tensor:
    """Build a random input tensor matching the model's preprocessing config."""
    size = model.preprocessing_config.input_size
    return torch.randn(batch_size, 3, size, size)


def _dummy_labels(batch_size: int = 2) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, 1)).float()


# Parametrise every test class over all discovered models
MODELS = _discover_models()
MODEL_IDS = [name for name, _ in MODELS]


@pytest.fixture(params=MODELS, ids=MODEL_IDS)
def model_and_name(request):
    """Yields (model_name, model_instance) for each registered model."""
    name, model = request.param
    # Return a fresh instance each time to avoid state leaking between tests
    kwargs = {"pretrained": False}
    kwargs.update(TEST_OVERRIDES.get(name, {}))
    fresh = create_model(name, **kwargs)
    return name, fresh

# 1. Instantiation
class TestInstantiation:

    def test_inherits_from_base(self, model_and_name):
        name, model = model_and_name
        assert isinstance(model, BaseClassifier), (
            f"'{name}' does not inherit from BaseClassifier"
        )

    def test_has_backbone_property(self, model_and_name):
        name, model = model_and_name
        bb = model.backbone
        assert isinstance(bb, nn.Module), (
            f"'{name}'.backbone must return nn.Module, got {type(bb)}"
        )

    def test_has_head_property(self, model_and_name):
        name, model = model_and_name
        head = model.head
        assert isinstance(head, nn.Module), (
            f"'{name}'.head must return nn.Module, got {type(head)}"
        )

    def test_feature_dim_is_positive_int(self, model_and_name):
        name, model = model_and_name
        dim = model.feature_dim
        assert isinstance(dim, int) and dim > 0, (
            f"'{name}'.feature_dim must be a positive int, got {dim}"
        )

    def test_preprocessing_config_type(self, model_and_name):
        name, model = model_and_name
        config = model.preprocessing_config
        assert isinstance(config, PreprocessingConfig), (
            f"'{name}'.preprocessing_config must return PreprocessingConfig"
        )

    def test_preprocessing_config_is_frozen(self, model_and_name):
        _, model = model_and_name
        config = model.preprocessing_config
        with pytest.raises(AttributeError):
            config.input_size = 999

# 2. Forward pass
class TestForwardPass:

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_output_shape(self, model_and_name, batch_size):
        name, model = model_and_name
        model.eval()
        x = _dummy_input(model, batch_size=batch_size)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (batch_size, 1), (
            f"'{name}' output shape {logits.shape}, expected ({batch_size}, 1)"
        )

    def test_output_is_raw_logits(self, model_and_name):
        """Output should be unbounded — no sigmoid applied.

        NOTE: This can false-positive for ViTs with pretrained=False.
        Random ViT weights + LayerNorm collapse outputs to near-identical
        values that happen to land in [0,1]. This is not sigmoid — it's
        feature collapse. If you see a tight cluster of identical values
        (e.g. all ~0.14), that's expected. A real sigmoid failure would
        show outputs spread across [0,1] but never exceeding it.
        """
        name, model = model_and_name
        model.eval()
        # Large random input to push outputs away from 0-1
        size = model.preprocessing_config.input_size
        x = torch.randn(8, 3, size, size) * 10
        with torch.no_grad():
            logits = model(x)
        assert logits.min() < 0 or logits.max() > 1, (
            f"'{name}' logits suspiciously bounded to [0,1] — "
            f"might be applying sigmoid in forward()"
        )

    def test_output_is_finite(self, model_and_name):
        name, model = model_and_name
        model.eval()
        x = _dummy_input(model)
        with torch.no_grad():
            logits = model(x)
        assert torch.isfinite(logits).all(), (
            f"'{name}' produced non-finite logits"
        )

# 3. Feature extraction
class TestFeatureExtraction:

    def test_feature_shape(self, model_and_name):
        name, model = model_and_name
        model.eval()
        x = _dummy_input(model)
        with torch.no_grad():
            features = model.get_features(x)
        assert features.shape == (2, model.feature_dim), (
            f"'{name}'.get_features() shape {features.shape}, "
            f"expected (2, {model.feature_dim})"
        )

    def test_features_are_2d(self, model_and_name):
        name, model = model_and_name
        model.eval()
        x = _dummy_input(model)
        with torch.no_grad():
            features = model.get_features(x)
        assert features.dim() == 2, (
            f"'{name}'.get_features() should return 2D tensor, got {features.dim()}D"
        )

    def test_features_are_finite(self, model_and_name):
        name, model = model_and_name
        model.eval()
        x = _dummy_input(model)
        with torch.no_grad():
            features = model.get_features(x)
        assert torch.isfinite(features).all(), (
            f"'{name}'.get_features() produced non-finite values"
        )

# 4. Freeze / unfreeze
class TestFreezeUnfreeze:

    def test_freeze_zeros_backbone_grad(self, model_and_name):
        name, model = model_and_name
        model.freeze_backbone()
        for pname, p in model.backbone.named_parameters():
            assert not p.requires_grad, (
                f"'{name}' backbone param '{pname}' still requires grad after freeze"
            )

    def test_freeze_keeps_head_trainable(self, model_and_name):
        name, model = model_and_name
        model.freeze_backbone()
        head_trainable = sum(1 for p in model.head.parameters() if p.requires_grad)
        assert head_trainable > 0, (
            f"'{name}' head has no trainable params after freeze_backbone()"
        )

    def test_unfreeze_restores_all(self, model_and_name):
        name, model = model_and_name
        model.freeze_backbone()
        model.unfreeze_backbone()
        frozen = [
            pname for pname, p in model.named_parameters()
            if not p.requires_grad
        ]
        assert len(frozen) == 0, (
            f"'{name}' has frozen params after unfreeze: {frozen[:5]}"
        )

    def test_trainable_count_drops_after_freeze(self, model_and_name):
        name, model = model_and_name
        before = model.num_trainable_params()
        model.freeze_backbone()
        after = model.num_trainable_params()
        assert after < before, (
            f"'{name}' trainable params didn't drop: {before} → {after}"
        )

# 5. Linear probe lock
class TestLinearProbeLock:

    def _assert_locked_state(self, model: BaseClassifier, name: str):
        assert model.training is True, f"'{name}' model.training should be True"
        assert model.backbone.training is False, f"'{name}' backbone must be eval (locked)"
        assert model.head.training is True, f"'{name}' head must be train"

    def test_set_linear_probe_mode(self, model_and_name):
        name, model = model_and_name
        model.set_linear_probe_mode()
        self._assert_locked_state(model, name)

    def test_lock_survives_model_train(self, model_and_name):
        """model.train() must not revert backbone to train mode."""
        name, model = model_and_name
        model.set_linear_probe_mode()
        model.train()  # this is the line that breaks naive implementations
        self._assert_locked_state(model, name)

    def test_lock_survives_eval_train_cycle(self, model_and_name):
        name, model = model_and_name
        model.set_linear_probe_mode()
        model.eval()
        model.train()
        self._assert_locked_state(model, name)

    def test_finetune_releases_lock(self, model_and_name):
        name, model = model_and_name
        model.set_linear_probe_mode()
        model.set_finetune_mode()
        assert model.backbone.training is True, f"'{name}' backbone should be train after finetune"
        assert model._linear_probe_locked is False, f"'{name}' lock should be released"

    def test_deterministic_features_when_locked(self, model_and_name):
        """Frozen backbone must produce identical features across calls."""
        name, model = model_and_name
        model.set_linear_probe_mode()
        x = _dummy_input(model)
        with torch.no_grad():
            f1 = model.get_features(x)
            f2 = model.get_features(x)
        assert torch.allclose(f1, f2), (
            f"'{name}' backbone not deterministic in linear probe mode"
        )

# 6. Backward pass
class TestBackwardPass:

    def test_head_gets_gradients(self, model_and_name):
        name, model = model_and_name
        model.train()
        x = _dummy_input(model)
        y = _dummy_labels()
        logits = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()

        head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
        assert len(head_grads) > 0, f"'{name}' head received no gradients"
        assert all(torch.isfinite(g).all() for g in head_grads), (
            f"'{name}' head gradients are not finite"
        )

    def test_backbone_gets_gradients_in_finetune(self, model_and_name):
        name, model = model_and_name
        model.set_finetune_mode()
        x = _dummy_input(model)
        y = _dummy_labels()
        logits = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()

        bb_grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
        assert len(bb_grads) > 0, (
            f"'{name}' backbone received no gradients in finetune mode"
        )

    def test_backbone_no_gradients_when_frozen(self, model_and_name):
        name, model = model_and_name
        model.set_linear_probe_mode()
        x = _dummy_input(model)
        y = _dummy_labels()
        logits = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()

        for pname, p in model.backbone.named_parameters():
            assert p.grad is None, (
                f"'{name}' frozen backbone param '{pname}' received gradient"
            )

# 7. Determinism
class TestDeterminism:

    def test_eval_mode_deterministic(self, model_and_name):
        name, model = model_and_name
        model.eval()
        x = _dummy_input(model)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), (
            f"'{name}' not deterministic in eval mode"
        )

# 8. Factory + registry integration
class TestRegistry:
    """These run once (not per model) — they test the registry itself."""

    def test_registry_is_not_empty(self):
        assert len(MODEL_REGISTRY) > 0, "MODEL_REGISTRY is empty"

    def test_all_registered_models_instantiate(self):
        for name in MODEL_REGISTRY:
            kwargs = {"pretrained": False}
            kwargs.update(TEST_OVERRIDES.get(name, {}))
            model = create_model(name, **kwargs)
            assert isinstance(model, BaseClassifier), (
                f"'{name}' from factory is not a BaseClassifier"
            )

    def test_factory_rejects_unknown_architecture(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model("nonexistent_model_xyz", pretrained=False)
