"""
Unit tests for Project Retinexis deployment app.

Run with:
    cd ~/Documents/Project-Retinexis
    /usr/local/bin/python3.11 -m pytest deployment/test_app.py -v

These tests cover the pure helper functions and logic — they do NOT run
actual model inference (that would require downloading 3.65 GB checkpoints).
Tests should complete in under 5 seconds.
"""

import sys
import os
import json
import pytest
import numpy as np
import torch

# Add deployment directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app import (
    # Configuration
    TASK_MODELS,
    HF_CHECKPOINTS,
    get_model_choices,
    get_model_description,
    # Preprocessing helpers
    _remove_top_left_overlay_cimt,
    _clahe_cimt,
    _clahe_htn,
    _resize_aspect_preserve_htn,
    # Multimodal helpers
    MetadataEncoder,
    _build_metadata_head_from_state,
    # FHIR output
    generate_fhir_observation,
    # UI helpers
    build_result_card,
    _extract_prediction,
    _extract_confidence,
    _summary_to_card,
)


# ─────────────────────────────────────────────────────────────────────────
# Configuration tests — make sure TASK_MODELS and HF_CHECKPOINTS are sane
# ─────────────────────────────────────────────────────────────────────────

class TestTaskConfiguration:
    """Check that TASK_MODELS and HF_CHECKPOINTS are consistent."""

    def test_tasks_exist(self):
        """Both CVD Risk and Hypertension tasks should exist."""
        assert "CVD Risk" in TASK_MODELS
        assert "Hypertension" in TASK_MODELS
        assert "CVD Risk" in HF_CHECKPOINTS
        assert "Hypertension" in HF_CHECKPOINTS

    def test_hypertension_has_real_models(self):
        """Hypertension should have at least the 3 image-only models with real checkpoints."""
        assert "EfficientNet-B2" in HF_CHECKPOINTS["Hypertension"]
        assert "RETFound-MAE" in HF_CHECKPOINTS["Hypertension"]
        assert "DINOv2-Large" in HF_CHECKPOINTS["Hypertension"]

    def test_cvd_has_real_models(self):
        """CVD Risk should have the 3 image-only models with real checkpoints."""
        assert "RETFound-MAE" in HF_CHECKPOINTS["CVD Risk"]
        assert "ConvNeXt V2-Base" in HF_CHECKPOINTS["CVD Risk"]
        assert "EfficientNet-B2" in HF_CHECKPOINTS["CVD Risk"]

    def test_every_checkpoint_has_model_entry(self):
        """Every model in HF_CHECKPOINTS must also exist in TASK_MODELS."""
        for task, checkpoints in HF_CHECKPOINTS.items():
            for model_name in checkpoints:
                assert model_name in TASK_MODELS[task], \
                    f"{model_name} in HF_CHECKPOINTS but missing from TASK_MODELS[{task}]"

    def test_all_models_have_required_fields(self):
        """Every model entry must have architecture, variant, and description."""
        for task, models in TASK_MODELS.items():
            for model_name, info in models.items():
                assert "architecture" in info, f"{task}/{model_name} missing architecture"
                assert "variant" in info, f"{task}/{model_name} missing variant"
                assert "description" in info, f"{task}/{model_name} missing description"

    def test_multimodal_models_are_flagged(self):
        """Models with 'Multimodal' in name should have multimodal: True."""
        for task, models in TASK_MODELS.items():
            for model_name, info in models.items():
                if "Multimodal" in model_name:
                    assert info.get("multimodal") is True, \
                        f"{task}/{model_name} has 'Multimodal' in name but multimodal flag is not True"


class TestModelChoices:
    """Check the dropdown helpers."""

    def test_get_model_choices_hypertension(self):
        choices = get_model_choices("Hypertension")
        assert isinstance(choices, list)
        assert len(choices) > 0
        assert "EfficientNet-B2" in choices

    def test_get_model_choices_cvd(self):
        choices = get_model_choices("CVD Risk")
        assert isinstance(choices, list)
        assert len(choices) > 0
        assert "RETFound-MAE" in choices

    def test_get_model_choices_unknown_task(self):
        """Unknown task should return empty list, not crash."""
        assert get_model_choices("FakeTask") == []

    def test_get_model_description_valid(self):
        desc = get_model_description("Hypertension", "EfficientNet-B2")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_model_description_invalid(self):
        """Unknown model should return empty string, not crash."""
        assert get_model_description("Hypertension", "NonexistentModel") == ""


# ─────────────────────────────────────────────────────────────────────────
# Preprocessing tests — shape, dtype, no crashes
# ─────────────────────────────────────────────────────────────────────────

class TestCIMTPreprocessing:
    """Test CIMT/CVD Risk preprocessing helpers."""

    def test_remove_overlay_shape_preserved(self):
        """Overlay removal should not change image shape."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 128
        out = _remove_top_left_overlay_cimt(img)
        assert out.shape == img.shape
        assert out.dtype == img.dtype

    def test_remove_overlay_blacks_out_white_region(self):
        """If top-left has white pixels, it should be replaced."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 128
        img[:50, :90] = 255  # Simulate white text overlay
        out = _remove_top_left_overlay_cimt(img)
        # The top-left 50x90 should no longer be white
        assert not np.all(out[:50, :90] == 255)

    def test_remove_overlay_leaves_image_alone_if_no_white(self):
        """If there's no white overlay, the image should be unchanged."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 128
        out = _remove_top_left_overlay_cimt(img)
        np.testing.assert_array_equal(out, img)

    def test_clahe_cimt_shape_preserved(self):
        """CLAHE should not change image shape."""
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out = _clahe_cimt(img)
        assert out.shape == img.shape
        assert out.dtype == img.dtype


class TestHypertensionPreprocessing:
    """Test Hypertension preprocessing helpers."""

    def test_clahe_htn_shape_preserved(self):
        """CLAHE on LAB should not change image shape."""
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out = _clahe_htn(img)
        assert out.shape == img.shape
        assert out.dtype == img.dtype

    def test_resize_aspect_preserve_square_input(self):
        """Square input should stay square at target size."""
        img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        out = _resize_aspect_preserve_htn(img, target_size=512)
        assert out.shape == (512, 512, 3)

    def test_resize_aspect_preserve_landscape_input(self):
        """Landscape input should be resized + center cropped to square."""
        img = np.random.randint(0, 255, (800, 1600, 3), dtype=np.uint8)
        out = _resize_aspect_preserve_htn(img, target_size=512)
        assert out.shape == (512, 512, 3)

    def test_resize_aspect_preserve_portrait_input(self):
        """Portrait input should be resized + center cropped to square."""
        img = np.random.randint(0, 255, (1600, 800, 3), dtype=np.uint8)
        out = _resize_aspect_preserve_htn(img, target_size=512)
        assert out.shape == (512, 512, 3)

    def test_resize_aspect_preserve_small_input(self):
        """Small input should be upscaled (or padded) to target size."""
        img = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        out = _resize_aspect_preserve_htn(img, target_size=512)
        assert out.shape == (512, 512, 3)

    def test_resize_aspect_preserve_custom_target(self):
        """Should respect custom target_size."""
        img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        out = _resize_aspect_preserve_htn(img, target_size=256)
        assert out.shape == (256, 256, 3)


# ─────────────────────────────────────────────────────────────────────────
# Multimodal tests — metadata encoder and head reconstruction
# ─────────────────────────────────────────────────────────────────────────

class TestMetadataEncoder:
    """Test the MetadataEncoder class used for multimodal inference."""

    def _make_encoder_dict(self):
        """Creates a sample encoder dict matching the real multimodal checkpoint format."""
        return {
            "continuous_cols": ["age"],
            "categorical_cols": ["gender"],
            "means": {"age": 57.72},
            "stds": {"age": 14.80},
            "category_maps": {"gender": {"1": 0, "2": 1, "Female": 2, "Male": 3}},
            "output_dim": 5,
        }

    def test_from_dict_loads_correctly(self):
        enc = MetadataEncoder.from_dict(self._make_encoder_dict())
        assert enc.output_dim == 5

    def test_encode_male_60(self):
        """Encoding age=60, gender=Male should produce tensor of shape (5,)."""
        enc = MetadataEncoder.from_dict(self._make_encoder_dict())
        out = enc.encode({"age": 60, "gender": "Male"})
        assert out.shape == (5,)
        # Age should be z-scored: (60 - 57.72) / 14.80 ≈ 0.154
        assert abs(out[0].item() - ((60 - 57.72) / 14.80)) < 0.01
        # Male is index 3, so one-hot should be [0, 0, 0, 1]
        assert out[1].item() == 0.0
        assert out[2].item() == 0.0
        assert out[3].item() == 0.0
        assert out[4].item() == 1.0

    def test_encode_female_45(self):
        """Encoding age=45, gender=Female should produce correct one-hot."""
        enc = MetadataEncoder.from_dict(self._make_encoder_dict())
        out = enc.encode({"age": 45, "gender": "Female"})
        assert out.shape == (5,)
        # Female is index 2
        assert out[3].item() == 1.0  # One-hot position 2 is element index 3 (age + 2)

    def test_encode_unknown_gender_produces_zeros(self):
        """Unknown category should give all-zero one-hot."""
        enc = MetadataEncoder.from_dict(self._make_encoder_dict())
        out = enc.encode({"age": 60, "gender": "Unknown"})
        # Age is fine, but the one-hot for unknown should be all zeros
        assert out.shape == (5,)
        assert out[1:].sum().item() == 0.0


class TestMetadataHeadBuilder:
    """Test that we can reconstruct a metadata head from a state dict."""

    def test_build_head_from_valid_state(self):
        """Should build a Sequential with the right shapes."""
        # Simulate a state dict from a multimodal EfficientNet-B2 checkpoint
        # Layer 0: Linear(1413, 128), Layer 3: Linear(128, 1)
        state = {
            "0.weight": torch.randn(128, 1413),
            "0.bias": torch.randn(128),
            "3.weight": torch.randn(1, 128),
            "3.bias": torch.randn(1),
        }
        head = _build_metadata_head_from_state(state)
        # Should be able to load the state dict
        head.load_state_dict(state)
        # Forward pass with correct shape
        x = torch.randn(1, 1413)
        out = head(x)
        assert out.shape == (1, 1)


# ─────────────────────────────────────────────────────────────────────────
# FHIR output tests
# ─────────────────────────────────────────────────────────────────────────

class TestFHIROutput:
    """Test the FHIR Observation generator."""

    def test_fhir_basic_structure(self):
        obs = generate_fhir_observation(
            task="Hypertension",
            model_name="EfficientNet-B2",
            probability=0.75,
            prediction=1,
        )
        assert obs["resourceType"] == "Observation"
        assert obs["status"] == "final"
        assert "code" in obs
        assert "valueQuantity" in obs

    def test_fhir_positive_interpretation(self):
        obs = generate_fhir_observation("Hypertension", "Test", 0.75, 1)
        code = obs["interpretation"][0]["coding"][0]["code"]
        assert code == "POS"

    def test_fhir_negative_interpretation(self):
        obs = generate_fhir_observation("Hypertension", "Test", 0.25, 0)
        code = obs["interpretation"][0]["coding"][0]["code"]
        assert code == "NEG"

    def test_fhir_probability_value(self):
        obs = generate_fhir_observation("Hypertension", "Test", 0.8, 1)
        assert obs["valueQuantity"]["value"] == 0.8

    def test_fhir_with_metadata_extensions(self):
        obs = generate_fhir_observation(
            "Hypertension", "Test", 0.7, 1,
            metadata={"age": 60, "sex": 1}
        )
        assert "extension" in obs
        assert len(obs["extension"]) == 2

    def test_fhir_without_metadata(self):
        obs = generate_fhir_observation("Hypertension", "Test", 0.5, 0)
        assert "extension" not in obs or len(obs.get("extension", [])) == 0

    def test_fhir_is_json_serializable(self):
        """Should be serializable to JSON without errors."""
        obs = generate_fhir_observation("CVD Risk", "Test", 0.5, 0, {"age": 50, "sex": 0})
        json_str = json.dumps(obs)
        assert len(json_str) > 0


# ─────────────────────────────────────────────────────────────────────────
# Summary parsing & result card tests
# ─────────────────────────────────────────────────────────────────────────

class TestSummaryParsing:
    """Test the regex-based summary parsing helpers."""

    def test_extract_positive(self):
        summary = "## Hypertension Prediction\n\n**Risk classification:** Positive"
        assert _extract_prediction(summary) == "Positive"

    def test_extract_negative(self):
        summary = "## Hypertension Prediction\n\n**Risk classification:** Negative"
        assert _extract_prediction(summary) == "Negative"

    def test_extract_prediction_none(self):
        assert _extract_prediction("Random text") is None

    def test_extract_confidence_valid(self):
        summary = "**Probability:** 75.3%\n\n**Confidence:** 85.0%\n"
        assert abs(_extract_confidence(summary) - 0.85) < 0.001

    def test_extract_confidence_invalid(self):
        """Invalid summary should return -1.0, not crash."""
        assert _extract_confidence("no confidence here") == -1.0


class TestResultCardBuilder:
    """Test the HTML result card generator."""

    def test_cvd_card_contains_right_values(self):
        html = build_result_card("CVD Risk", "RETFound-MAE", 0.75, 1, 0.85)
        assert "75.0%" in html
        assert "85.0%" in html
        assert "RETFound-MAE" in html
        assert "Cardiovascular risk" in html
        assert "Positive" in html

    def test_htn_card_has_htn_class(self):
        html = build_result_card("Hypertension", "EfficientNet-B2", 0.3, 0, 0.4)
        assert "htn" in html
        assert "Hypertension" in html
        assert "Negative" in html

    def test_negative_card_shows_negative_badge(self):
        html = build_result_card("CVD Risk", "Test", 0.2, 0, 0.6)
        assert "badge-negative" in html
        assert "Negative" in html

    def test_positive_card_shows_positive_badge(self):
        html = build_result_card("CVD Risk", "Test", 0.9, 1, 0.8)
        assert "badge-positive" in html
        assert "Positive" in html


class TestSummaryToCard:
    """Test the summary→card converter."""

    def test_valid_summary_produces_card(self):
        summary = (
            "## Hypertension Prediction\n\n"
            "**Risk classification:** Positive\n\n"
            "**Probability:** 75.0%\n\n"
            "**Confidence:** 85.0%\n\n"
            "**Model:** Test\n\n"
        )
        card = _summary_to_card(summary, "Hypertension", "Test")
        assert "result-card" in card
        assert "75.0%" in card

    def test_no_image_summary_produces_placeholder(self):
        card = _summary_to_card("*No image uploaded for Left Eye.*", "Hypertension", "Test")
        assert "No image" in card


if __name__ == "__main__":
    pytest.main([__file__, "-v"])