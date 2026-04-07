from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, _expected_calibration_error


def test_compute_metrics_returns_expected_values_for_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])

    metrics = compute_metrics(y_true, y_prob)

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["ece"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["n_samples"] == 4
    assert metrics["prevalence"] == pytest.approx(0.5)
    assert metrics["threshold"] == pytest.approx(0.5)


def test_compute_metrics_respects_custom_threshold():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.2, 0.6, 0.8, 0.7])

    metrics = compute_metrics(y_true, y_prob, threshold=0.75)

    assert metrics["f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["threshold"] == pytest.approx(0.75)


def test_compute_metrics_handles_single_class_without_crashing():
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.6, 0.7, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_prob)

    assert math.isnan(metrics["auroc"])
    assert math.isnan(metrics["auprc"])
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(0.0)
    assert metrics["specificity"] == pytest.approx(0.0)
    assert metrics["prevalence"] == pytest.approx(1.0)


def test_expected_calibration_error_is_zero_for_empty_input():
    assert _expected_calibration_error(np.array([]), np.array([])) == pytest.approx(0.0)


def test_expected_calibration_error_is_positive_for_miscalibrated_predictions():
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    ece = _expected_calibration_error(y_true, y_prob, n_bins=4)

    assert ece > 0.0
