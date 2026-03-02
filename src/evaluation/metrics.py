"""
evaluation.metrics — Metric computation for binary classification.

All metrics computed from (y_true, y_prob) arrays. The trainer calls
compute_metrics() at end of each val epoch and at final evaluation.

Handles degenerate cases (single class in a fold) by returning NaN
for AUROC/AUPRC instead of crashing — important because small subsets
during debugging or data efficiency experiments may be single-class.

Usage:
    metrics = compute_metrics(y_true, y_prob)
    # {"auroc": 0.85, "auprc": 0.78, "ece": 0.03, "f1": 0.80, ...}
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full metric suite for binary classification.

    Args:
        y_true: Ground truth labels, shape (N,), values in {0, 1}.
        y_prob: Predicted probabilities, shape (N,), values in [0, 1].
        threshold: Decision threshold for hard predictions.

    Returns:
        Dict with keys: auroc, auprc, ece, f1, sensitivity, specificity,
        n_samples, prevalence, threshold.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    n_samples = len(y_true)
    n_pos = (y_true == 1).sum()
    prevalence = float(n_pos / n_samples) if n_samples > 0 else 0.0

    # Threshold-invariant metrics
    # These are NaN when only one class is present (degenerate fold).
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))

    # Threshold-dependent metrics
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))

    if n_samples > 0 and len(np.unique(y_true)) >= 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0

    # Calibration
    ece = _expected_calibration_error(y_true, y_prob)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "n_samples": n_samples,
        "prevalence": prevalence,
        "threshold": threshold,
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width bins.

    Measures how well predicted probabilities match observed frequencies.
    Lower is better. 0.0 = perfectly calibrated.

    We report this alongside AUROC because class weighting (if used)
    directly affects calibration — a model trained with pos_weight != 1
    will have distorted posteriors.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    if total == 0:
        return 0.0

    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        # Include left edge in first bin
        if i == 0:
            mask = mask | (y_prob == bin_edges[0])

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (bin_count / total) * abs(bin_acc - bin_conf)

    return float(ece)
