"""
training.loss — Loss function construction.

Wraps BCEWithLogitsLoss with optional class weighting. The trainer
calls build_loss() once at init and uses the returned criterion for
both Phase 1 and Phase 2.

Usage:
    criterion = build_loss(
        use_weighted_bce=config.use_weighted_bce,
        pos_weight=config.pos_weight,
        train_labels=train_labels,
        device=device,
    )
    loss = criterion(logits, labels)


CLASS IMBALANCE NOTES (for team reference):

CIMT dataset is ~70.75% positive (thickened CIMT). When use_weighted_bce
is True, pos_weight = n_neg / n_pos ≈ 0.41, which DOWNWEIGHTS the
majority positive class in BCEWithLogitsLoss.

Current recommendation: start Phase 1 baselines with use_weighted_bce=False.

Rationale:
  - AUROC (our primary metric) is threshold-invariant, so class weighting
    doesn't change ranking performance — only calibration and the default
    decision boundary.
  - Weighted loss distorts the learned posterior away from the true class
    prior, which hurts ECE (calibration). If calibration matters for the
    deployment narrative, post-hoc Platt scaling on the val fold is more
    principled than loss reweighting.
  - We can always add a single weighted-loss ablation on one model to
    show we considered it and report the delta.

For external cohort evaluation:
  - Zero-shot: no training, loss not used. Class ratio affects threshold-
    dependent metrics but not AUROC/AUPRC.
  - Adapted (light fine-tune): build_loss auto-computes pos_weight from
    whatever external training labels it receives. If external cohort is
    70% positive too, same weight. If 50/50, pos_weight ≈ 1.0. No code
    changes needed.

The pos_weight is computed PER FOLD because stratification preserves
class ratio approximately, not exactly. Never hardcode a weight.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_pos_weight(labels: np.ndarray) -> float:
    """Compute pos_weight = num_negative / num_positive.

    For BCEWithLogitsLoss, pos_weight scales the loss contribution of
    positive examples. A value < 1 downweights the majority class when
    positives are the majority (e.g. CIMT ~70% positive → weight ≈ 0.41).

    Args:
        labels: 1D array of 0s and 1s.

    Returns:
        Weight as a float. Returns 1.0 if all labels are the same class
        (avoids division by zero in degenerate cases).
    """
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return float(n_neg / n_pos)


def build_loss(
    use_weighted_bce: bool = False,
    pos_weight: float | None = None,
    train_labels: np.ndarray | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Build the loss function.

    Args:
        use_weighted_bce: Whether to apply class weighting.
        pos_weight: Explicit weight. If None and use_weighted_bce is True,
            auto-computed from train_labels.
        train_labels: 1D array of training labels. Required when
            use_weighted_bce=True and pos_weight is None.
        device: Device to place the pos_weight tensor on.

    Returns:
        A BCEWithLogitsLoss instance ready for use.
    """
    if use_weighted_bce:
        if pos_weight is None:
            if train_labels is None:
                raise ValueError(
                    "use_weighted_bce=True and pos_weight=None, but no "
                    "train_labels provided to auto-compute weight."
                )
            pos_weight = compute_pos_weight(train_labels)
        weight_tensor = torch.tensor([pos_weight], device=device)
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    return nn.BCEWithLogitsLoss()
