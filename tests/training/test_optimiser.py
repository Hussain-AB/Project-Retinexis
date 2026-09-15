from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.loss import build_loss, compute_pos_weight
from src.training.optimiser import build_layer_wise_param_groups, split_decay_or_no_decay
from src.training.scheduler import build_cosine_warmup_scheduler


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4, 4))
        self.layer2 = nn.Sequential(nn.Linear(4, 4))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._backbone = _Backbone()
        self._head = nn.Linear(4, 1)

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @property
    def head(self) -> nn.Module:
        return self._head


def test_split_decay_or_no_decay_filters_frozen_parameters():
    weight = nn.Parameter(torch.ones(2, 2))
    bias = nn.Parameter(torch.ones(2))
    frozen = nn.Parameter(torch.ones(2, 2), requires_grad=False)

    decay, no_decay = split_decay_or_no_decay([weight, bias, frozen])

    assert decay == [weight]
    assert no_decay == [bias]


def test_build_layer_wise_param_groups_assigns_head_and_backbone_learning_rates():
    model = _Model()

    param_groups = build_layer_wise_param_groups(
        model=model,
        base_lr=1e-3,
        lr_decay_factor=0.5,
        weight_decay=0.1,
    )

    assert len(param_groups) == 6
    assert param_groups[0]["lr"] == pytest.approx(1e-3)
    assert param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert param_groups[1]["lr"] == pytest.approx(1e-3)
    assert param_groups[1]["weight_decay"] == pytest.approx(0.0)
    assert param_groups[2]["lr"] == pytest.approx(2.5e-4)
    assert param_groups[3]["lr"] == pytest.approx(2.5e-4)
    assert param_groups[4]["lr"] == pytest.approx(5e-4)
    assert param_groups[5]["lr"] == pytest.approx(5e-4)


def test_build_layer_wise_param_groups_uses_metadata_head_when_provided():
    model = _Model()
    metadata_head = nn.Linear(6, 1)

    param_groups = build_layer_wise_param_groups(
        model=model,
        base_lr=1e-4,
        lr_decay_factor=0.8,
        weight_decay=0.01,
        metadata_head=metadata_head,
    )

    metadata_param_ids = {id(param) for param in metadata_head.parameters()}
    head_group_param_ids = {
        id(param)
        for group in param_groups[:2]
        for param in group["params"]
    }
    model_head_param_ids = {id(param) for param in model.head.parameters()}

    assert metadata_param_ids <= head_group_param_ids
    assert model_head_param_ids.isdisjoint(head_group_param_ids)


def test_compute_pos_weight_returns_ratio_for_binary_labels():
    labels = np.array([1, 0, 0, 0])

    assert compute_pos_weight(labels) == pytest.approx(3.0)


def test_compute_pos_weight_returns_one_for_single_class_labels():
    assert compute_pos_weight(np.array([1, 1, 1])) == pytest.approx(1.0)
    assert compute_pos_weight(np.array([0, 0, 0])) == pytest.approx(1.0)


def test_build_loss_returns_unweighted_bce_by_default():
    criterion = build_loss()

    assert isinstance(criterion, nn.BCEWithLogitsLoss)
    assert criterion.pos_weight is None


def test_build_loss_uses_explicit_pos_weight():
    criterion = build_loss(use_weighted_bce=True, pos_weight=2.5)

    assert criterion.pos_weight is not None
    assert criterion.pos_weight.item() == pytest.approx(2.5)


def test_build_loss_can_compute_pos_weight_from_labels():
    labels = np.array([1, 0, 0, 0])

    criterion = build_loss(use_weighted_bce=True, train_labels=labels)

    assert criterion.pos_weight.item() == pytest.approx(3.0)


def test_build_loss_requires_labels_when_weighting_is_enabled():
    with pytest.raises(ValueError, match="train_labels provided to auto-compute weight"):
        build_loss(use_weighted_bce=True, pos_weight=None, train_labels=None)


def test_build_cosine_warmup_scheduler_exposes_expected_scaling_curve():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimiser = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = build_cosine_warmup_scheduler(
        optimiser=optimiser,
        warmup_epochs=2,
        total_epochs=6,
        min_lr=0.01,
        base_lr=0.1,
    )

    lr_lambda = scheduler.lr_lambdas[0]

    assert lr_lambda(0) == pytest.approx(0.5)
    assert lr_lambda(1) == pytest.approx(1.0)
    assert lr_lambda(2) == pytest.approx(1.0)
    assert lr_lambda(6) == pytest.approx(0.1)
    assert lr_lambda(4) < lr_lambda(2)
