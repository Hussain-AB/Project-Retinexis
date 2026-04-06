from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from src.utils.reproducibility import get_rng_states, seed_everything, set_rng_states


def test_seed_everything_reproducibly_resets_python_numpy_and_torch():
    seed_everything(123, deterministic=True)
    first_python = random.random()
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)

    seed_everything(123, deterministic=True)
    second_python = random.random()
    second_numpy = np.random.rand(3)
    second_torch = torch.rand(3)

    assert first_python == pytest.approx(second_python)
    assert np.allclose(first_numpy, second_numpy)
    assert torch.allclose(first_torch, second_torch)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_get_and_set_rng_states_restore_sequences():
    seed_everything(7, deterministic=False)
    states = get_rng_states()

    first_python = random.random()
    first_numpy = np.random.rand(4)
    first_torch = torch.rand(4)

    set_rng_states(states)

    second_python = random.random()
    second_numpy = np.random.rand(4)
    second_torch = torch.rand(4)

    assert first_python == pytest.approx(second_python)
    assert np.allclose(first_numpy, second_numpy)
    assert torch.allclose(first_torch, second_torch)
