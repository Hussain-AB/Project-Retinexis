"""
utils.reproducibility — Deterministic training setup and RNG state management.

Called once at the start of every run by train.py. Also provides
get_rng_states() / set_rng_states() for checkpoint save/restore,
which ensures identical data ordering after SLURM resume.

Usage:
    seed_everything(42, deterministic=True)

    # In checkpoint saving:
    checkpoint["rng_states"] = get_rng_states()

    # In checkpoint loading:
    set_rng_states(checkpoint["rng_states"])
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs and configure PyTorch for reproducibility.

    Args:
        seed: Random seed for Python, NumPy, and PyTorch.
        deterministic: If True, enables torch deterministic algorithms
            and disables cuDNN benchmark. Slower but reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for some CUDA ops to be deterministic
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_rng_states() -> dict:
    """Capture all RNG states for checkpoint saving.

    Returns a dict safe to pass to torch.save. Includes Python,
    NumPy, PyTorch CPU, and all CUDA device states.
    """
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict) -> None:
    """Restore RNG states from a checkpoint.

    This ensures the DataLoader produces the same batch order after
    a SLURM resume, which is critical for reproducibility.
    """
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    
    t_state = states["torch"]
    if not isinstance(t_state, torch.Tensor):
        t_state = torch.ByteTensor(t_state)
    torch.random.set_rng_state(t_state.cpu())
    
    if "cuda" in states and torch.cuda.is_available():
        c_states = states["cuda"]
        if isinstance(c_states, (list, tuple)):
            c_states = [
                torch.ByteTensor(s).cpu() if not isinstance(s, torch.Tensor) else s.cpu()
                for s in c_states
            ]
        torch.cuda.set_rng_state_all(c_states)

