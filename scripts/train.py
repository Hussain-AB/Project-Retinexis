"""
scripts/train.py — Main training entry point.

Thin orchestrator: parse args, load config, create model, hand off
to Trainer. All real logic lives in src/.

Usage:
    python scripts/train.py --config tasks/cimt/configs/efficientnet_b2_phase1.yaml --fold 0
    python scripts/train.py --config X.yaml --phase1_only
    python scripts/train.py --config X.yaml --fold 2 --override phase2_lr 1e-4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import warnings

# Make src/ importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.schema import TrainConfig
from src.config.loader import load_config
from src.models import create_model # this won't work until we bring the models over
from src.utils.reproducibility import seed_everything
from src.training.trainer import Trainer

# Trigger dataset registration (classes register in DATASET_REGISTRY at import)
import src.data.datasets  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retinal CVD risk model training")

    # Required
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )

    # Common overrides
    parser.add_argument("--fold", type=int, default=None, help="CV fold index (0-4)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--phase1_only", action="store_true", help="Run Phase 1 only (sets phase2_epochs=0)")
    parser.add_argument("--skip_phase1", action="store_true", help="Skip Phase 1, go straight to Phase 2")
    parser.add_argument("--resume_weights_only", action="store_true", help="Only load model weights from checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    # Arbitrary overrides: --override key value [key value ...]
    parser.add_argument(
        "--override", nargs=2, action="append", default=[],
        metavar=("KEY", "VALUE"),
        help="Override config field (e.g. --override phase2_lr 1e-4)",
    )

    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    """Collect CLI arguments into a config override dict."""
    overrides = {}

    if args.fold is not None:
        overrides["fold"] = args.fold
    if args.checkpoint_path is not None:
        overrides["checkpoint_path"] = args.checkpoint_path
    if args.phase1_only:
        overrides["phase2_epochs"] = 0
    if args.skip_phase1:
        overrides["skip_phase1"] = True
    if args.resume_weights_only:
        overrides["resume_weights_only"] = True
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir

    # --override key value pairs
    for key, value in args.override:
        overrides[key] = value

    return overrides


def select_device() -> torch.device:
    """Pick the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon — useful for local testing, same idea as with SGP
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main() -> None:
    warnings.filterwarnings("ignore", message=".*pin_memory.*")
    args = parse_args()
    overrides = build_overrides(args)

    # Load and validate config
    config = load_config(args.config, overrides)

    print(f"{'=' * 70}")
    print(f"Training: {config.architecture} on {config.dataset} (fold {config.fold})")
    print(f"{'=' * 70}")
    print(f"  Config: {args.config}")
    print(f"  Output: {config.run_dir}")

    # Seed for reproducibility
    seed_everything(config.seed, deterministic=config.deterministic)

    # Select device
    device = select_device()
    print(f"  Device: {device}")

    # Create model
    model = create_model(config.architecture, **config.model_kwargs)
    print(f"  Model:  {config.architecture} ({model.num_total_params():,} params)")
    print(f"  Input:  {model.preprocessing_config.input_size}px")

    # Run training
    start = time.time()
    trainer = Trainer(config, model, device)
    results = trainer.run()

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"Done in {elapsed / 60:.1f} minutes")
    print(f"Best val AUROC: {results.get('best_val_auroc', 'N/A')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
