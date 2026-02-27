"""
training.logging — Experiment logging (CSV + Weights & Biases).

CSV logging is always active (the ground truth record). W&B is
optional — if wandb is not installed or WANDB_MODE=disabled, the
logger falls back gracefully.

This keeps all logging concerns out of trainer.py. The trainer calls
logger.log_epoch() and logger.log_summary() without caring whether
W&B is active.

W&B Setup:
    pip install wandb
    wandb login
    # On Ibex, wandb is pre-installed. Just set WANDB_API_KEY.
    # I have already setup a Weights & Balances account for this project
    # so please ask me for the WANDB_API_KEY.

Usage:
    logger = TrainLogger(config)
    logger.log_epoch(epoch=0, phase="phase1", metrics={...})
    logger.log_summary({"best_val_auroc": 0.87})
    logger.finish()
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# W&B is optional — this has graceful fallback if not installed but we should always try to use
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class TrainLogger:
    """Dual CSV + W&B logger for training runs.

    CSV is always written (it's the permanent record). W&B is initialised
    if available and config.use_wandb is True.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

        # CSV setup (always active)
        self.log_path = config.log_dir / "training_log.csv"
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "phase", "train_loss", "val_loss",
                    "val_auroc", "val_auprc", "val_ece", "val_f1",
                    "val_sensitivity", "val_specificity",
                    "lr", "elapsed_sec",
                ])

        # W&B setup (optional)
        self.use_wandb = getattr(config, "use_wandb", False) and WANDB_AVAILABLE
        self._wandb_run = None

        if self.use_wandb:
            self._wandb_run = wandb.init(
                # project name — shared across both teams
                project=getattr(config, "wandb_project", "retinexis"),
                # entity — the W&B team/org config set
                entity=getattr(config, "wandb_entity", None),
                # run name — auto-generated from config for easy filtering
                name=f"{config.dataset}/{config.architecture}/fold_{config.fold}",
                # group — lets you group folds of the same model together
                group=f"{config.dataset}/{config.architecture}",
                # job_type distinguishes phase1 screening from phase2 full runs
                job_type="phase1" if config.phase2_epochs == 0 else "full",
                # log the full config so runs are fully reproducible from W&B
                config=config.to_dict(),
                # resume support — if the run was killed, W&B picks up where
                # it left off using the same run ID
                resume="allow",
                # don't let W&B take over the console
                settings=wandb.Settings(console="off"),
            )

    def log_epoch(
        self,
        epoch: int,
        phase: str,
        train_loss: float,
        val_loss: float,
        val_metrics: dict,
        lr: float,
        elapsed: float,
    ) -> None:
        """Log one epoch to both CSV and W&B."""
        # CSV (always)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                phase,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_metrics.get('auroc', 0.0):.6f}",
                f"{val_metrics.get('auprc', 0.0):.6f}",
                f"{val_metrics.get('ece', 0.0):.6f}",
                f"{val_metrics.get('f1', 0.0):.6f}",
                f"{val_metrics.get('sensitivity', 0.0):.6f}",
                f"{val_metrics.get('specificity', 0.0):.6f}",
                f"{lr:.8f}",
                f"{elapsed:.1f}",
            ])

        # W&B (if active)
        if self.use_wandb and self._wandb_run is not None:
            log_dict = {
                "epoch": epoch,
                "phase": phase,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/auroc": val_metrics.get("auroc", 0.0),
                "val/auprc": val_metrics.get("auprc", 0.0),
                "val/ece": val_metrics.get("ece", 0.0),
                "val/f1": val_metrics.get("f1", 0.0),
                "val/sensitivity": val_metrics.get("sensitivity", 0.0),
                "val/specificity": val_metrics.get("specificity", 0.0),
                "lr": lr,
                "epoch_time_sec": elapsed,
            }
            wandb.log(log_dict, step=epoch)

    def log_summary(self, summary: dict) -> None:
        """Log final summary metrics (best AUROC, etc.) to W&B."""
        if self.use_wandb and self._wandb_run is not None:
            for key, value in summary.items():
                wandb.run.summary[key] = value

    def finish(self) -> None:
        """Clean up W&B run. Call at the end of training."""
        if self.use_wandb and self._wandb_run is not None:
            wandb.finish()
