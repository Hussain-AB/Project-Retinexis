"""
training.trainer — Trainer class for Phase 1 (linear probe) and Phase 2 (fine-tune).

Central orchestrator for a single training run (1 model × 1 fold).
Model-agnostic and dataset-agnostic — all specifics come from config
and the contracts in base.py / base_dataset.py.

Lifecycle:
    trainer = Trainer(config, model, device)
    results = trainer.run()

    run() orchestrates:
        1. Resume from checkpoint if config.checkpoint_path is set
        2. Phase 1 (linear probe) unless skip_phase1
        3. Phase 2 (fine-tune) unless phase2_epochs == 0
        4. Load best checkpoint → final eval → save predictions CSV
        5. Return metrics dict

Phase 1: frozen backbone, head-only, flat LR, 5 epochs, ~15 min.
Phase 2: all params, layer-wise LR decay, cosine warmup, early stopping, up to 50 epochs.

Logging is handled by TrainLogger (src/training/logging.py), which
writes CSV always and W&B when enabled. The trainer never calls
wandb directly.

Checkpoints include model weights, optimiser state, scheduler state,
RNG states, and full config. See _save_checkpoint for exact format.

RESUME NOTE (spike finding, 2026-02-26):
    _load_checkpoint restores model weights and RNG states, and stashes
    the raw checkpoint dict. _run_phase2 then restores optimiser and
    scheduler state after constructing them, preserving Adam momentum
    and correct LR schedule positioning across SLURM preemptions.

METADATA FUSION:
    When config.use_metadata=True, the trainer constructs a separate
    metadata_head that takes model.get_features() + encoded metadata.
    The model's own head is bypassed but stays attached for state_dict
    consistency. See _build_metadata_head() and _forward().
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.schema import TrainConfig
from src.models.base import BaseClassifier
from src.data.loader import create_dataloaders, rebuild_loaders
from src.data.base_dataset import MetadataEncoder
from src.training.loss import build_loss
from src.training.optimiser import build_layer_wise_param_groups
from src.training.scheduler import build_cosine_warmup_scheduler
from src.training.logging import TrainLogger
from src.evaluation.metrics import compute_metrics
from src.utils.reproducibility import get_rng_states, set_rng_states


# HELPER: Metadata head construction
# Small enough to stay here. Extract to metadata.py if this file gets too big.
def _build_metadata_head(
    feature_dim: int,
    metadata_dim: int,
    hidden_dim: int = 0,
) -> nn.Module:
    """Build classification head for metadata fusion mode.

    Maps (B, feature_dim + metadata_dim) → (B, 1).
    If hidden_dim > 0: two-layer MLP with ReLU + dropout.
    If hidden_dim == 0: single linear projection.
    """
    input_dim = feature_dim + metadata_dim
    if hidden_dim > 0:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
    return nn.Linear(input_dim, 1)

#  TRAINER CLASS
class Trainer:
    """Runs Phase 1 + Phase 2 for a single model × fold."""

    def __init__(
        self,
        config: TrainConfig,
        model: BaseClassifier,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.model = model.to(device)

        # Build data loaders (Phase 1 batch size initially)
        print(f"\n  Loading data...")
        print(f"    Dataset:    {config.dataset}")
        print(f"    Labels CSV: {config.labels_csv}")
        print(f"    Splits:     {config.splits_file} (fold {config.fold})")
        print(f"    Image dir:  {config.data_dir}")
        (self.train_loader, self.inner_val_loader, self.val_loader,
         self.metadata_encoder) = create_dataloaders(
            config, model, batch_size=config.phase1_batch_size)
        print(f"    Train:     {len(self.train_loader.dataset)} samples ({len(self.train_loader)} batches)")
        print(f"    Inner val: {len(self.inner_val_loader.dataset)} samples ({len(self.inner_val_loader)} batches)")
        print(f"    Val:       {len(self.val_loader.dataset)} samples ({len(self.val_loader)} batches)")

        # Metadata head (only when use_metadata=True)
        # Lives on the trainer, not the model. model.head is bypassed
        # but stays attached for state_dict consistency.
        self.metadata_head: nn.Module | None = None
        if config.use_metadata and self.metadata_encoder is not None:
            self.metadata_head = _build_metadata_head(
                feature_dim=model.feature_dim,
                metadata_dim=self.metadata_encoder.output_dim,
                hidden_dim=config.metadata_head_hidden,
            ).to(device)

        # Loss function
        # Auto-computes pos_weight from training labels if needed.
        # See loss.py header for class imbalance rationale.
        train_labels = self.train_loader.dataset.labels
        self.criterion = build_loss(
            use_weighted_bce=config.use_weighted_bce,
            pos_weight=config.pos_weight,
            train_labels=train_labels,
            device=device,
        )

        # Create output directories (must happen before logger)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.predictions_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)

        # Logger (CSV always, W&B if enabled)
        self.logger = TrainLogger(config)

        # Training state
        self.best_val_auroc: float = 0.0
        self.start_epoch: int = 0
        self.start_phase: str = "phase1"

        # Stash for checkpoint resume data. _load_checkpoint stores
        self._resume_checkpoint: dict | None = None

        # Partial backbone warm-start (vessel model: load image backbone
        # weights from a trained image-only checkpoint, ignore the rest)
        if getattr(config, "backbone_checkpoint_path", None):
            self._load_backbone_weights(config.backbone_checkpoint_path)

        # Refuse to silently overwrite a finished run.
        # Every previous relaunch reset best_val_auroc to 0 and then overwrote
        # best.pt with the LAST restart's model rather than the best across
        # restarts, so the surviving checkpoint was not the best one. Fail loudly
        # instead; delete the run directory to start clean, or pass
        # --checkpoint_path to resume properly.
        best_existing = config.checkpoint_dir / "best.pt"
        if best_existing.exists() and config.checkpoint_path is None:
            raise RuntimeError(
                f"{best_existing} already exists and no --checkpoint_path was given.\n"
                f"Starting fresh would overwrite it with a possibly worse model.\n"
                f"  * to restart cleanly: delete {config.run_dir}\n"
                f"  * to resume:          pass --checkpoint_path {config.checkpoint_dir / 'last.pt'}"
            )

        self._start_time = time.time()

        # Save resolved config (exists even if job is killed)
        config.save(config.run_dir / "config_resolved.yaml")

    #  RUN — main entry point
    def run(self) -> dict:
        """Execute full training lifecycle. Returns metrics dict."""
        results: dict[str, Any] = {}

        # Resume if checkpoint provided
        if self.config.checkpoint_path:
            self._load_checkpoint(self.config.checkpoint_path)

        # Phase 1
        if not self.config.skip_phase1 and self.start_phase == "phase1":
            phase1_auroc = self._run_phase1()
            results["phase1_auroc"] = phase1_auroc

        # Phase 2
        if self.config.phase2_epochs > 0:
            phase2_auroc = self._run_phase2()
            results["phase2_auroc"] = phase2_auroc

        # Final evaluation with best checkpoint
        best_path = self.config.checkpoint_dir / "best.pt"
        if best_path.exists():
            self._load_model_weights(best_path)
            final_metrics = self._evaluate_final()
            results["final_metrics"] = final_metrics
            results["best_val_auroc"] = final_metrics.get("auroc", self.best_val_auroc)
        else:
            results["best_val_auroc"] = self.best_val_auroc

        # Provenance record (written even if W&B is unavailable)
        try:
            self._write_manifest(results)
        except Exception as exc:  # never let bookkeeping kill a finished run
            print(f"  [warn] could not write run manifest: {exc}")

        # Log summary to W&B
        self.logger.log_summary(results)
        self.logger.finish()

        return results

    #  PHASE 1: LINEAR PROBE
    def _run_phase1(self) -> float:
        """Frozen backbone, head-only training. Returns final val AUROC."""
        config = self.config
        print(f"\n{'─' * 50}")
        print(f"Phase 1: Linear Probe ({config.phase1_epochs} epochs)")
        print(f"{'─' * 50}")

        # Freeze backbone, lock it in eval mode (BatchNorm/dropout frozen),
        # keep head in train mode. The lock prevents model.train() from
        # silently reverting the backbone — see BaseClassifier.train().
        self.model.set_linear_probe_mode()

        # Build optimiser on head params only (flat LR, no decay).
        # In metadata mode, optimise metadata_head instead.
        # For vessel models, also include vessel backbone (new branch).
        if self.metadata_head is not None:
            head_params = list(self.metadata_head.parameters())
        else:
            head_params = list(self.model.head.parameters())

        extra_modules = getattr(self.model, "extra_modules", None)
        if extra_modules:
            for module in extra_modules:
                head_params += [p for p in module.parameters() if p.requires_grad]

        optimiser = AdamW(head_params, lr=config.phase1_lr)

        trainable = sum(p.numel() for p in head_params if p.requires_grad)
        print(f"  Trainable params: {trainable:,} (head only)")

        # Training loop
        final_auroc = 0.0

        for epoch in range(config.phase1_epochs):
            epoch_start = time.time()

            train_loss = self._train_one_epoch(optimiser)
            val_loss, val_metrics = self._validate_one_epoch()
            elapsed = time.time() - epoch_start

            auroc = val_metrics.get("auroc", 0.0)
            final_auroc = auroc

            if auroc > self.best_val_auroc:
                self.best_val_auroc = auroc

            self.logger.log_epoch(
                epoch=epoch,
                phase="phase1",
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                lr=config.phase1_lr,
                elapsed=elapsed,
            )

            print(
                f"  Epoch {epoch}/{config.phase1_epochs - 1} — "
                f"train_loss: {train_loss:.4f}, "
                f"val_loss: {val_loss:.4f}, "
                f"val_auroc: {auroc:.4f} "
                f"({elapsed:.1f}s)"
            )

        # Save single checkpoint after Phase 1 completes.
        # Serves as Phase 2 starting point.
        last_path = self.config.checkpoint_dir / "last.pt"
        self._save_checkpoint(
            path=last_path,
            epoch=config.phase1_epochs - 1,
            phase="phase1",
            optimiser=optimiser,
            scheduler=None,
            metrics=val_metrics,
        )

        # Also save as best.pt if no Phase 2 will run (phase1_only mode)
        if config.phase2_epochs == 0:
            best_path = self.config.checkpoint_dir / "best.pt"
            self._save_checkpoint(
                path=best_path,
                epoch=config.phase1_epochs - 1,
                phase="phase1",
                optimiser=optimiser,
                scheduler=None,
                metrics=val_metrics,
            )

        print(f"  Phase 1 complete. Best AUROC: {self.best_val_auroc:.4f}")
        return final_auroc

    # PHASE 2: FINE-TUNE
    def _run_phase2(self) -> float:
        """Full fine-tuning with early stopping. Returns best val AUROC."""
        config = self.config
        print(f"\n{'─' * 50}")
        print(f"Phase 2: Fine-tune ({config.phase2_epochs} epochs)")
        print(f"{'─' * 50}")

        self.model.set_finetune_mode()

        self.train_loader, self.inner_val_loader, self.val_loader = rebuild_loaders(
            self.train_loader, self.inner_val_loader, self.val_loader,
            batch_size=config.phase2_batch_size, config=config,
        )

        param_groups = build_layer_wise_param_groups(
            model=self.model,
            base_lr=config.phase2_lr,
            lr_decay_factor=config.lr_decay_factor,
            weight_decay=config.weight_decay,
            metadata_head=self.metadata_head,
            extra_modules=getattr(self.model, "extra_modules", None),
        )
        optimiser = AdamW(param_groups)

        # Anneal over the REALISTIC horizon, not the 50-epoch cap. With
        # total_epochs=50 and early stopping at 10-33, the cosine multiplier was
        # still 0.97 when training ended, so the LR never decayed and every model
        # fine-tuned at an effectively constant rate. phase2_epochs remains the
        # hard cap and patience is untouched; only the annealing horizon changes.
        scheduler = build_cosine_warmup_scheduler(
            optimiser=optimiser,
            warmup_epochs=config.warmup_epochs,
            total_epochs=(config.scheduler_total_epochs or config.phase2_epochs),
            min_lr=config.min_lr,
            base_lr=config.phase2_lr,
        )

        # Restore optimiser/scheduler state on resume
        if self._resume_checkpoint is not None and self.start_phase == "phase2":
            if "optimiser_state_dict" in self._resume_checkpoint:
                optimiser.load_state_dict(self._resume_checkpoint["optimiser_state_dict"])
            if "scheduler_state_dict" in self._resume_checkpoint and \
                    self._resume_checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(self._resume_checkpoint["scheduler_state_dict"])
            self._resume_checkpoint = None

        trainable = self.model.num_trainable_params()
        if self.metadata_head is not None:
            trainable += sum(p.numel() for p in self.metadata_head.parameters())
        print(f"  Trainable params: {trainable:,} (all layers)")

        patience_counter = 0
        min_delta = config.min_delta
        # Phase 2 tracks its OWN best, starting from zero.
        # Seeding it with Phase 1's best meant fine-tuning had to beat the linear
        # probe before it would write any checkpoint -- and _run_phase1 writes
        # best.pt itself when phase2_epochs == 0. That is how SE-ResNeXt ended up
        # with a best.pt, a predictions file and a row in the Phase 2 summary
        # table while all five of its training logs contain phase-1 epochs only.
        phase2_best_auroc = 0.0

        start = self.start_epoch if self.start_phase == "phase2" else 0

        for epoch in range(start, config.phase2_epochs):
            epoch_start = time.time()

            train_loss = self._train_one_epoch(optimiser)
            val_loss, val_metrics = self._validate_one_epoch()
            scheduler.step()
            elapsed = time.time() - epoch_start

            auroc = val_metrics.get("auroc", 0.0)
            lr = optimiser.param_groups[0]["lr"]

            self.logger.log_epoch(
                epoch=epoch,
                phase="phase2",
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                lr=lr,
                elapsed=elapsed,
            )

            print(
                f"  Epoch {epoch}/{config.phase2_epochs - 1} — "
                f"train_loss: {train_loss:.4f}, "
                f"val_loss: {val_loss:.4f}, "
                f"val_auroc: {auroc:.4f}, "
                f"lr: {lr:.2e} "
                f"({elapsed:.1f}s)"
            )

            last_path = self.config.checkpoint_dir / "last.pt"
            self._save_checkpoint(
                path=last_path,
                epoch=epoch,
                phase="phase2",
                optimiser=optimiser,
                scheduler=scheduler,
                metrics=val_metrics,
            )

            if auroc > phase2_best_auroc + min_delta:
                phase2_best_auroc = auroc
                self.best_val_auroc = max(self.best_val_auroc, auroc)
                patience_counter = 0

                best_path = self.config.checkpoint_dir / "best.pt"
                self._save_checkpoint(
                    path=best_path,
                    epoch=epoch,
                    phase="phase2",
                    optimiser=optimiser,
                    scheduler=scheduler,
                    metrics=val_metrics,
                )
                print(f"    ↑ New best AUROC: {auroc:.4f} — saved best.pt")
            else:
                patience_counter += 1
                print(f"    No improvement ({patience_counter}/{config.patience})")

            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch} (patience {config.patience} exhausted)")
                break

        print(f"  Phase 2 complete. Best AUROC: {phase2_best_auroc:.4f}")
        return phase2_best_auroc

    #  CORE LOOPS: shared by both phases
    def _train_one_epoch(self, optimiser: AdamW) -> float:
        """One training epoch. Returns average loss."""
        self.model.train()
        if self.metadata_head is not None:
            self.metadata_head.train()

        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="  Train", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            logits = self._forward(batch, images)
            loss = self.criterion(logits, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

        return running_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_one_epoch(self) -> tuple[float, dict]:
        """One val epoch. Returns (avg_loss, metrics_dict)."""
        self.model.eval()
        if self.metadata_head is not None:
            self.metadata_head.eval()

        running_loss = 0.0
        n_batches = 0
        all_probs = []
        all_labels = []

        # Inner validation, never the reported fold. Early stopping and
        # best-checkpoint selection both read this, so using the outer fold here
        # is what made every previously reported AUROC optimistically biased.
        pbar = tqdm(self.inner_val_loader, desc=" InVal", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            logits = self._forward(batch, images)
            loss = self.criterion(logits, labels)

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = running_loss / max(n_batches, 1)
        y_prob = np.concatenate(all_probs, axis=0).ravel()
        y_true = np.concatenate(all_labels, axis=0).ravel()
        metrics = compute_metrics(y_true, y_prob)

        return avg_loss, metrics

    #  FORWARD PASS: image-only / vessel / metadata / vessel+metadata
    def _forward(self, batch: dict, images: torch.Tensor) -> torch.Tensor:
        """Routes forward pass based on active modalities."""
        vessel_mask = None
        if getattr(self.config, "use_vessel_masks", False):
            vessel_mask = batch["vessel_mask"].to(self.device, non_blocking=True)

        if self.metadata_head is not None:
            if vessel_mask is not None:
                features = self.model.get_features(images, vessel_mask)
            else:
                features = self.model.get_features(images)
            metadata = batch["metadata"].to(self.device, non_blocking=True)
            combined = torch.cat([features, metadata], dim=1)
            return self.metadata_head(combined)
        elif vessel_mask is not None:
            return self.model(images, vessel_mask)
        else:
            return self.model(images)

    #  CHECKPOINTING
    def _save_checkpoint(
        self, path: Path, epoch: int, phase: str,
        optimiser: AdamW, scheduler: LambdaLR | None,
        metrics: dict,
    ) -> None:
        """Save checkpoint with everything needed for exact resumption."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "phase": phase,
            "seed": int(self.config.seed),
            "input_size": int(self.model.preprocessing_config.input_size),
            "best_val_auroc": self.best_val_auroc,
            "metrics": metrics,
            "config": self.config.to_dict(),
            "rng_states": get_rng_states(),
        }
        if self.metadata_head is not None and self.metadata_encoder is not None:
            checkpoint["metadata_encoder"] = self.metadata_encoder.to_dict()
            checkpoint["metadata_head_state_dict"] = self.metadata_head.state_dict()

        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint for resumption. See docstring RESUME NOTE."""
        print(f"  Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "metadata_head_state_dict" in checkpoint and self.metadata_head is not None:
            self.metadata_head.load_state_dict(checkpoint["metadata_head_state_dict"])
        if "metadata_encoder" in checkpoint and self.metadata_encoder is not None:
            self.metadata_encoder = MetadataEncoder.from_dict(checkpoint["metadata_encoder"])

        if self.config.resume_weights_only:
            print("  Note: resume_weights_only=True. Ignoring optimizer and epoch state.")
            return

        self.start_epoch = checkpoint["epoch"] + 1
        self.start_phase = checkpoint["phase"]
        self.best_val_auroc = checkpoint.get("best_val_auroc", 0.0)

        if "rng_states" in checkpoint:
            set_rng_states(checkpoint["rng_states"])

        # Stash for optimiser/scheduler restoration in _run_phase2
        self._resume_checkpoint = checkpoint

        print(f"  Resumed: phase={self.start_phase}, epoch={self.start_epoch}, "
              f"best_auroc={self.best_val_auroc:.4f}")

    def _load_model_weights(self, path: Path) -> None:
        """Load only model weights (for final eval with best.pt)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "metadata_head_state_dict" in checkpoint and self.metadata_head is not None:
            self.metadata_head.load_state_dict(checkpoint["metadata_head_state_dict"])

    def _load_backbone_weights(self, path: str) -> None:
        """Partial warm-start: copy only shape-matching keys from checkpoint.

        Used to initialise the image backbone of EfficientNetVessel from a
        trained image-only checkpoint. Keys that don't exist in the current
        model or have a shape mismatch (e.g. the head, or vessel backbone keys
        that don't appear in the source checkpoint) are silently skipped.
        """
        print(f"  Partial backbone warm-start from: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        source_state = checkpoint["model_state_dict"]
        target_state = self.model.state_dict()

        matched = {
            k: v for k, v in source_state.items()
            if k in target_state and v.shape == target_state[k].shape
        }
        skipped = len(source_state) - len(matched)

        target_state.update(matched)
        self.model.load_state_dict(target_state)
        print(f"  Loaded {len(matched)} matching keys, skipped {skipped} "
              f"(shape mismatch or new layers — kept pretrained init)")

    #  FINAL EVALUATION
    def _predict(self, loader: DataLoader, split: str) -> pd.DataFrame:
        """Run inference over one loader; return ONE ROW PER IMAGE.

        Every row carries the identity needed to join a prediction back to its
        image, patient, cohort and attribution map. The previous schema had only
        patient_id / y_true / y_prob / y_pred, so a prediction could not be tied
        to an image at all and image_id had to be recovered by a positional join
        against the metadata.

        y_logit is stored RAW. scripts/calibrate.py used to recover logits by
        inverting a CLAMPED sigmoid -- clamp(p, 1e-7, 1-1e-7) -- which destroys
        the tails and loses precision near 0 and 1, so every temperature fitted
        that way was contaminated. Fit temperature from y_logit instead.
        """
        self.model.eval()
        if self.metadata_head is not None:
            self.metadata_head.eval()

        ident_keys = ("sample_id", "image_path", "patient_id", "source_dataset")
        ident: dict[str, list] = {k: [] for k in ident_keys}
        logits_all, probs_all, labels_all = [], [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  {split:<9}", leave=False):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                logits = self._forward(batch, images)
                probs = torch.sigmoid(logits)

                logits_all.append(logits.float().cpu().numpy())
                probs_all.append(probs.float().cpu().numpy())
                labels_all.append(labels.cpu().numpy())
                for k in ident_keys:
                    ident[k].extend(batch[k])

        y_logit = np.concatenate(logits_all, axis=0).ravel()
        y_prob = np.concatenate(probs_all, axis=0).ravel()
        y_true = np.concatenate(labels_all, axis=0).ravel()

        return pd.DataFrame(
            {
                "sample_id": ident["sample_id"],
                "patient_id": ident["patient_id"],
                "source_dataset": ident["source_dataset"],
                "split": split,
                "y_true": y_true,
                "y_logit": y_logit,
                "y_prob": y_prob,
                "y_pred": (y_prob >= 0.5).astype(int),
                "image_path": ident["image_path"],
            }
        )

    def _evaluate_final(self) -> dict:
        """Score the best checkpoint on BOTH splits and write per-image CSVs.

        inner_val is written so temperature and the operating threshold can be
        fitted off the reported fold. val is the held-out outer fold and is the
        number that gets reported; its filename is unchanged, so
        aggregate_folds.py, calibrate.py, summarize_all_models.py and
        xai/check_frame.py keep working.
        """
        print(f"\n{'-' * 50}")
        print("Final Evaluation (best checkpoint)")
        print(f"{'-' * 50}")

        self.config.predictions_dir.mkdir(parents=True, exist_ok=True)

        inner_df = self._predict(self.inner_val_loader, "inner_val")
        inner_df.to_csv(self.config.inner_val_predictions_path, index=False)
        inner_metrics = compute_metrics(
            inner_df["y_true"].values, inner_df["y_prob"].values)

        val_df = self._predict(self.val_loader, "val")
        val_df.to_csv(self.config.val_predictions_path, index=False)
        metrics = compute_metrics(val_df["y_true"].values, val_df["y_prob"].values)

        print(f"  {'':<14}{'inner_val':>11}{'val (held out)':>16}")
        for key in ("auroc", "auprc", "ece", "f1", "sensitivity", "specificity"):
            print(f"  {key:<14}{inner_metrics[key]:>11.4f}{metrics[key]:>16.4f}")
        print(f"  {'n_samples':<14}{inner_metrics['n_samples']:>11}{metrics['n_samples']:>16}")
        print(f"  {'patients':<14}{inner_df['patient_id'].nunique():>11}"
              f"{val_df['patient_id'].nunique():>16}")

        # Within-cohort breakdown, free when source_dataset varies.
        if val_df["source_dataset"].nunique() > 1:
            print("\n  val AUROC by source cohort:")
            for src, grp in val_df.groupby("source_dataset"):
                m = compute_metrics(grp["y_true"].values, grp["y_prob"].values)
                print(f"    {src:<16} n={len(grp):<7} prevalence={m['prevalence']:.4f} "
                      f"auroc={m['auroc']:.4f}")

        print(f"\n  Predictions: {self.config.val_predictions_path}")
        print(f"               {self.config.inner_val_predictions_path}")

        self._final_inner_metrics = inner_metrics
        return metrics

    def _write_manifest(self, results: dict) -> None:
        """Provenance record, one JSON per run directory.

        The two data hashes are the point: they are what lets the paper state,
        provably, that every model saw byte-identical data. That claim is the
        strongest methodological credential this project has and it was
        previously unrecorded.
        """
        def _sha256(path) -> str | None:
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                return h.hexdigest()
            except Exception:
                return None

        def _git(*args: str) -> str | None:
            try:
                return subprocess.check_output(
                    ["git", *args], cwd=Path(__file__).resolve().parents[2],
                    stderr=subprocess.DEVNULL, timeout=10,
                ).decode().strip()
            except Exception:
                return None

        cfg = self.config
        best = cfg.checkpoint_dir / "best.pt"
        phase_of_best = epoch_of_best = None
        if best.exists():
            try:
                ck = torch.load(best, map_location="cpu", weights_only=False)
                phase_of_best, epoch_of_best = ck.get("phase"), ck.get("epoch")
            except Exception:
                pass

        manifest = {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "task": cfg.dataset,
            "architecture": cfg.architecture,
            "fold": cfg.fold,
            "seed": int(cfg.seed),
            "deterministic": bool(cfg.deterministic),
            "amp": False,
            "input_size": int(self.model.preprocessing_config.input_size),
            "crop_strategy": self.model.preprocessing_config.crop_strategy,
            "phase_of_best": phase_of_best,
            "epoch_of_best": epoch_of_best,
            "phase2_epochs_cap": cfg.phase2_epochs,
            "scheduler_total_epochs": cfg.scheduler_total_epochs,
            "patience": cfg.patience,
            "min_delta": cfg.min_delta,
            "use_weighted_bce": bool(cfg.use_weighted_bce),
            "pos_weight_config": cfg.pos_weight,
            "use_metadata": bool(cfg.use_metadata),
            "use_vessel_masks": bool(getattr(cfg, "use_vessel_masks", False)),
            "labels_csv": str(cfg.labels_csv),
            "labels_csv_sha256": _sha256(cfg.labels_csv),
            "splits_file": str(cfg.splits_file),
            "splits_file_sha256": _sha256(cfg.splits_file),
            "n_train": len(self.train_loader.dataset),
            "n_inner_val": len(self.inner_val_loader.dataset),
            "n_val": len(self.val_loader.dataset),
            "wall_clock_sec": round(time.time() - getattr(self, "_start_time", time.time()), 1),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "best_val_auroc": results.get("best_val_auroc"),
            "final_metrics": results.get("final_metrics"),
            "inner_val_metrics": getattr(self, "_final_inner_metrics", None),
        }
        cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"  Manifest:    {cfg.manifest_path}")
