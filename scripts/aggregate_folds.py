"""
Aggregate Phase 1 results across CV folds.

Reads val_predictions.csv from each fold directory, computes per-fold
metrics, and reports mean ± std. Logs a summary run to W&B grouped
with the individual fold runs.

Usage:
    python scripts/aggregate_folds.py \
        --run-dir outputs/cimt/efficientnet \
        --n-folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics

SUMMARY_METRICS = ["auroc", "auprc", "ece", "f1", "sensitivity", "specificity"]


def main():
    parser = argparse.ArgumentParser(description="Aggregate CV fold results")
    parser.add_argument("--run-dir", required=True, help="e.g. outputs/cimt/efficientnet")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--wandb-project", type=str, default="retinexis")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Skip W&B logging")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fold_metrics = []

    print(f"\n{'─' * 50}")
    print(f"Aggregating folds: {run_dir}")
    print(f"{'─' * 50}")

    for fold in range(args.n_folds):
        csv_path = run_dir / f"fold_{fold}" / "predictions" / "val_predictions.csv"
        if not csv_path.exists():
            print(f"  WARNING: Missing fold {fold} — {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        metrics = compute_metrics(df["y_true"].values, df["y_prob"].values)
        metrics["fold"] = fold
        metrics["n_samples"] = len(df)
        fold_metrics.append(metrics)
        print(f"  Fold {fold}: AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  (n={len(df)})")

    if not fold_metrics:
        print("ERROR: No fold results found.")
        return

    # Compute summary stats
    summary = {}
    print(f"\n{'─' * 50}")
    print(f"Summary ({len(fold_metrics)}/{args.n_folds} folds)")
    print(f"{'─' * 50}")

    for key in SUMMARY_METRICS:
        values = [m[key] for m in fold_metrics if not np.isnan(m.get(key, float("nan")))]
        if values:
            mean, std = np.mean(values), np.std(values)
            summary[key] = {"mean": mean, "std": std}
            print(f"  {key:15s}: {mean:.4f} ± {std:.4f}")

    # Save per-fold CSV
    summary_path = run_dir / "cv_summary.csv"
    pd.DataFrame(fold_metrics).to_csv(summary_path, index=False)
    print(f"\n  Per-fold results saved: {summary_path}")

    # Extract dataset/architecture from run_dir path
    # Expected structure: outputs/{dataset}/{architecture}
    parts = run_dir.parts
    dataset = parts[-2]
    architecture = parts[-1]

    # Log to W&B with same group as individual fold runs so everything
    # appears together in the dashboard
    if not args.no_wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                # Matches the group set in TrainLogger for fold runs
                group=f"{dataset}/{architecture}",
                name=f"{dataset}/{architecture}/cv_summary",
                job_type="aggregate",
                config={
                    "dataset": dataset,
                    "architecture": architecture,
                    "n_folds": len(fold_metrics),
                },
            )

            # Log with same keys as fold runs so they appear
            # in the same W&B table columns
            wandb.summary["best_val_auroc"] = summary["auroc"]["mean"]
            wandb.summary["phase1_auroc"] = summary["auroc"]["mean"]

            # Mean ± std for all metrics
            for key, vals in summary.items():
                wandb.summary[f"cv_mean/{key}"] = vals["mean"]
                wandb.summary[f"cv_std/{key}"] = vals["std"]

            # Log a formatted string for easy reading in the table
            auroc_str = f"{summary['auroc']['mean']:.4f} ± {summary['auroc']['std']:.4f}"
            auprc_str = f"{summary['auprc']['mean']:.4f} ± {summary['auprc']['std']:.4f}"
            wandb.summary["cv_auroc"] = auroc_str
            wandb.summary["cv_auprc"] = auprc_str

            # Log per-fold table (viewable in W&B UI)
            table = wandb.Table(dataframe=pd.DataFrame(fold_metrics))
            wandb.log({"fold_results": table})

            wandb.finish()
            print("  Summary logged to W&B")

        except ImportError:
            print("  W&B not installed — skipping W&B logging")
        except Exception as e:
            print(f"  W&B logging failed: {e}")

    print()


if __name__ == "__main__":
    main()
