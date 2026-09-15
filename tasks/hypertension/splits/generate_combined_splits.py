#!/usr/bin/env python3
"""
Generate nested, patient-grouped, cohort-stratified 5-fold splits for the
combined BRSET + ODIR hypertension dataset.

This used to re-implement StratifiedGroupKFold inline and never validated the
result. It now calls the shared implementation in src/data/splits.py so CIMT
and hypertension are guaranteed to be split by identical code, and it FAILS
rather than warns if validation does not pass.

Two behaviours differ from the previous version:

  1. Three splits per fold instead of two. `val_sample_ids` keeps its original
     meaning -- the held-out fold that gets reported -- and the new
     `inner_val_sample_ids` is a patient-disjoint slice of the training set used
     for early stopping, checkpoint selection, temperature fitting and threshold
     choice. Without it, the reported number is biased by model selection.

  2. Stratification is on label x source_dataset, not label alone. BRSET is
     20.9% positive and ODIR 3.1%, so balancing only the label let the cohort
     mix drift between folds and added variance unrelated to the model.

Input must be de-duplicated first -- generate_splits() raises on duplicate
sample IDs. Run scripts/fix_combined_metadata.py.

    python tasks/hypertension/splits/generate_combined_splits.py
    python tasks/hypertension/splits/generate_combined_splits.py --inner-val-frac 0.0   # v1 behaviour
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.splits import generate_splits, save_splits, validate_splits  # noqa: E402

DEFAULT_CSV = PROJECT_DIR / "data" / "hypertension_combined" / "metadata_processed" / "combined_metadata_dedup.csv"
DEFAULT_OUT = PROJECT_DIR / "tasks" / "hypertension" / "splits" / "hypertension_combined_5fold_splits.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inner-val-frac", type=float, default=0.15)
    p.add_argument("--strat-extra-col", default="source_dataset",
                   help="Cross the label with this column when stratifying. "
                        "Pass an empty string to stratify on the label alone.")
    args = p.parse_args()

    csv_path, out_path = Path(args.csv), Path(args.out)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run scripts/fix_combined_metadata.py first."
        )

    df = pd.read_csv(csv_path)
    strat = args.strat_extra_col or None
    print(f"[splits] source={csv_path}")
    print(f"[splits] rows={len(df)} patients={df.patient_id.nunique()} "
          f"prevalence={df.label.mean():.4f}")

    split_dict = generate_splits(
        df,
        group_col="patient_id",
        label_col="label",
        sample_id_col="image_id",
        task_name="hypertension_combined",
        n_splits=args.n_splits,
        seed=args.seed,
        inner_val_frac=args.inner_val_frac,
        strat_extra_col=strat,
    )

    summary = validate_splits(split_dict, df, strat_extra_col=strat)

    o = summary["overall"]
    print(f"[splits] overall rows={o['n_rows']} patients={o['n_patients']} "
          f"pos_rate={o['positive_rate']:.4f}")
    for f in summary["folds"]:
        mix = " ".join(f"{k}={v:.3f}" for k, v in sorted(f["val_mix"].items()))
        print(f"[fold {f['fold']}] train={f['n_train_samples']} "
              f"inner_val={f['n_inner_val_samples']} val={f['n_val_samples']} | "
              f"patients {f['n_train_patients']}/{f['n_inner_val_patients']}/{f['n_val_patients']} | "
              f"pos {f['train_pos_rate']:.4f}/{f['inner_val_pos_rate'] or float('nan'):.4f}/{f['val_pos_rate']:.4f} | "
              f"val_mix {mix}")

    if summary["errors"]:
        print("\nValidation issues:")
        for err in summary["errors"]:
            print(f"  - {err}")
        raise SystemExit("Split validation FAILED. Nothing written.")

    save_splits(split_dict, out_path)
    print(f"\n[splits] saved {out_path}")


if __name__ == "__main__":
    main()
