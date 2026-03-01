from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from repo root without package installation
# Expected usage from repo root: python scripts/generate_splits.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import generate_splits, validate_splits, save_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate patient-grouped stratified CV splits and save them to JSON."
    )
    p.add_argument("--csv", required=True, help="Path to metadata CSV")
    p.add_argument("--out", required=True, help="Output JSON path for splits")
    p.add_argument("--task-name", required=True, help="Task name (e.g., brset, odir, cimt)")
    p.add_argument("--group-col", required=True, help="Group column (e.g., patient_id or ID)")
    p.add_argument("--label-col", required=True, help="Binary label column for stratification")
    p.add_argument(
        "--sample-id-col",
        default="image_id",
        help="Sample ID column (row/image ID). Default: image_id",
    )
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds (default: 5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed absolute deviation in val positive rate from overall (default: 0.05)",
    )
    p.add_argument(
        "--keep-na-labels",
        action="store_true",
        help="Do not drop NA labels before split generation (not recommended).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standard cleanup for split generation
    if not args.keep_na_labels:
        df = df.dropna(subset=[args.group_col, args.label_col, args.sample_id_col]).copy()

    split_dict = generate_splits(
        df,
        group_col=args.group_col,
        label_col=args.label_col,
        n_splits=args.n_splits,
        seed=args.seed,
        sample_id_col=args.sample_id_col,
        task_name=args.task_name,
    )

    summary = validate_splits(
        split_dict,
        df,
        group_col=args.group_col,
        label_col=args.label_col,
        sample_id_col=args.sample_id_col,
        tolerance=args.tolerance,
    )

    # Print concise summary
    overall = summary["overall"]
    print(
        f"[splits] task={args.task_name} rows={overall['n_rows']} "
        f"patients={overall['n_patients']} pos_rate={overall['positive_rate']:.4f}"
    )
    for f in summary["folds"]:
        print(
            f"[fold {f['fold']}] "
            f"train_samples={f['n_train_samples']} val_samples={f['n_val_samples']} "
            f"train_patients={f['n_train_patients']} val_patients={f['n_val_patients']} "
            f"train_pos={f['train_pos_rate']:.4f} val_pos={f['val_pos_rate']:.4f}"
        )

    if summary["errors"]:
        print("\nValidation issues:")
        for err in summary["errors"]:
            print(f" - {err}")
        raise SystemExit("Split validation failed. Not saving JSON.")

    save_splits(split_dict, out_path)
    print(f"\nSaved splits to: {out_path}")


if __name__ == "__main__":
    main()
