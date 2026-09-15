#!/usr/bin/env python3
"""
scripts/fix_combined_metadata.py — de-duplicate a combined metadata CSV.

WHY THIS EXISTS
---------------
data/hypertension_combined/metadata_processed/combined_metadata.csv contains
6,029 fully-duplicated rows out of 26,957 (22.4%), all from ODIR. Every row is
an exact copy -- same image_id, patient_id, label, age, gender, path.

The duplicates do NOT leak across folds: generate_combined_splits.py groups by
patient_id, so copies of an image always land in the same fold. But they do:

  * shrink the effective sample. 26,957 rows are really 20,928 images, so every
    confidence interval and fold std computed on them is too narrow.
  * shift prevalence. 12.56% -> 15.23% once de-duplicated.
  * skew the cohort mix. 53:47 BRSET:ODIR -> 68:32.
  * double the gradient weight of the duplicated images during training, and
    they are overwhelmingly negatives (5,829 neg vs 200 pos).

This script removes them and writes a new file; the original is never modified.

WHAT IT DOES NOT DO
-------------------
It does NOT touch the `gender` column. That column mixes BRSET's numeric codes
('1', '2') with ODIR's strings ('Male', 'Female') because merge_metadata.py
reads the raw `patient_sex` in preference to the processed `gender` column.

  !! OPEN ISSUE !!
  Because the four values partition perfectly by source_dataset, any one-hot
  encoding of `gender` -- including MetadataEncoder.fit() in
  src/data/base_dataset.py -- produces features that identify which cohort an
  image came from. A model given "age + sex" is really being given
  "age + sex + site". Relevant to the metadata-fusion arm and to any
  demographic baseline. Left as-is by explicit decision; see CHANGELOG_ISBI.md.

USAGE
-----
    python scripts/fix_combined_metadata.py \
        --csv data/hypertension_combined/metadata_processed/combined_metadata.csv \
        --out data/hypertension_combined/metadata_processed/combined_metadata_dedup.csv

Add --check-images to verify every referenced file exists (slow on a network
mount; skip it if the data directory is remote).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="De-duplicate a combined metadata CSV.")
    p.add_argument("--csv", required=True, help="Input metadata CSV")
    p.add_argument("--out", required=True, help="Output CSV path (input is never modified)")
    p.add_argument("--sample-id-col", default="image_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--source-col", default="source_dataset",
                   help="Column naming the origin cohort. Ignored if absent.")
    p.add_argument("--image-root", default=None,
                   help="Root for relative_image_path when --check-images is used.")
    p.add_argument("--check-images", action="store_true",
                   help="Verify every referenced image exists. Slow on network mounts.")
    return p.parse_args()


def describe(df: pd.DataFrame, name: str, sid: str, label: str, source: str | None) -> None:
    print(f"  {name}:")
    print(f"    rows                {len(df)}")
    print(f"    unique {sid:<12}{df[sid].nunique()}")
    print(f"    prevalence          {df[label].mean():.4f}")
    if source and source in df.columns:
        mix = df[source].value_counts(normalize=True).to_dict()
        counts = df[source].value_counts().to_dict()
        print(f"    by source           {counts}")
        print(f"    source mix          { {k: round(v, 4) for k, v in mix.items()} }")
        for src, grp in df.groupby(source):
            print(f"      {src:<16} n={len(grp):<7} prevalence={grp[label].mean():.4f}")


def main() -> None:
    args = parse_args()
    csv_path, out_path = Path(args.csv), Path(args.out)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if out_path.resolve() == csv_path.resolve():
        raise ValueError("--out must differ from --csv; this script never overwrites its input.")

    df = pd.read_csv(csv_path)
    sid, label, source = args.sample_id_col, args.label_col, args.source_col

    for col in (sid, label):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in {csv_path}. Have: {list(df.columns)}")

    print("=" * 74)
    print(f"BEFORE  ({csv_path})")
    print("=" * 74)
    describe(df, "input", sid, label, source)

    # --- safety: only collapse rows that are genuinely identical -------------
    dup_ids = df[sid][df[sid].duplicated()].unique()
    conflicts = []
    for gid, grp in df[df[sid].isin(dup_ids)].groupby(sid):
        if grp.drop_duplicates().shape[0] > 1:
            conflicts.append(gid)
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} sample IDs appear more than once with DIFFERENT content "
            f"(e.g. {conflicts[:5]}). These are not safe to collapse automatically -- "
            f"they must be resolved by hand before de-duplication."
        )

    n_exact_dup_rows = int(df.duplicated().sum())
    n_dup_ids = int(df[sid].duplicated().sum())
    print(f"\n  fully-duplicated rows   {n_exact_dup_rows}")
    print(f"  duplicated {sid} rows    {n_dup_ids}")
    if n_exact_dup_rows != n_dup_ids:
        print("  NOTE: these differ, meaning some ids repeat with identical content only "
              "on a subset of columns. The conflict check above passed, so collapsing is safe.")

    clean = df.drop_duplicates(subset=sid, keep="first").reset_index(drop=True)
    assert clean[sid].is_unique, "internal error: ids still not unique after dedup"

    # --- optional: every referenced image must exist -------------------------
    if args.check_images:
        if "relative_image_path" not in clean.columns:
            raise ValueError("--check-images needs a 'relative_image_path' column.")
        root = Path(args.image_root) if args.image_root else csv_path.parents[1] / "images"
        missing = [p for p in clean.relative_image_path if not (root / p).exists()]
        if missing:
            raise ValueError(f"{len(missing)} referenced images missing under {root}, "
                             f"e.g. {missing[:5]}")
        print(f"\n  image existence         OK ({len(clean)} files under {root})")

    print("\n" + "=" * 74)
    print(f"AFTER   ({out_path})")
    print("=" * 74)
    describe(clean, "output", sid, label, source)

    print(f"\n  removed                 {len(df) - len(clean)} rows "
          f"({(len(df) - len(clean)) / max(1, len(df)):.1%})")
    print("\n  gender column left UNCHANGED by explicit decision -- see the module "
          "docstring for the open issue this leaves.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
