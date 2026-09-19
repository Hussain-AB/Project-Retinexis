#!/usr/bin/env python3
"""Validate a finished run directory before trusting the pipeline.

    python scripts/smoke_check.py outputs_smoke/cimt/efficientnet/fold_0

Checks the things a training run can get silently wrong: predictions that do
not correspond to their logits, a split that leaked, a manifest that records a
commit the code did not run from, a "best" checkpoint that came from phase 1.

Exit 0 = the pipeline produced a trustworthy run. Exit 1 = do not submit a grid.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLS = [
    "sample_id", "patient_id", "source_dataset", "split",
    "y_true", "y_logit", "y_prob", "y_pred", "image_path",
]

ERRORS: list[str] = []
WARNS: list[str] = []
PASSES: list[str] = []


def check(cond: bool, ok: str, bad: str, fatal: bool = True) -> bool:
    if cond:
        PASSES.append(ok)
    else:
        (ERRORS if fatal else WARNS).append(bad)
    return cond


def load_preds(path: Path, label: str) -> pd.DataFrame | None:
    if not check(path.exists(), f"{label} exists", f"{label} missing: {path}"):
        return None
    df = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str})

    check(list(df.columns) == EXPECTED_COLS,
          f"{label} has the 9-column schema",
          f"{label} columns are {list(df.columns)}, expected {EXPECTED_COLS}")
    check(len(df) > 0, f"{label} is non-empty", f"{label} has 0 rows")

    if {"y_logit", "y_prob"} <= set(df.columns):
        lo, pr = df["y_logit"].to_numpy(float), df["y_prob"].to_numpy(float)
        check(np.isfinite(lo).all() and np.isfinite(pr).all(),
              f"{label} has no NaN/inf in y_logit or y_prob",
              f"{label} contains NaN or inf")
        if np.isfinite(lo).all():
            recomputed = 1.0 / (1.0 + np.exp(-lo))
            worst = float(np.max(np.abs(recomputed - pr))) if len(lo) else 0.0
            check(worst < 1e-5,
                  f"{label}: sigmoid(y_logit) == y_prob (max diff {worst:.2e})",
                  f"{label}: y_prob does not match sigmoid(y_logit), max diff {worst:.3e} "
                  f"-- the logits are not the logits behind these probabilities")
    if {"y_prob", "y_pred"} <= set(df.columns):
        implied = (df["y_prob"].to_numpy(float) >= 0.5).astype(int)
        n_bad = int((implied != df["y_pred"].to_numpy(int)).sum())
        check(n_bad == 0,
              f"{label}: y_pred == (y_prob >= 0.5)",
              f"{label}: {n_bad} rows where y_pred disagrees with y_prob at 0.5")
    if "y_true" in df.columns:
        check(set(df["y_true"].unique()) <= {0, 1},
              f"{label}: y_true is binary",
              f"{label}: y_true has values {sorted(set(df['y_true'].unique()))}")
    if "sample_id" in df.columns:
        d = int(df["sample_id"].duplicated().sum())
        check(d == 0, f"{label}: no duplicate sample_id",
              f"{label}: {d} duplicated sample_id rows -- an image was scored twice")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--expect-phase", default="phase2")
    ap.add_argument("--expect-seed", type=int, default=None)
    args = ap.parse_args()

    rd: Path = args.run_dir
    if not rd.is_dir():
        print(f"not a directory: {rd}")
        return 1
    print(f"run_dir: {rd}\n")

    # The trainer writes these under run_dir/predictions/ (see
    # TrainConfig.val_predictions_path). Accept the run_dir root too, so an
    # older layout still validates rather than reporting a false failure.
    def find(name: str) -> Path:
        for cand in (rd / "predictions" / name, rd / name):
            if cand.exists():
                return cand
        return rd / "predictions" / name

    val = load_preds(find("val_predictions.csv"), "val_predictions.csv")
    inner = load_preds(find("inner_val_predictions.csv"), "inner_val_predictions.csv")

    if val is not None and inner is not None:
        vs, ins = set(val["sample_id"]), set(inner["sample_id"])
        check(not (vs & ins), "inner-val and val share no images",
              f"{len(vs & ins)} images appear in BOTH inner_val and val -- "
              f"model selection leaked into the reported split")
        vp, ip = set(val["patient_id"]), set(inner["patient_id"])
        check(not (vp & ip), "inner-val and val share no patients",
              f"{len(vp & ip)} PATIENTS appear in both inner_val and val")
        check(val["split"].nunique() == 1 and inner["split"].nunique() == 1,
              "each file carries a single split label",
              "a predictions file mixes split labels")

    mpath = rd / "run_manifest.json"
    if check(mpath.exists(), "run_manifest.json exists", f"missing {mpath}"):
        man = json.loads(mpath.read_text())

        check(bool(man.get("git_sha")), f"manifest records git_sha {str(man.get('git_sha'))[:12]}",
              "manifest git_sha is null -- the run cannot be tied to any commit")
        check(man.get("git_dirty") is False,
              "tree was clean at run time",
              "git_dirty=true -- the recorded SHA does not describe the code that ran",
              fatal=False)
        for k in ("labels_csv_sha256", "splits_file_sha256"):
            check(bool(man.get(k)), f"manifest records {k}",
                  f"manifest {k} is null -- cannot prove which data this run saw")
        check(man.get("amp") is False, "fp32 confirmed (amp=false)",
              f"manifest says amp={man.get('amp')}, expected false")
        check(man.get("phase_of_best") == args.expect_phase,
              f"best checkpoint came from {args.expect_phase}",
              f"phase_of_best={man.get('phase_of_best')!r}, expected {args.expect_phase!r} "
              f"-- a phase-1 probe was reported as a fine-tuned result")
        isz = man.get("input_size")
        check(isinstance(isz, int) and isz > 0, f"input_size recorded ({isz})",
              f"input_size is {isz!r}")
        if args.expect_seed is not None:
            check(man.get("seed") == args.expect_seed,
                  f"seed {args.expect_seed} recorded",
                  f"manifest seed is {man.get('seed')!r}, expected {args.expect_seed}")
        if val is not None:
            check(man.get("n_val") == len(val),
                  f"n_val matches the predictions file ({len(val)})",
                  f"manifest n_val={man.get('n_val')} but val_predictions.csv has {len(val)} rows")
        if inner is not None:
            check(man.get("n_inner_val") == len(inner),
                  f"n_inner_val matches the predictions file ({len(inner)})",
                  f"manifest n_inner_val={man.get('n_inner_val')} but the file has {len(inner)} rows")

        sf = man.get("splits_file")
        if sf and Path(sf).exists() and val is not None and inner is not None:
            payload = json.loads(Path(sf).read_text())
            entry = next((e for e in payload.get("folds", []) if e.get("fold") == man.get("fold")), None)
            if entry:
                check(set(map(str, entry["val_sample_ids"])) == set(val["sample_id"]),
                      "val rows are exactly the fold's val_sample_ids",
                      "val_predictions.csv does not match the splits file for this fold")
                check(set(map(str, entry["inner_val_sample_ids"])) == set(inner["sample_id"]),
                      "inner-val rows are exactly the fold's inner_val_sample_ids",
                      "inner_val_predictions.csv does not match the splits file for this fold")

        if man.get("task", "").startswith("hypertension") and val is not None:
            filled = val["source_dataset"].notna() & (val["source_dataset"].astype(str) != "")
            check(bool(filled.all()), "source_dataset populated on every val row",
                  f"{int((~filled).sum())} val rows have an empty source_dataset -- "
                  f"per-cohort AUROC will be wrong")

    check((rd / "checkpoints" / "best.pt").exists(),
          "best.pt written", f"missing {rd / 'checkpoints' / 'best.pt'}")

    for line in PASSES:
        print(f"  PASS   {line}")
    for line in WARNS:
        print(f"  WARN   {line}")
    for line in ERRORS:
        print(f"  ERROR  {line}")
    print(f"\n{len(PASSES)} passed, {len(WARNS)} warnings, {len(ERRORS)} errors")
    if ERRORS:
        print("PIPELINE NOT TRUSTWORTHY -- do not submit the grid.")
        return 1
    print("Pipeline looks sound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
