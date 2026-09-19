#!/usr/bin/env python3
"""Pre-flight validation for the ISBI retrain grid.

Run this AFTER editing the configs and regenerating the splits, and BEFORE
submitting anything. It loads every config the grid will use, resolves it
through TrainConfig exactly as the trainer does, and checks the things that
have silently gone wrong on this project before.

    python scripts/preflight_check.py                       # default config set
    python scripts/preflight_check.py --seeds 42 1337 2024
    python scripts/preflight_check.py --configs tasks/cimt/configs/phase2/*.yaml

Exit code 0 = safe to submit. 1 = at least one ERROR. WARNs never fail the run.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.schema import TrainConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOBS = [
    "tasks/cimt/configs/phase2/*.yaml",
    "tasks/hypertension/configs/phase2/*.yaml",
]

ERRORS: list[str] = []
WARNS: list[str] = []


def err(tag: str, msg: str) -> None:
    ERRORS.append(f"[{tag}] {msg}")


def warn(tag: str, msg: str) -> None:
    WARNS.append(f"[{tag}] {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# data-level checks, done once per (labels_csv, splits_file) pair
# --------------------------------------------------------------------------
_data_cache: dict[tuple[str, str], dict] = {}


def check_data(cfg: TrainConfig, tag: str) -> dict | None:
    key = (cfg.labels_csv, cfg.splits_file)
    if key in _data_cache:
        return _data_cache[key]

    labels = ROOT / cfg.labels_csv
    splits = ROOT / cfg.splits_file
    if not labels.exists():
        err(tag, f"labels_csv does not exist: {cfg.labels_csv}")
        return None
    if not splits.exists():
        err(tag, f"splits_file does not exist: {cfg.splits_file}")
        return None

    df = pd.read_csv(labels, dtype={cfg.sample_id_col: str})
    dupes = int(df[cfg.sample_id_col].duplicated().sum())
    if dupes:
        err(tag, f"{cfg.labels_csv} has {dupes} duplicate {cfg.sample_id_col} values "
                 f"-- run scripts/fix_combined_metadata.py first")

    payload = json.loads(splits.read_text())
    folds = payload.get("folds", payload if isinstance(payload, list) else [])
    meta = payload if isinstance(payload, dict) else {}
    version = meta.get("version")
    if version != 2:
        err(tag, f"{cfg.splits_file} has version={version!r}, expected 2 "
                 f"-- regenerate with the nested-split generator")
    if meta.get("inner_val_frac") not in (None, cfg.inner_val_frac):
        warn(tag, f"splits were built with inner_val_frac={meta.get('inner_val_frac')} "
                  f"but the config says {cfg.inner_val_frac} -- the splits file wins")
    if cfg.source_col and meta.get("strat_extra_col") != cfg.source_col:
        warn(tag, f"config sets source_col={cfg.source_col!r} but the splits were "
                  f"stratified on strat_extra_col={meta.get('strat_extra_col')!r} -- "
                  f"cohort mix is not pinned across folds")

    # a duplicated sample_id is already an ERROR above; dedupe here only so the
    # remaining checks can still run and report the rest of the problems.
    lookup = df.drop_duplicates(subset=cfg.sample_id_col).set_index(cfg.sample_id_col)
    all_ids = set(df[cfg.sample_id_col].astype(str))
    for entry in folds:
        f = entry.get("fold")
        tr = set(map(str, entry.get("train_sample_ids", [])))
        iv = set(map(str, entry.get("inner_val_sample_ids", [])))
        va = set(map(str, entry.get("val_sample_ids", [])))

        if not iv:
            err(tag, f"fold {f}: inner_val_sample_ids is empty or missing")
        for name, a, b in (("train/inner_val", tr, iv),
                           ("train/val", tr, va),
                           ("inner_val/val", iv, va)):
            if a & b:
                err(tag, f"fold {f}: {len(a & b)} sample_ids appear in both {name}")

        missing = (tr | iv | va) - all_ids
        if missing:
            err(tag, f"fold {f}: {len(missing)} split ids are absent from the labels CSV "
                     f"(e.g. {sorted(missing)[:3]})")

        # patient-level leakage
        if cfg.patient_id_col in df.columns:
            pid = lookup[cfg.patient_id_col].astype(str)
            ptr, piv, pva = (set(pid.reindex(list(s & all_ids)).dropna()) for s in (tr, iv, va))
            for name, a, b in (("train/inner_val", ptr, piv),
                               ("train/val", ptr, pva),
                               ("inner_val/val", piv, pva)):
                if a & b:
                    err(tag, f"fold {f}: {len(a & b)} PATIENTS appear in both {name}")

        # cohort balance
        if cfg.source_col and cfg.source_col in df.columns:
            src = lookup[cfg.source_col].astype(str)
            for split_name, s in (("train", tr), ("inner_val", iv), ("val", va)):
                counts = Counter(src.reindex(list(s & all_ids)).dropna())
                total = sum(counts.values()) or 1
                frac = {k: v / total for k, v in counts.items()}
                entry.setdefault("_cohort", {})[split_name] = frac

    info = {
        "labels_sha256": sha256(labels),
        "splits_sha256": sha256(splits),
        "n_rows": len(df),
        "prevalence": float(df[cfg.label_col].mean()) if cfg.label_col in df else None,
        "folds": folds,
        "source_col": cfg.source_col,
    }
    _data_cache[key] = info
    return info


# --------------------------------------------------------------------------
# per-config checks
# --------------------------------------------------------------------------
def check_config(path: Path, seed: int) -> None:
    tag = f"{path.relative_to(ROOT)}#seed{seed}"
    raw = yaml.safe_load(path.read_text()) or {}
    raw["seed"] = seed
    try:
        cfg = TrainConfig.from_dict(raw)
        cfg.validate()
    except Exception as exc:  # noqa: BLE001
        err(tag, f"config does not load: {type(exc).__name__}: {exc}")
        return

    # 1. the restart guard will fire if best.pt is already there
    best = ROOT / cfg.checkpoint_dir / "best.pt"
    if best.exists() and cfg.checkpoint_path is None:
        err(tag, f"{best.relative_to(ROOT)} already exists and checkpoint_path is unset "
                 f"-- the trainer's restart guard will abort this job at startup")

    # 2. loss
    if cfg.use_weighted_bce:
        warn(tag, "use_weighted_bce=true -- intended only for the single ablation arm")

    # 3. seed must reach run_dir, or two seeds overwrite each other
    rd = str(cfg.run_dir)
    if seed != TrainConfig.REFERENCE_SEED and f"seed{seed}" not in rd:
        err(tag, f"seed {seed} does not appear in run_dir ({rd}) -- runs would collide")

    # 4. site leakage through the metadata head
    if cfg.use_metadata:
        leaky = [c for c in cfg.metadata_columns if c.lower() in {"gender", "sex", "source_dataset", "site"}]
        if leaky and cfg.source_col:
            err(tag, f"metadata_columns contains {leaky} on a multi-cohort dataset -- "
                     f"BRSET codes gender '1'/'2' and ODIR 'Male'/'Female', so this "
                     f"one-hot encodes the source site as a feature")

    # 5. vessel arm
    if cfg.use_vessel_masks:
        vd = ROOT / cfg.vessel_mask_dir
        if not vd.is_dir():
            err(tag, f"use_vessel_masks=true but vessel_mask_dir does not exist: {cfg.vessel_mask_dir}")
        elif not any(vd.iterdir()):
            err(tag, f"vessel_mask_dir is empty: {cfg.vessel_mask_dir}")

    # 6. warm-start checkpoints must not point at the discarded runs
    if cfg.checkpoint_path:
        cp = ROOT / cfg.checkpoint_path
        if not cp.exists():
            err(tag, f"checkpoint_path does not exist: {cfg.checkpoint_path}")
        elif not str(cfg.checkpoint_path).startswith(str(cfg.output_dir).split("/")[0]):
            warn(tag, f"checkpoint_path ({cfg.checkpoint_path}) is outside output_dir "
                      f"({cfg.output_dir}) -- confirm it is not an old discarded run")

    # 7. resolution must be pinned where it is otherwise read from a model card
    if cfg.architecture == "dinov3" and "input_size_override" not in (cfg.model_kwargs or {}):
        warn(tag, "dinov3 takes its input size from the HF processor at runtime -- "
                  "pin input_size_override so a model-card update cannot change it mid-grid")

    # 8. data
    check_data(cfg, tag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    args = ap.parse_args()

    patterns = args.configs or DEFAULT_GLOBS
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(p).resolve() for p in glob.glob(str(ROOT / pat)))
    paths = sorted(set(paths))
    if not paths:
        print(f"no configs matched: {patterns}")
        return 1

    print(f"checking {len(paths)} configs x {len(args.seeds)} seeds "
          f"({len(paths) * len(args.seeds)} jobs)\n")
    for p in paths:
        for s in args.seeds:
            check_config(p, s)

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "tasks", "scripts"],
                                        cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        sha, dirty = "unknown", ""
    print(f"git HEAD: {sha}")
    if dirty:
        warn("git", f"{len(dirty.splitlines())} uncommitted changes under src/ tasks/ scripts/ "
                    f"-- the grid would not be reproducible from this SHA")

    for key, info in _data_cache.items():
        print(f"\ndata: {key[0]}")
        print(f"  rows={info['n_rows']}  prevalence={info['prevalence']:.4f}"
              if info["prevalence"] is not None else f"  rows={info['n_rows']}")
        print(f"  labels sha256 {info['labels_sha256'][:16]}")
        print(f"  splits sha256 {info['splits_sha256'][:16]}")
        for entry in info["folds"]:
            n = (len(entry.get("train_sample_ids", [])),
                 len(entry.get("inner_val_sample_ids", [])),
                 len(entry.get("val_sample_ids", [])))
            line = f"  fold {entry.get('fold')}: train={n[0]} inner_val={n[1]} val={n[2]}"
            coh = entry.get("_cohort")
            if coh:
                parts = []
                for split_name in ("train", "inner_val", "val"):
                    frac = coh.get(split_name, {})
                    parts.append(split_name + "=" + ",".join(f"{k}:{v:.3f}" for k, v in sorted(frac.items())))
                line += "\n      cohort " + " | ".join(parts)
            print(line)

    print()
    for w in WARNS:
        print(f"WARN  {w}")
    for e in ERRORS:
        print(f"ERROR {e}")
    print(f"\n{len(ERRORS)} errors, {len(WARNS)} warnings")
    if ERRORS:
        print("DO NOT SUBMIT -- fix the errors above first.")
        return 1
    print("OK -- safe to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
