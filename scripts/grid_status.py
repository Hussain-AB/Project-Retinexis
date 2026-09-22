#!/usr/bin/env python3
"""Triage the grid: which of the 96 runs produced usable output, and why not.

    python scripts/grid_status.py                     # newest manifest
    python scripts/grid_status.py --manifest <path>
    python scripts/grid_status.py --verbose           # full error tails

Resolves each manifest row through TrainConfig, so the run directory it checks
is exactly the one the trainer wrote to -- no path guessing. Then groups the
failures by error signature, because 20 jobs failing the same way is one problem,
not twenty.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config.schema import TrainConfig  # noqa: E402

# lines worth surfacing from a failed job's stderr
NOISE = re.compile(
    r"^\s*$|^wandb:|^\s*warnings\.warn|UserWarning|^\s*return |^Traceback|"
    r"^\s+File \"|^\s+[a-z_]+ = |^\s*\^+\s*$|^srun:|^slurmstepd:"
)


def signature(text: str) -> str:
    """The most informative single line of a traceback."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    for l in reversed(lines):
        if re.match(r"^[A-Za-z_.]*(Error|Exception|Exit|Killed)\b", l.strip()):
            return l.strip()[:200]
    for l in reversed(lines):
        if not NOISE.match(l):
            return l.strip()[:200]
    return "(no error text found)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--logs", type=Path, default=ROOT / "outputs_isbi" / "_logs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.manifest is None:
        cands = sorted((ROOT / "outputs_isbi" / "_grid").glob("manifest_*.tsv"))
        if not cands:
            print("no manifest found under outputs_isbi/_grid/")
            return 1
        args.manifest = cands[-1]
    print(f"manifest: {args.manifest.relative_to(ROOT)}\n")

    rows = [l.split("\t") for l in args.manifest.read_text().splitlines() if l.strip()]

    complete, partial, missing = [], [], []
    for idx, (task, model, fold, seed, outdir, cfgpath) in enumerate(rows):
        raw = yaml.safe_load((ROOT / cfgpath).read_text()) or {}
        raw.update(fold=int(fold), seed=int(seed), output_dir=outdir)
        cfg = TrainConfig.from_dict(raw)
        rd = ROOT / cfg.run_dir

        have = {
            "val": (ROOT / cfg.val_predictions_path).exists(),
            "inner": (ROOT / cfg.inner_val_predictions_path).exists(),
            "manifest": (ROOT / cfg.manifest_path).exists(),
            "best": (ROOT / cfg.checkpoint_dir / "best.pt").exists(),
        }
        rec = dict(idx=idx, task=task, model=model, fold=fold, seed=seed,
                   run_dir=cfg.run_dir, have=have)
        if all(have.values()):
            try:
                man = json.loads((ROOT / cfg.manifest_path).read_text())
                rec["sha"] = (man.get("git_sha") or "?")[:12]
                rec["dirty"] = man.get("git_dirty")
                rec["auroc"] = (man.get("final_metrics") or {}).get("auroc")
                rec["epochs"] = man.get("epoch_of_best")
                rec["secs"] = man.get("wall_clock_sec")
            except Exception:
                pass
            complete.append(rec)
        elif any(have.values()):
            partial.append(rec)
        else:
            missing.append(rec)

    owners: dict[str, list[int]] = {}
    for rec in complete + partial + missing:
        owners.setdefault(str(rec["run_dir"]), []).append(rec["idx"])
    shared = {k: v for k, v in owners.items() if len(v) > 1}
    if shared:
        print("  COLLISION: manifest rows share a run_dir, so their results are "
              "the SAME run reported more than once:")
        for rd, idxs in sorted(shared.items()):
            print(f"    {rd}  <- rows {idxs}")
        print()

    print(f"  complete : {len(complete):>3} / {len(rows)}")
    print(f"  partial  : {len(partial):>3}   (some artifacts, run did not finish)")
    print(f"  missing  : {len(missing):>3}   (no output at all)\n")

    # --- provenance uniformity, the thing that decides if the grid is citable
    shas = collections.Counter(r.get("sha") for r in complete)
    dirty = sum(1 for r in complete if r.get("dirty"))
    print(f"  git_sha across completed runs: {dict(shas)}")
    if len(shas) > 1:
        print("  WARNING: more than one commit produced this grid -- the tree changed mid-run")
    if dirty:
        print(f"  WARNING: {dirty} completed runs recorded git_dirty=true")
    print()

    # --- failures, grouped by cause
    failed = partial + missing
    if failed:
        groups: dict[str, list] = collections.defaultdict(list)
        for rec in failed:
            text = ""
            for pat in (f"*_{rec['idx']}.err", f"*_{rec['idx']}.out"):
                hits = sorted(args.logs.glob(pat))
                if hits:
                    text += hits[-1].read_text(errors="replace")
            rec["log_found"] = bool(text)
            groups[signature(text) if text else "(no log file found)"].append(rec)

        print(f"{len(failed)} failed runs, {len(groups)} distinct causes:\n")
        for sig, recs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"  [{len(recs):>2}x] {sig}")
            for r in recs[:6]:
                print(f"          row {r['idx']:<3} {r['task']:<13} {r['model']:<17} "
                      f"fold {r['fold']} seed {r['seed']}")
            if len(recs) > 6:
                print(f"          ... and {len(recs) - 6} more")
            print()

        if args.verbose:
            print("=" * 70)
            for rec in failed:
                for pat in (f"*_{rec['idx']}.err",):
                    for f in sorted(args.logs.glob(pat)):
                        print(f"\n--- row {rec['idx']} : {f.name}")
                        tail = [l for l in f.read_text(errors="replace").splitlines()
                                if not NOISE.match(l)]
                        print("\n".join(tail[-25:]))

    print("=" * 70)
    print("completed runs:\n")
    print(f"  {'row':<4}{'task':<14}{'model':<18}{'fold':<5}{'seed':<6}"
          f"{'auroc':<9}{'ep':<4}{'hours':<7}")
    for r in sorted(complete, key=lambda r: r["idx"]):
        au = f"{r['auroc']:.4f}" if isinstance(r.get("auroc"), (int, float)) else "-"
        hrs = f"{r['secs']/3600:.2f}" if isinstance(r.get("secs"), (int, float)) else "-"
        print(f"  {r['idx']:<4}{r['task']:<14}{r['model']:<18}{r['fold']:<5}"
              f"{r['seed']:<6}{au:<9}{str(r.get('epochs','-')):<4}{hrs:<7}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
