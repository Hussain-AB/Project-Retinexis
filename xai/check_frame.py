"""
STEP 1 -- run this before anything else.

Three preconditions, each of which silently invalidates the whole study if it
fails:

  1. CANONICAL-FRAME AUROC reproduces the recorded val_predictions.csv AUROC.
     The canonical frame drops each model's own crop strategy (see frames.py).
     That is a legitimate methodological choice ONLY if it does not change what
     the models do. If AUROC moves by more than ~0.01, say so in the paper --
     or fall back to per-model frames and accept the confound explicitly.

  2. VESSEL MASKS REGISTER TO THE TRAINING IMAGES. scripts/generate_vessel_masks.py
     defaults to the raw images dir; the configs train on images_preprocessed/.
     Mismatched aspect ratios mean every enrichment number is wrong.

  3. DISC ESTIMATION IS SANE -- disc inside the FOV, not on the rim.

Usage:
    python -m xai.check_frame --task cimt --fold 0
    python -m xai.check_frame --task hypertension --fold 0 --models dinov2_large efficientnet_b2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from .anatomy import estimate_disc, load_vessel_channels
from .folddata import prepare_fold
from .frames import CANON_SIZE, fov_geometry, fov_mask, load_canonical, load_canonical_mask
from .modelio import MODEL_KEYS, PROJECT_ROOT, CanonicalScorer, checkpoint_path, load_model


def recorded_auroc(cfg, project_root: Path) -> float | None:
    run = Path(cfg.output_dir)
    if not run.is_absolute():
        run = project_root / run
    csv = run / cfg.dataset / cfg.architecture / f"fold_{cfg.fold}" / "predictions" / "val_predictions.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return float(roc_auc_score(df["y_true"], df["y_prob"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cimt", "hypertension"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=list(MODEL_KEYS))
    ap.add_argument("--vessel-dir", default=None)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--canon-size", type=int, default=CANON_SIZE)
    ap.add_argument("--tolerance", type=float, default=0.010)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    ok = True

    # ---------------------------------------------------------------- 1. AUROC
    print("=" * 74)
    print("1. canonical-frame AUROC vs recorded fold AUROC")
    print("=" * 74)
    print(f"{'model':<20}{'recorded':>10}{'canonical':>12}{'delta':>10}{'n':>7}  ")

    rows = []
    frame_df = None
    for key in args.models:
        try:
            lm = load_model(args.task, key, args.fold, device)
        except FileNotFoundError as e:
            print(f"{key:<20}  SKIP -- {e}".replace("\n", " ")[:110])
            ok = False
            continue

        df = prepare_fold(lm.cfg, args.fold, args.vessel_dir).head(args.n_images)
        frame_df = df if frame_df is None else frame_df
        scorer = CanonicalScorer(lm, device)

        logits = []
        for i in range(0, len(df), args.batch_size):
            chunk = df.iloc[i:i + args.batch_size]
            batch = np.stack([load_canonical(p, args.canon_size) for p in chunk["_image_path"]])
            logits.append(scorer.score_batch(batch.astype(np.float32)))
        logits = np.concatenate(logits)

        canon = float(roc_auc_score(df["_label"].values, logits))
        rec = recorded_auroc(lm.cfg, PROJECT_ROOT)
        delta = (canon - rec) if rec is not None else np.nan
        flag = "" if (rec is None or abs(delta) <= args.tolerance) else "  <-- CHECK"
        if rec is not None and abs(delta) > args.tolerance:
            ok = False
        print(f"{key:<20}{(f'{rec:.4f}' if rec else 'n/a'):>10}{canon:>12.4f}"
              f"{(f'{delta:+.4f}' if rec else '   n/a'):>10}{len(df):>7}{flag}")
        rows.append(dict(model=key, recorded=rec, canonical=canon, delta=delta, n=len(df)))

        del lm, scorer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n  NOTE: the canonical AUROC is computed on a subsample and on the")
    print("        held-out fold only, so small deltas are expected. A delta")
    print(f"        above {args.tolerance:.3f} means the frame change is not neutral.\n")

    # ------------------------------------------------------- 2. mask alignment
    print("=" * 74)
    print("2. vessel-mask registration")
    print("=" * 74)
    if args.vessel_dir is None:
        print("  no --vessel-dir given; skipping (REQUIRED before run_attribution)\n")
        ok = False
    elif frame_df is None:
        print("  no fold loaded; skipping\n")
    else:
        bad = 0
        checked = 0
        for _, r in frame_df.head(40).iterrows():
            vp = Path(r["_vessel_path"])
            if not vp.exists():
                bad += 1
                continue
            iw, ih = Image.open(r["_image_path"]).size
            vw, vh = Image.open(vp).size
            checked += 1
            if abs((iw / ih) - (vw / vh)) > 0.02:
                bad += 1
        print(f"  checked {checked} pairs, {bad} missing or aspect-mismatched")
        if bad:
            ok = False
            print("  !! Regenerate masks from cfg.data_dir, NOT the raw images dir:")
            print("     python scripts/generate_vessel_masks.py \\")
            print("       --images-dir <cfg.data_dir> --save-dir <vessel_dir>\n")
        else:
            print("  OK\n")

    # ------------------------------------------------------------ 3. disc / QC
    print("=" * 74)
    print("3. optic-disc estimate sanity")
    print("=" * 74)
    if frame_df is not None and args.vessel_dir is not None:
        off = 0
        n = 0
        for _, r in frame_df.head(40).iterrows():
            if not Path(r["_vessel_path"]).exists():
                continue
            img = load_canonical(r["_image_path"], args.canon_size)
            fov = fov_mask(img)
            ves = load_vessel_channels(load_canonical_mask(r["_vessel_path"], args.canon_size))
            disc = estimate_disc(img, fov, ves["vessel"])
            g = fov_geometry(fov)
            d = np.hypot(disc.cx - g.cx, disc.cy - g.cy) / g.radius
            n += 1
            if not fov[int(disc.cy), int(disc.cx)] or d > 0.85:
                off += 1
        if n:
            print(f"  {n - off}/{n} disc estimates inside the FOV and off the rim")
            print("  Now eyeball them: python -m xai.figures disc-qc --task "
                  f"{args.task} --fold {args.fold} --vessel-dir {args.vessel_dir}")
            print("  Report the hit rate over ~30 images in the paper.\n")
    else:
        print("  needs --vessel-dir; skipping\n")

    print("=" * 74)
    print("READY" if ok else "NOT READY -- fix the flagged items above first")
    print("=" * 74)

    out = PROJECT_ROOT / "outputs" / "xai" / f"frame_check_{args.task}_fold{args.fold}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
