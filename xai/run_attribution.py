"""
STEP 2 -- compute attribution maps.

Fold-matched and strictly out-of-fold: for fold k, every model's fold-k
checkpoint is scored on the images held out in fold k. Maps are written one
.npz per (task, method, model, fold) so the job is resumable -- an overnight
run that dies at 3am picks up where it stopped.

    python -m xai.run_attribution --task cimt --method rise \
        --vessel-dir data/cimt/vessel_masks --n-images 200 --n-masks 2000

COST. RISE is N forward passes per image per model. The two large ViTs
dominate: DINOv2-L runs at 504px (1296 tokens) and RETFound-DINOv2 at 518px,
so they are roughly 5-8x the cost of ConvNeXtV2 at 224px. On one modern GPU
with --amp, budget very roughly:

    efficientnet_b2 / convnextv2 / retfound_mae / dinov3   ~2-4 s per image
    dinov2_large / retfound_dinov2                        ~15-25 s per image

At n_images=200, n_masks=2000, six models, one fold: ~4-6 h per task. Start
with --n-images 200 --folds 0 to get a complete pipeline end-to-end, then
widen. `--method ig` is ~50x cheaper if you need a same-day answer.

CONVERGENCE. Every map carries split_half_r, the correlation between two
independent halves of the mask set. If the median is below 0.90, raise
--n-masks; a map that has not converged will read as 'model disagreement'.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .attribution import integrated_gradients, occlusion, rise
from .folddata import prepare_fold
from .frames import CANON_SIZE, blurred_baseline, fov_mask, load_canonical
from .modelio import MODEL_KEYS, PROJECT_ROOT, CanonicalScorer, load_model, stratified_subsample


def out_path(root: Path, task: str, method: str, model: str, fold: int) -> Path:
    return root / "outputs" / "xai" / "maps" / task / method / f"{model}_fold{fold}.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cimt", "hypertension"])
    ap.add_argument("--method", default="rise", choices=["rise", "occlusion", "ig"])
    ap.add_argument("--models", nargs="*", default=list(MODEL_KEYS))
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--vessel-dir", default=None)
    ap.add_argument("--n-images", type=int, default=200,
                    help="stratified subsample per fold; 0 = all held-out images")
    ap.add_argument("--n-masks", type=int, default=2000)
    ap.add_argument("--rise-s", type=int, default=7)
    ap.add_argument("--rise-p", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ig-steps", type=int, default=64)
    ap.add_argument("--ig-smooth", type=int, default=0)
    ap.add_argument("--occ-patch", type=int, default=48)
    ap.add_argument("--occ-stride", type=int, default=24)
    ap.add_argument("--canon-size", type=int, default=CANON_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--checkpoint-suffix", default=None,
                    help="for the seed-ceiling runs, e.g. 'seed1' -> best_seed1.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  task={args.task}  method={args.method}")

    for fold in args.folds:
        for key in args.models:
            dest = out_path(PROJECT_ROOT, args.task, args.method, key, fold)
            if args.checkpoint_suffix:
                dest = dest.with_name(f"{key}_fold{fold}_{args.checkpoint_suffix}.npz")
            if dest.exists() and not args.overwrite:
                print(f"[skip] {dest.name} exists")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)

            override = None
            if args.checkpoint_suffix:
                from .modelio import checkpoint_path, load_config, config_path
                cfg0 = load_config(config_path(args.task, key), overrides={"fold": fold})
                override = checkpoint_path(cfg0).with_name(f"best_{args.checkpoint_suffix}.pt")

            try:
                lm = load_model(args.task, key, fold, device, checkpoint_override=override)
            except FileNotFoundError as e:
                print(f"[skip] {key} fold{fold}: {e}".replace("\n", " ")[:120])
                continue

            df = prepare_fold(lm.cfg, fold, args.vessel_dir)
            df = stratified_subsample(df, "_label", args.n_images, seed=args.seed)
            scorer = CanonicalScorer(lm, device, amp=not args.no_amp)

            print(f"\n{key} fold{fold}  n={len(df)}  input={lm.input_size}px")
            t0 = time.time()

            maps, ids, labels, logits, shr, rels = [], [], [], [], [], []
            for i, (_, row) in enumerate(df.iterrows()):
                img = load_canonical(row["_image_path"], args.canon_size)
                base = blurred_baseline(img)

                if args.method == "rise":
                    res = rise(scorer, img, n_masks=args.n_masks, s=args.rise_s,
                               p=args.rise_p, batch_size=args.batch_size,
                               seed=args.seed + i, baseline=base)
                    sal, r, lg = res.saliency, res.split_half_r, res.base_logit
                elif args.method == "occlusion":
                    sal = occlusion(scorer, img, patch=args.occ_patch, stride=args.occ_stride,
                                    batch_size=args.batch_size, baseline=base)
                    r = np.nan
                    lg = float(scorer.score_batch(img[None].astype(np.float32))[0])
                else:
                    sal = integrated_gradients(scorer, img, steps=args.ig_steps,
                                               smooth_samples=args.ig_smooth,
                                               seed=args.seed + i, baseline=base)
                    r = np.nan
                    lg = float(scorer.score_batch(img[None].astype(np.float32))[0])

                maps.append(sal.astype(np.float16))   # half precision: 512x512 x 200 = 100 MB
                ids.append(row["_sample_id"])
                rels.append(row["_rel"])
                labels.append(float(row["_label"]))
                logits.append(lg)
                shr.append(r)

                if (i + 1) % 10 == 0 or i + 1 == len(df):
                    el = time.time() - t0
                    print(f"  {i+1}/{len(df)}  {el/(i+1):.1f}s/img  eta {el/(i+1)*(len(df)-i-1)/60:.0f}m",
                          flush=True)

            np.savez_compressed(
                dest,
                maps=np.stack(maps),
                sample_id=np.array(ids),
                rel_path=np.array(rels),
                label=np.array(labels),
                logit=np.array(logits),
                split_half_r=np.array(shr, dtype=np.float32),
                meta=json.dumps(dict(
                    task=args.task, model=key, fold=fold, method=args.method,
                    canon_size=args.canon_size, n_masks=args.n_masks,
                    rise_s=args.rise_s, rise_p=args.rise_p, ig_steps=args.ig_steps,
                    ig_smooth=args.ig_smooth, occ_patch=args.occ_patch,
                    occ_stride=args.occ_stride, seed=args.seed,
                    input_size=lm.input_size, ckpt=str(lm.ckpt_path),
                    baseline="gaussian_blur_sigma0.05",
                )),
            )
            med = float(np.nanmedian(shr)) if args.method == "rise" else float("nan")
            print(f"  saved {dest.name}  median split-half r = {med:.3f}")
            if args.method == "rise" and np.isfinite(med) and med < 0.90:
                print("  !! split-half r < 0.90 -- RAISE --n-masks before trusting this")

            del lm, scorer
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
