"""
Grad-CAM + Integrated Gradients for EVERY held-out image of one trained run,
with the regional numbers computed in the same pass.

One call = one trained run (task, model, fold, seed, tree). The images are the
run's own held-out test fold, read from its predictions/val_predictions.csv, so
the set is exactly what the run was scored on and no image is one it trained on.

    python -m xai.run_maps --task hypertension --model dinov2_large --fold 0 \
        --vessel-dir data/hypertension_combined/vessel_masks

    # smoke test: 16 images, then stop
    python -m xai.run_maps --task cimt --model efficientnet_b2 --fold 0 \
        --vessel-dir data/cimt/vessel_masks --limit 16 --out-root outputs/xai_smoke

Why this runner exists next to run_attribution.py: that one stores full
512 x 512 maps and leaves the region analysis to analyze.py, which is right for
a 200-image subsample but would need ~130 GB for every image of every fold.
This one computes the region metrics at full resolution inside the job and keeps
only a 128 x 128 copy of each map (for agreement statistics and figures).
Region definitions, null, normalisation and the canonical frame are imported
unchanged from anatomy.py / metrics.py / frames.py / attribution.py, so the
numbers are the same quantities analyze.py produces.

Per image and method it records, inside the field of view:
  * for every ROI in anatomy.ROI_ORDER: enrichment, mass, pointing game,
    top-10% IoU, and the same for a smoothness-matched random null
  * the radial profile (16 bins, 0-4 disc diameters)
  * the canonical-frame logit next to the logit the run recorded for that
    image (fidelity: does the canonical frame change what the model does?)
  * IG only: completeness = sum(IG) / (logit(image) - logit(baseline)).
    Values far from 1 mean too few integration steps.

Resumable: work is written in chunks; a restarted job skips finished chunks.
Output:  {out_root}/{task}/{model}/fold{k}_seed{s}[_rep]/
            chunk_XXX_regions.csv  chunk_XXX_images.csv  chunk_XXX_radial.csv
            chunk_XXX_maps.npz     run_info.json         DONE
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .anatomy import ROI_ORDER, build_rois, estimate_disc, load_vessel_channels
from .attribution import integrated_gradients, map_smoothness_sigma, random_smooth_map
from .frames import (CANON_SIZE, blurred_baseline, fov_geometry, fov_mask, load_canonical,
                     load_canonical_mask, normalise_map)
from .gradcam import GradCAM
from .metrics import enrichment, mass_fraction, pointing_game, radial_profile, soft_iou_topk
from .modelio import (MODEL_FAMILY, MODEL_KEYS, PROJECT_ROOT, CanonicalScorer, LoadedModel,
                      config_path, create_model, load_config, load_model, stratified_subsample)

N_RADIAL = 16
LOWRES = 128


# ----------------------------------------------------------------------------
# run and image resolution
# ----------------------------------------------------------------------------
def run_dir_for(task: str, key: str, fold: int, seed: int, tree: str, root: Path) -> tuple[Path, object]:
    cfg = load_config(config_path(task, key, root), overrides={"fold": fold})
    tag = f"fold_{fold}" + ("" if seed == 42 else f"_seed{seed}")
    rd = root / tree / cfg.dataset / cfg.architecture / tag
    return rd, cfg


def resolve_vessel(image_path: str, data_dir: Path, vessel_dir: Path) -> Path | None:
    """Find the RRWNet mask for an image. Tries, in order: the image's path
    relative to the config's data_dir (subfolders kept) with .png, the same
    with its own extension, then the bare file name. The first that exists wins."""
    p = Path(image_path)
    cands = []
    try:
        rel = p.relative_to(data_dir)
        cands += [vessel_dir / rel.with_suffix(".png"), vessel_dir / rel]
    except ValueError:
        pass
    if len(p.parts) >= 2:
        cands += [vessel_dir / p.parent.name / (p.stem + ".png")]
    cands += [vessel_dir / (p.stem + ".png"), vessel_dir / p.name]
    for c in cands:
        if c.exists():
            return c
    return None


def lowres(a: np.ndarray, size: int = LOWRES) -> tuple[np.ndarray, float]:
    """128 x 128 copy, scaled to max |value| = 1 so float16 storage cannot
    underflow (raw IG / ViT Grad-CAM values can be ~1e-7). The scale is kept
    alongside; every downstream statistic is invariant to it."""
    small = cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    s = float(np.abs(small).max())
    return (small / s if s > 0 else small).astype(np.float16), s


def region_rows(sal, fov, rois, disc, null_seed):
    norm = normalise_map(sal, fov)
    null = random_smooth_map(*sal.shape, sigma=map_smoothness_sigma(sal), seed=null_seed)
    null_norm = normalise_map(null, fov)
    fov_area = float(fov.sum())
    rows = []
    for roi_name in ROI_ORDER:
        roi = rois[roi_name]
        rows.append(dict(roi=roi_name, area_frac=float((roi & fov).sum()) / fov_area,
                         enrichment=enrichment(norm, roi, fov), mass=mass_fraction(norm, roi, fov),
                         pointing=pointing_game(sal, roi, fov), iou10=soft_iou_topk(sal, roi, fov, 0.10),
                         null_enrichment=enrichment(null_norm, roi, fov),
                         null_iou10=soft_iou_topk(null, roi, fov, 0.10)))
    return rows, radial_profile(norm, fov, disc, n_bins=N_RADIAL), norm


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cimt", "hypertension"])
    ap.add_argument("--model", required=True, choices=list(MODEL_KEYS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tree", default="outputs_isbi",
                    help="outputs_isbi, or outputs_isbi_rep for the exact-repeat run")
    ap.add_argument("--methods", nargs="+", default=["gradcam", "ig"], choices=["gradcam", "ig"])
    ap.add_argument("--vessel-dir", default=None, help="RRWNet masks; required unless --no-rois")
    ap.add_argument("--no-rois", action="store_true", help="maps and fidelity only (no masks needed)")
    ap.add_argument("--allow-missing-masks", type=float, default=0.01,
                    help="max share of images without a mask before the job refuses to run")
    ap.add_argument("--out-root", default="outputs/xai_full")
    ap.add_argument("--limit", type=int, default=0, help="stratified subsample (smoke tests)")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=16, help="Grad-CAM images per pass")
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--ig-batch", type=int, default=8, help="IG interpolation points per pass")
    ap.add_argument("--gradcam-layer", default=None, help="override the documented default layer")
    ap.add_argument("--canon-size", type=int, default=CANON_SIZE)
    ap.add_argument("--random-init", action="store_true",
                    help="model-randomisation null: same architecture, freshly initialised weights, "
                         "no checkpoint. Run once per model on fold 0 (e.g. --limit 500).")
    args = ap.parse_args()

    root = PROJECT_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd, cfg = run_dir_for(args.task, args.model, args.fold, args.seed, args.tree, root)
    man_f = rd / "run_manifest.json"
    if not man_f.exists():
        raise SystemExit(f"no run at {rd} (run_manifest.json missing)")
    man = json.loads(man_f.read_text())
    if int(man["fold"]) != args.fold or int(man["seed"]) != args.seed:
        raise SystemExit(f"{man_f}: manifest says fold {man['fold']} seed {man['seed']}")
    if man.get("git_dirty"):
        raise SystemExit(f"{man_f}: run was trained from a dirty tree -- refusing")
    ckpt = rd / "checkpoints" / "best.pt"

    tag = (f"fold{args.fold}_seed{args.seed}" + ("_rep" if args.tree.endswith("_rep") else "")
           + ("_randinit" if args.random_init else ""))
    out = Path(args.out_root)
    out = (out if out.is_absolute() else root / out) / args.task / args.model / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print(f"[done] {out} -- delete DONE to recompute")
        return 0

    preds = pd.read_csv(rd / "predictions" / "val_predictions.csv",
                        dtype={"sample_id": str, "patient_id": str})
    preds["_label"] = preds.y_true.astype(float)
    if args.limit:
        preds = stratified_subsample(preds, "_label", args.limit, seed=0)
    preds = preds.reset_index(drop=True)
    preds["_abs"] = [str(p if Path(p).is_absolute() else root / p) for p in preds.image_path]
    missing_img = [p for p in preds._abs if not Path(p).exists()]
    if missing_img:
        raise SystemExit(f"{len(missing_img)} test images not found, e.g. {missing_img[:2]}")

    data_dir = Path(cfg.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else root / data_dir
    if not args.no_rois:
        if not args.vessel_dir:
            raise SystemExit("--vessel-dir is required for region numbers (or pass --no-rois)")
        vd = Path(args.vessel_dir)
        vd = vd if vd.is_absolute() else root / vd
        preds["_vessel"] = [str(resolve_vessel(p, data_dir, vd) or "") for p in preds._abs]
        miss = float((preds._vessel == "").mean())
        print(f"vessel masks: {int((preds._vessel != '').sum())}/{len(preds)} found "
              f"(e.g. {preds._vessel.iloc[0]})")
        if miss > args.allow_missing_masks:
            raise SystemExit(f"{miss:.1%} of test images have no mask under {vd}. Example image: "
                             f"{preds._abs.iloc[0]}. Generate masks from {data_dir} first.")

    if args.random_init:
        # Adebayo et al. (2018) sanity check. If a trained model's map looks like
        # this one, the map reflects the input and the architecture, not what the
        # model learned. Gradient x input methods are known to highlight edges
        # (vessels, disc, FOV rim) even for random weights.
        torch.manual_seed(0)
        kw = dict(cfg.model_kwargs or {})
        if "pretrained" in kw:
            kw["pretrained"] = False
        net = create_model(cfg.architecture, **kw).eval().to(device)
        for prm in net.parameters():
            prm.requires_grad_(False)
        lm = LoadedModel(key=args.model, task=args.task, fold=args.fold, model=net, cfg=cfg,
                         input_size=net.preprocessing_config.input_size, ckpt_path=Path("random-init"))
    else:
        lm = load_model(args.task, args.model, args.fold, device, project_root=root,
                        checkpoint_override=ckpt)
    scorer = CanonicalScorer(lm, device, amp=False)     # fp32 everywhere: gradients and fidelity
    family = MODEL_FAMILY[args.model]
    probe = load_canonical(preds._abs.iloc[0], args.canon_size)
    cam = GradCAM(scorer, family, probe, args.gradcam_layer) if "gradcam" in args.methods else None

    info = dict(task=args.task, model=args.model, family=family, fold=args.fold, seed=args.seed,
                tree=args.tree, run_dir=str(rd.relative_to(root)),
                checkpoint=("random-init" if args.random_init else str(ckpt.relative_to(root))),
                random_init=args.random_init,
                run_git_sha=man.get("git_sha"), input_size=lm.input_size, canon_size=args.canon_size,
                methods=args.methods, ig_steps=args.ig_steps, baseline="gaussian_blur_sigma0.05",
                gradcam_layer=(cam.layer.__dict__ if cam else None), lowres=LOWRES,
                n_images=len(preds), limit=args.limit, rois=not args.no_rois,
                host=socket.gethostname(),
                gpu=(torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"))
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))

    t0 = time.time()
    n_chunks = (len(preds) + args.chunk - 1) // args.chunk
    for c in range(n_chunks):
        stem = out / f"chunk_{c:03d}"
        if Path(f"{stem}_maps.npz").exists():
            continue
        part = preds.iloc[c * args.chunk:(c + 1) * args.chunk]
        img_rows, reg_rows, rad_rows = [], [], []
        lo = {m: [] for m in args.methods}
        lo_scale = {m: [] for m in args.methods}
        lo_fov = []
        for b in range(0, len(part), args.batch_size):
            batch = part.iloc[b:b + args.batch_size]
            imgs = np.stack([load_canonical(p, args.canon_size) for p in batch._abs])
            canon_logit = scorer.score_batch(imgs.astype(np.float32))
            cams = cam(imgs)[0] if cam else None
            for j, (_, r) in enumerate(batch.iterrows()):
                img = imgs[j]
                fov = fov_mask(img)
                geom = None
                if not args.no_rois and r._vessel:
                    ves = load_vessel_channels(load_canonical_mask(r._vessel, args.canon_size))
                    disc = estimate_disc(img, fov, ves["vessel"])
                    geom = (build_rois(img, fov, ves, disc), disc)
                g = fov_geometry(fov)
                base_row = dict(sample_id=r.sample_id, patient_id=r.patient_id,
                                source_dataset=r.source_dataset, label=float(r.y_true),
                                recorded_logit=float(r.y_logit), canonical_logit=float(canon_logit[j]),
                                disc_x=(geom[1].cx if geom else np.nan), disc_y=(geom[1].cy if geom else np.nan),
                                fov_cx=g.cx, fov_cy=g.cy, fov_r=g.radius)
                maps = {}
                if cam:
                    maps["gradcam"] = cams[j]
                if "ig" in args.methods:
                    base_img = blurred_baseline(img)
                    ig = integrated_gradients(scorer, img, steps=args.ig_steps, batch_size=args.ig_batch,
                                              baseline=base_img)
                    lb = float(scorer.score_batch(base_img[None].astype(np.float32))[0])
                    maps["ig"] = ig
                    gap = float(canon_logit[j]) - lb
                    base_row["ig_baseline_logit"] = lb
                    base_row["ig_completeness"] = float(ig.sum()) / gap if abs(gap) > 1e-6 else np.nan
                for m, sal in maps.items():
                    row = dict(base_row, method=m, map_max=float(sal.max()), map_min=float(sal.min()),
                               positive_share_in_fov=float(np.clip(sal[fov], 0, None).sum()
                                                           / max(np.abs(sal[fov]).sum(), 1e-12)))
                    if geom:
                        rr, prof, _ = region_rows(sal, fov, geom[0], geom[1], null_seed=int(c * 10**6 + b + j))
                        for x in rr:
                            reg_rows.append(dict(sample_id=r.sample_id, method=m, **x))
                        rad_rows.append(dict(sample_id=r.sample_id, method=m,
                                             **{f"r{k:02d}": float(v) for k, v in enumerate(prof)}))
                    img_rows.append(row)
                    small, scale = lowres(sal)
                    lo[m].append(small)
                    lo_scale[m].append(scale)
                lo_fov.append(cv2.resize(fov.astype(np.float32), (LOWRES, LOWRES), interpolation=cv2.INTER_AREA) > 0.5)
        pd.DataFrame(img_rows).to_csv(f"{stem}_images.csv", index=False)
        pd.DataFrame(reg_rows).to_csv(f"{stem}_regions.csv", index=False)
        pd.DataFrame(rad_rows).to_csv(f"{stem}_radial.csv", index=False)
        np.savez_compressed(f"{stem}_maps.npz", sample_id=part.sample_id.to_numpy().astype(str),
                            fov=np.stack(lo_fov), **{m: np.stack(v) for m, v in lo.items()},
                            **{f"{m}_scale": np.array(v, np.float32) for m, v in lo_scale.items()})
        done = min((c + 1) * args.chunk, len(preds))
        el = time.time() - t0
        print(f"  chunk {c + 1}/{n_chunks}  {done}/{len(preds)} images  "
              f"{el / max(done - c * 0, 1):.2f} s/img so far", flush=True)

    imgs = pd.concat([pd.read_csv(f) for f in sorted(out.glob("chunk_*_images.csv"))])
    d = imgs.drop_duplicates("sample_id")
    fid = dict(n=len(d), max_abs_logit_diff=float((d.canonical_logit - d.recorded_logit).abs().max()),
               median_abs_logit_diff=float((d.canonical_logit - d.recorded_logit).abs().median()),
               corr=float(np.corrcoef(d.canonical_logit, d.recorded_logit)[0, 1]))
    if "ig" in args.methods:
        ig = imgs[imgs.method == "ig"].ig_completeness
        fid["ig_completeness_median"] = float(ig.median())
        fid["ig_completeness_p10_p90"] = [float(ig.quantile(.1)), float(ig.quantile(.9))]
    (out / "DONE").write_text(json.dumps(fid, indent=2))
    print("fidelity:", json.dumps(fid))
    if fid["median_abs_logit_diff"] > 0.25 and not args.random_init:
        print("!! canonical-frame logits differ from the recorded ones by more than 0.25 (median). "
              "Report this, or the maps describe a slightly different input than the model was scored on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
