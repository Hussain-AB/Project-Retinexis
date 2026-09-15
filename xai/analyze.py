"""
STEP 3 -- turn saved maps into the numbers that go in the paper.

Produces four CSVs under outputs/xai/tables/:

  regional_<task>_<method>.csv   per (image, model, roi): enrichment, mass,
                                 pointing game, soft-IoU, plus the SAME metrics
                                 for a smoothness-matched random null
  radial_<task>_<method>.csv     attribution density vs distance from the disc
  agreement_<task>_<method>.csv  per (image, model_a, model_b): Spearman,
                                 top-10% IoU, centroid shift in disc diameters
  summary_<task>_<method>.csv    bootstrapped means + CIs, and paired
                                 model-vs-null tests with Holm correction

Read the summary first. The claim 'models with equal AUROC use different
evidence' needs, in order:
  (a) the CEILING -- same-architecture agreement (--ceiling-suffix runs)
      clearly above cross-architecture agreement. Without this you have noise.
  (b) enrichment differences that survive Holm correction.
  (c) run_ablation.py showing the enrichment differences have causal bite.

Usage:
    python -m xai.analyze --task cimt --method rise --vessel-dir data/cimt/vessel_masks
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .anatomy import ROI_ORDER, build_rois, estimate_disc, load_vessel_channels
from .attribution import map_smoothness_sigma, random_smooth_map
from .frames import (CANON_SIZE, fov_mask, load_canonical, load_canonical_mask,
                     normalise_map)
from .metrics import (bootstrap_ci, centroid_shift, enrichment, holm_bonferroni,
                      iou_agreement, mass_fraction, paired_bootstrap_diff,
                      pointing_game, radial_profile, soft_iou_topk,
                      spearman_agreement)
from .paths import MODEL_FAMILY, PROJECT_ROOT

N_RADIAL_BINS = 16


def load_maps(task: str, method: str, root: Path = PROJECT_ROOT) -> dict:
    d = root / "outputs" / "xai" / "maps" / task / method
    out = {}
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        stem = f.stem
        suffix = None
        parts = stem.split("_fold")
        model = parts[0]
        rest = parts[1].split("_", 1)
        fold = int(rest[0])
        if len(rest) > 1:
            suffix = rest[1]
        out[(model, fold, suffix)] = dict(
            maps=z["maps"], sample_id=z["sample_id"], rel=z["rel_path"],
            label=z["label"], logit=z["logit"], shr=z["split_half_r"], meta=meta,
        )
    if not out:
        raise FileNotFoundError(f"No maps in {d}. Run run_attribution.py first.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cimt", "hypertension"])
    ap.add_argument("--method", default="rise")
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--image-dir", default=None,
                    help="defaults to the task config's data_dir")
    ap.add_argument("--canon-size", type=int, default=CANON_SIZE)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ceiling-suffix", default=None,
                    help="suffix of same-architecture reruns, e.g. 'seed1'")
    args = ap.parse_args()

    store = load_maps(args.task, args.method)

    # image dir
    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        from .modelio import config_path, load_config
        any_model = next(iter(store))[0]
        cfg = load_config(config_path(args.task, any_model), overrides={"fold": 0})
        image_dir = Path(cfg.data_dir)
        if not image_dir.is_absolute():
            image_dir = PROJECT_ROOT / image_dir
    vessel_dir = Path(args.vessel_dir)
    if not vessel_dir.is_absolute():
        vessel_dir = PROJECT_ROOT / vessel_dir

    primary = {k: v for k, v in store.items() if k[2] is None}
    models = sorted({k[0] for k in primary})
    print(f"models: {models}")
    print(f"split-half r (median per model):")
    for m in models:
        rs = np.concatenate([v["shr"] for k, v in primary.items() if k[0] == m])
        rs = rs[np.isfinite(rs)]
        if rs.size:
            print(f"  {m:<20}{np.median(rs):.3f}")

    # ---------------- per-image geometry cache, keyed by rel path -------------
    geom_cache: dict[str, dict] = {}

    def geometry(rel: str):
        if rel in geom_cache:
            return geom_cache[rel]
        img = load_canonical(str(image_dir / rel), args.canon_size)
        fov = fov_mask(img)
        vp = vessel_dir / Path(rel).name
        if not vp.exists():
            geom_cache[rel] = None
            return None
        ves = load_vessel_channels(load_canonical_mask(str(vp), args.canon_size))
        disc = estimate_disc(img, fov, ves["vessel"])
        rois = build_rois(img, fov, ves, disc)
        geom_cache[rel] = dict(fov=fov, disc=disc, rois=rois)
        return geom_cache[rel]

    # ---------------- regional + radial --------------------------------------
    reg_rows, rad_rows = [], []
    for (model, fold, suffix), rec in store.items():
        tag = suffix or "primary"
        for i, rel in enumerate(rec["rel"]):
            g = geometry(str(rel))
            if g is None:
                continue
            sal = rec["maps"][i].astype(np.float32)
            fov, rois, disc = g["fov"], g["rois"], g["disc"]
            norm = normalise_map(sal, fov)

            sigma = map_smoothness_sigma(sal)
            null = random_smooth_map(*sal.shape, sigma=sigma, seed=i)
            null_norm = normalise_map(null, fov)

            for roi_name in ROI_ORDER:
                roi = rois[roi_name]
                reg_rows.append(dict(
                    task=args.task, method=args.method, run=tag, model=model,
                    family=MODEL_FAMILY.get(model, "?"), fold=fold,
                    sample_id=str(rec["sample_id"][i]), label=float(rec["label"][i]),
                    roi=roi_name,
                    area_frac=float((roi & fov).sum()) / float(fov.sum()),
                    enrichment=enrichment(norm, roi, fov),
                    mass=mass_fraction(norm, roi, fov),
                    pointing=pointing_game(sal, roi, fov),
                    iou10=soft_iou_topk(sal, roi, fov, 0.10),
                    null_enrichment=enrichment(null_norm, roi, fov),
                    null_iou10=soft_iou_topk(null, roi, fov, 0.10),
                ))

            prof = radial_profile(norm, fov, disc, n_bins=N_RADIAL_BINS)
            for b, v in enumerate(prof):
                rad_rows.append(dict(
                    task=args.task, method=args.method, run=tag, model=model,
                    family=MODEL_FAMILY.get(model, "?"), fold=fold,
                    sample_id=str(rec["sample_id"][i]), bin=b,
                    dd_lo=b * 4.0 / N_RADIAL_BINS, dd_hi=(b + 1) * 4.0 / N_RADIAL_BINS,
                    density=v,
                ))

    regional = pd.DataFrame(reg_rows)
    radial = pd.DataFrame(rad_rows)

    # ---------------- agreement ----------------------------------------------
    agr_rows = []
    for fold in sorted({k[1] for k in store}):
        recs = {k: v for k, v in store.items() if k[1] == fold}
        index = {k: {str(s): i for i, s in enumerate(v["sample_id"])} for k, v in recs.items()}

        ordered = sorted(recs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or ""))
        for (ka, va), (kb, vb) in itertools.combinations(ordered, 2):
            same_arch = ka[0] == kb[0]
            kind = "ceiling" if same_arch else "cross_model"
            if same_arch and ka[2] == kb[2]:
                continue
            shared = set(index[ka]) & set(index[kb])
            for sid in shared:
                ia, ib = index[ka][sid], index[kb][sid]
                rel = str(va["rel"][ia])
                g = geometry(rel)
                if g is None:
                    continue
                a = va["maps"][ia].astype(np.float32)
                b = vb["maps"][ib].astype(np.float32)
                fov, disc = g["fov"], g["disc"]
                agr_rows.append(dict(
                    task=args.task, method=args.method, kind=kind, fold=fold,
                    sample_id=sid,
                    model_a=f"{ka[0]}:{ka[2] or 'primary'}",
                    model_b=f"{kb[0]}:{kb[2] or 'primary'}",
                    arch_a=ka[0], arch_b=kb[0],
                    family_a=MODEL_FAMILY.get(ka[0], "?"),
                    family_b=MODEL_FAMILY.get(kb[0], "?"),
                    spearman=spearman_agreement(a, b, fov),
                    iou10=iou_agreement(a, b, fov, 0.10),
                    centroid_dd=centroid_shift(a, b, fov, disc.dd),
                ))
    agreement = pd.DataFrame(agr_rows)

    # ---------------- summary -------------------------------------------------
    sums = []
    prim = regional[regional["run"] == "primary"]
    for (model, roi), grp in prim.groupby(["model", "roi"]):
        m, lo, hi = bootstrap_ci(grp["enrichment"].values, args.n_boot)
        nm, nlo, nhi = bootstrap_ci(grp["null_enrichment"].values, args.n_boot)
        d, dlo, dhi, p = paired_bootstrap_diff(
            grp["enrichment"].values, grp["null_enrichment"].values, args.n_boot)
        sums.append(dict(
            task=args.task, method=args.method, model=model,
            family=MODEL_FAMILY.get(model, "?"), roi=roi, n=len(grp),
            enrichment=m, ci_lo=lo, ci_hi=hi,
            null_enrichment=nm, null_lo=nlo, null_hi=nhi,
            vs_null=d, vs_null_lo=dlo, vs_null_hi=dhi, p_raw=p,
            pointing=grp["pointing"].mean(), iou10=grp["iou10"].mean(),
        ))
    summary = pd.DataFrame(sums)
    if len(summary):
        summary["p_holm"] = holm_bonferroni(summary["p_raw"].values)

    out = PROJECT_ROOT / "outputs" / "xai" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.task}_{args.method}"
    regional.to_csv(out / f"regional_{tag}.csv", index=False)
    radial.to_csv(out / f"radial_{tag}.csv", index=False)
    agreement.to_csv(out / f"agreement_{tag}.csv", index=False)
    summary.to_csv(out / f"summary_{tag}.csv", index=False)

    # ---------------- console verdict ----------------------------------------
    print("\n" + "=" * 74)
    print("ENRICHMENT vs smoothness-matched null (Holm-corrected)")
    print("=" * 74)
    key = summary[summary["roi"].isin(["vessel", "zone_b", "disc", "macula", "periphery"])]
    for roi in ["vessel", "zone_b", "disc", "macula", "periphery"]:
        sub = key[key["roi"] == roi].sort_values("enrichment", ascending=False)
        if not len(sub):
            continue
        print(f"\n  {roi}")
        for _, r in sub.iterrows():
            sig = "*" if r["p_holm"] < 0.05 else " "
            print(f"    {r['model']:<20}{r['enrichment']:6.2f} "
                  f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]  null {r['null_enrichment']:5.2f}  "
                  f"p={r['p_holm']:.3g}{sig}")

    if len(agreement):
        print("\n" + "=" * 74)
        print("AGREEMENT -- the ceiling question")
        print("=" * 74)
        for kind, grp in agreement.groupby("kind"):
            m, lo, hi = bootstrap_ci(grp["spearman"].values, args.n_boot)
            mi, _, _ = bootstrap_ci(grp["iou10"].values, args.n_boot)
            mc, _, _ = bootstrap_ci(grp["centroid_dd"].values, args.n_boot)
            print(f"  {kind:<14} spearman {m:5.3f} [{lo:.3f},{hi:.3f}]   "
                  f"iou10 {mi:5.3f}   centroid {mc:4.2f} DD   n={len(grp)}")
        if "ceiling" in set(agreement["kind"]):
            c = agreement[agreement["kind"] == "ceiling"]["spearman"].mean()
            x = agreement[agreement["kind"] == "cross_model"]["spearman"].mean()
            print(f"\n  ceiling - cross_model = {c - x:+.3f}")
            print("  A small or negative gap means cross-architecture 'divergence' is")
            print("  indistinguishable from run-to-run noise. Do NOT write the")
            print("  divergence claim in that case.")
        else:
            print("\n  NO CEILING RUNS FOUND. Train 2-3 extra seeds for at least one ViT")
            print("  and one CNN, save as best_seed1.pt / best_seed2.pt, then rerun")
            print("  run_attribution.py --checkpoint-suffix seed1 (and seed2).")
            print("  Without this the agreement numbers are uninterpretable.")

    print(f"\nsaved 4 tables to {out}")


if __name__ == "__main__":
    main()
