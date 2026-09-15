"""
STEP 4 -- the causal test. This is what turns the study into a paper.

Enrichment says a model's attribution concentrates on vessels. It does not say
the model USES the vessels: a saliency method can point at a region the model
would happily do without. So mask each anatomical region (blend it to the
blurred baseline, exactly as in the attribution perturbations), re-run
inference on the whole held-out fold, and measure the AUROC drop.

    "RETFound shows 2.1x vessel enrichment and loses 0.06 AUROC when vessels
     are masked; EfficientNet-B2 shows 1.2x and loses 0.01"

is a publishable sentence. The enrichment number alone is not.

A model whose AUROC is untouched by masking the region its saliency highlights
is a model whose explanation does not describe its computation -- which is a
finding in its own right, and worth stating plainly if that is what you get.

Also runs the DEMOGRAPHIC CONFOUND check when --age-col / --sex-col are given.
Fundus networks read age and sex well; if these models predict CIMT largely
through demographic proxies, then 'where they look' is partly 'where they read
age from', and a reviewer will ask. Better to have the number.

Usage:
    python -m xai.run_ablation --task cimt --vessel-dir data/cimt/vessel_masks \
        --folds 0 1 2 3 4 --age-col age --sex-col sex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .anatomy import ABLATION_ROIS, build_rois, estimate_disc, load_vessel_channels
from .folddata import prepare_fold
from .frames import (CANON_SIZE, blurred_baseline, fov_mask, load_canonical,
                     load_canonical_mask)
from .metrics import bootstrap_auc_delta
from .modelio import MODEL_FAMILY, MODEL_KEYS, PROJECT_ROOT, CanonicalScorer, load_model, stratified_subsample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["cimt", "hypertension"])
    ap.add_argument("--models", nargs="*", default=list(MODEL_KEYS))
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--rois", nargs="*", default=ABLATION_ROIS)
    ap.add_argument("--n-images", type=int, default=0, help="0 = all held-out images")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--canon-size", type=int, default=CANON_SIZE)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--age-col", default=None)
    ap.add_argument("--sex-col", default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, conf_rows = [], []

    for fold in args.folds:
        geom_cache: dict[str, dict] = {}

        for key in args.models:
            try:
                lm = load_model(args.task, key, fold, device)
            except FileNotFoundError as e:
                print(f"[skip] {key} fold{fold}: {str(e)[:90]}")
                continue

            df = prepare_fold(lm.cfg, fold, args.vessel_dir)
            if args.n_images:
                df = stratified_subsample(df, "_label", args.n_images)
            scorer = CanonicalScorer(lm, device, amp=not args.no_amp)
            print(f"\n{key} fold{fold}  n={len(df)}")

            variants = ["none"] + list(args.rois)
            scores = {v: [] for v in variants}
            labels, kept = [], []

            for i in range(0, len(df), args.batch_size):
                chunk = df.iloc[i:i + args.batch_size]
                imgs, bases, roi_stack, ok = [], [], [], []

                for _, row in chunk.iterrows():
                    rel = row["_rel"]
                    if rel not in geom_cache:
                        img = load_canonical(row["_image_path"], args.canon_size)
                        vp = Path(row["_vessel_path"])
                        if not vp.exists():
                            geom_cache[rel] = None
                        else:
                            fov = fov_mask(img)
                            ves = load_vessel_channels(
                                load_canonical_mask(str(vp), args.canon_size))
                            disc = estimate_disc(img, fov, ves["vessel"])
                            geom_cache[rel] = dict(
                                fov=fov, rois=build_rois(img, fov, ves, disc))
                    g = geom_cache[rel]
                    if g is None:
                        ok.append(False)
                        continue
                    ok.append(True)
                    img = load_canonical(row["_image_path"], args.canon_size)
                    imgs.append(img.astype(np.float32))
                    bases.append(blurred_baseline(img).astype(np.float32))
                    roi_stack.append(g["rois"])
                    labels.append(float(row["_label"]))
                    kept.append(row["_sample_id"])

                if not imgs:
                    continue
                batch = np.stack(imgs)
                base = np.stack(bases)

                for v in variants:
                    if v == "none":
                        scores[v].append(scorer.score_batch(batch))
                    else:
                        pert = batch.copy()
                        for j, rois in enumerate(roi_stack):
                            m = rois[v][..., None].astype(np.float32)
                            pert[j] = batch[j] * (1 - m) + base[j] * m
                        scores[v].append(scorer.score_batch(pert))

            if not labels:
                continue
            y = np.array(labels)
            s = {v: np.concatenate(scores[v]) for v in variants}
            auc_base = roc_auc_score(y, s["none"])

            for v in args.rois:
                d, lo, hi = bootstrap_auc_delta(y, s["none"], s[v], args.n_boot)
                rows.append(dict(
                    task=args.task, model=key, family=MODEL_FAMILY.get(key, "?"),
                    fold=fold, roi=v, n=len(y),
                    auc_base=auc_base, auc_ablated=roc_auc_score(y, s[v]),
                    delta_auc=d, ci_lo=lo, ci_hi=hi,
                ))
                print(f"  mask {v:<12} AUROC {auc_base:.4f} -> "
                      f"{roc_auc_score(y, s[v]):.4f}   delta {d:+.4f} [{lo:+.4f},{hi:+.4f}]")

            # demographic confound: how much of this model's ranking is explained
            # by age/sex alone, and how well do the model's scores track them?
            if args.age_col or args.sex_col:
                sub = df[df["_sample_id"].isin(set(kept))].set_index("_sample_id").loc[kept]
                cols = [c for c in (args.age_col, args.sex_col) if c and c in sub.columns]
                if cols:
                    X = pd.get_dummies(sub[cols], drop_first=True).astype(float)
                    X = X.fillna(X.mean())
                    if X.notna().all().all() and len(np.unique(y)) > 1:
                        demo = LogisticRegression(max_iter=1000).fit(X, y)
                        auc_demo = roc_auc_score(y, demo.predict_proba(X)[:, 1])
                        r = float(np.corrcoef(s["none"], demo.predict_proba(X)[:, 1])[0, 1])
                        conf_rows.append(dict(
                            task=args.task, model=key, fold=fold, n=len(y),
                            auc_model=auc_base, auc_demographics_only=auc_demo,
                            corr_model_vs_demographics=r, columns=",".join(cols),
                        ))
                        print(f"  demographics-only AUROC {auc_demo:.4f} "
                              f"(in-sample); corr(model, demo) = {r:.3f}")

            del lm, scorer
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out = PROJECT_ROOT / "outputs" / "xai" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    ab = pd.DataFrame(rows)
    ab.to_csv(out / f"ablation_{args.task}.csv", index=False)
    if conf_rows:
        pd.DataFrame(conf_rows).to_csv(out / f"confound_{args.task}.csv", index=False)

    if len(ab):
        print("\n" + "=" * 74)
        print("AUROC DROP WHEN A REGION IS MASKED (pooled over folds)")
        print("=" * 74)
        piv = ab.groupby(["model", "roi"])["delta_auc"].mean().unstack()
        print(piv.round(4).to_string())
        print("\nPair each row with its enrichment from analyze.py. High enrichment +")
        print("near-zero AUROC drop = the explanation does not describe the model.")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
