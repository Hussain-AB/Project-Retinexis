#!/usr/bin/env python3
"""Audit of the BRSET hypertension label, and every hypertension model re-scored
on the images whose label can be trusted.

    python scripts/isbi_label_audit.py --root . --brset-labels <path>/labels_brset.csv \
        --out label_audit_out --boot 2000

CPU only, a few minutes. Reads predictions and the original BRSET labels file;
writes only to --out.

Why
---
The combined hypertension CSV sets a BRSET image to positive when the
'comorbidities' field mentions hypertension and to NEGATIVE when the field is
empty (scripts that built data/hypertension_combined). About half of BRSET has
no clinical record, so those images are negative by construction, not by
diagnosis. Almost all of BRSET's Nikon NF5050 images are in that group, so the
camera alone predicts the label. This script quantifies that, and re-scores
every hypertension configuration on:

  BRSET all                        what the main tables report
  BRSET, record present            only images whose label reflects a record
  BRSET, record present, Canon CR  the same with the camera held fixed

It also answers, on the record-present images, whether the image adds anything
beyond age and sex (logistic models fitted on the run's inner-validation
record-present rows, scored on its test fold), and how strongly each model's
score separates the two cameras among negatives (the shortcut).

The reconstructed label must equal y_true for every BRSET test image of every
run, otherwise the script stops: that check proves the labels file is the one
the labels were built from. BRSET sex coding is the documented one
(PhysioNet BRSET: 1 = male, 2 = female).

Bootstrap: patient-level, same seed and draw order as isbi_stats.py, shared by
every configuration within a subset, so all comparisons are paired.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isbi_stats as S  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fail(msg: str):
    print(f"\nERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_brset(path: Path) -> pd.DataFrame:
    b = pd.read_csv(path, low_memory=False)
    need = {"image_id", "patient_age", "patient_sex", "comorbidities", "camera", "quality"}
    if not need <= set(b.columns):
        fail(f"{path} lacks columns {sorted(need - set(b.columns))}; expected the original BRSET labels_brset.csv")
    b = b[b.quality == "Adequate"].copy()
    b["sample_id"] = "brset_" + b.image_id.astype(str)
    b["record"] = b.comorbidities.notna()
    b["label_rebuilt"] = (b.record & b.comorbidities.astype(str).str.lower().str.contains("hypertension")).astype(float)
    b["male"] = b.patient_sex.map({1: 1.0, 2: 0.0})
    return b.set_index("sample_id")


def design(d: pd.DataFrame, kind: str, mu: float, sd: float) -> np.ndarray:
    a = (d.patient_age.to_numpy(float) - mu) / sd
    s = d.male.to_numpy(float)
    cols = {"age+sex": [a, a ** 2, s], "image+age+sex": [d.y_logit.to_numpy(float), a, a ** 2, s]}[kind]
    return np.column_stack(cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--brset-labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("label_audit_out"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression

    meta = load_brset(a.brset_labels)
    md = [f"# BRSET hypertension label audit\n",
          f"Labels file: `{a.brset_labels}` (sha256 {sha256(a.brset_labels)[:12]}), "
          f"{len(meta)} Adequate-quality images, {meta.patient_id.nunique()} patients.\n"]

    # ---- 1. how the label is distributed ----------------------------------
    t = (meta.assign(age_missing=meta.patient_age.isna())
         .groupby(["camera", "record", "age_missing"]).label_rebuilt.agg(images="size", positives="sum", prevalence="mean")
         .reset_index())
    t.to_csv(a.out / "label_structure.csv", index=False)
    from sklearn.metrics import roc_auc_score
    y = meta.label_rebuilt.to_numpy()
    facts = dict(
        images=len(meta), no_record=int((~meta.record).sum()),
        no_record_share=float((~meta.record).mean()),
        positives_without_record=int(meta.label_rebuilt[~meta.record].sum()),
        age_missing=int(meta.patient_age.isna().sum()),
        age_missing_and_no_record=int((meta.patient_age.isna() & ~meta.record).sum()),
        prevalence_canon=float(meta.label_rebuilt[meta.camera == "Canon CR"].mean()),
        prevalence_nikon=float(meta.label_rebuilt[meta.camera == "NIKON NF5050"].mean()),
        nikon_without_record_share=float((~meta.record[meta.camera == "NIKON NF5050"]).mean()),
        auroc_camera_alone=float(roc_auc_score(y, (meta.camera == "Canon CR").astype(int))),
        auroc_age_missing_flag_alone=float(roc_auc_score(y, meta.patient_age.notna().astype(int))),
        prevalence_record_present=float(meta.label_rebuilt[meta.record].mean()),
    )
    pd.Series(facts).to_csv(a.out / "label_facts.csv", header=["value"])
    md += ["## How the label is made\n", S.md_table(pd.DataFrame([facts]).T.reset_index().rename(
        columns={"index": "fact", 0: "value"})), "", S.md_table(t.round(4)), ""]

    # ---- 2. every hypertension configuration on three subsets -------------
    runs = S.load_runs(a.root)
    cfgs = [(arm, lab) for arm, lab in runs[(runs.task == "hypertension") & (runs.seed == 42) & (~runs.replicate)]
            [["arm", "label"]].drop_duplicates().itertuples(index=False)
            if S.fold_runs(runs, arm, lab, "hypertension")]
    if not cfgs:
        fail("no hypertension runs found under --root")
    subsets = {
        "BRSET all": lambda d: d.source_dataset == "brset",
        "BRSET, record present": lambda d: (d.source_dataset == "brset") & d.record.eq(True),
        "BRSET, record present, Canon CR": lambda d: (d.source_dataset == "brset") & d.record.eq(True)
                                                     & (d.camera == "Canon CR"),
    }
    ci_rows, cmp_rows, cam_rows, rho_rows = [], [], [], []
    mismatches = 0
    for sname, sel in subsets.items():
        rng = np.random.default_rng(a.seed)
        acc = {}
        for f in range(5):
            folds = {c: S.fold_runs(runs, *c, "hypertension")[f] for c in cfgs}
            ref = None
            for c, r in folds.items():
                te = S.preds(r.val_csv).merge(meta[["record", "camera", "patient_age", "male", "label_rebuilt"]],
                                               left_on="sample_id", right_index=True, how="left")
                br = te.source_dataset == "brset"
                if te.loc[br, "label_rebuilt"].isna().any():
                    fail(f"{r.run_dir}: {int(te.loc[br, 'label_rebuilt'].isna().sum())} BRSET test images are not in "
                         "the labels file (wrong file, or a different quality filter)")
                mismatches += int((te.loc[br, "label_rebuilt"] != te.loc[br, "y_true"]).sum())
                keep = sel(te).to_numpy()
                if ref is None:
                    ref = te
                    W = S.patient_weights(te.patient_id.to_numpy()[keep], a.boot, rng)
                    yk = te.y_true.to_numpy()[keep]
                    agek = te.patient_age.to_numpy()[keep]
                    scores = {}
                    if np.isfinite(agek).all():
                        scores[("baseline", "age alone")] = agek
                if not te.sample_id.equals(ref.sample_id):
                    fail(f"fold {f}: {c} was scored on different images")
                scores[c] = te.y_prob.to_numpy()[keep]
                if sname == "BRSET all":
                    neg = te[br & (te.y_true == 0)]
                    cam_rows.append(dict(arm=c[0], config=c[1], fold=f,
                                         camera_auroc_among_negatives=S.auc((neg.camera == "Canon CR").astype(int), neg.y_prob),
                                         mean_risk_canon_neg=float(neg.y_prob[neg.camera == "Canon CR"].mean()),
                                         mean_risk_nikon_neg=float(neg.y_prob[neg.camera == "NIKON NF5050"].mean())))
                if sname == "BRSET, record present":
                    k = te[keep]
                    rho_rows.append(dict(arm=c[0], config=c[1], fold=f,
                                         spearman_score_age=spearmanr(k.y_logit, k.patient_age).statistic))
            for c, s in scores.items():
                e = acc.setdefault(c, [0.0, np.zeros(a.boot)])
                e[0] += S.auc(yk, s) / 5
                e[1] += S.auc_weighted(yk, s, W) / 5
            if f == 0:
                ci_rows.append(dict(subset=sname, arm="(info)", config="fold-0 images / prevalence",
                                    auroc=int(keep.sum()), ci_lo=float(yk.mean()), ci_hi=np.nan))
        for c, (pt, d) in acc.items():
            lo, hi = S.ci(d)
            ci_rows.append(dict(subset=sname, arm=c[0], config=c[1], auroc=pt, ci_lo=lo, ci_hi=hi))
        if ("baseline", "age alone") in acc:
            for c in cfgs:
                d = acc[c][1] - acc[("baseline", "age alone")][1]
                lo, hi = S.ci(d)
                cmp_rows.append(dict(subset=sname, comparison="image minus age alone", a=c[1], arm=c[0], b="age alone",
                                     diff=acc[c][0] - acc[("baseline", "age alone")][0], ci_lo=lo, ci_hi=hi, p=S.pval(d)))
        res_pairs = [(("resolution", "dinov2_large_res224"), ("core", "DINOv2-L")),
                     (("resolution", "dinov2_large_res336"), ("core", "DINOv2-L")),
                     (("resolution", "dinov2_large_res224"), ("resolution", "dinov2_large_res336")),
                     (("core", "ConvNeXtV2-B"), ("resolution", "convnextv2_res448"))]
        for lo_c, hi_c in res_pairs:
            if lo_c in acc and hi_c in acc:
                d = acc[hi_c][1] - acc[lo_c][1]
                lo, hi = S.ci(d)
                cmp_rows.append(dict(subset=sname, comparison="larger minus smaller input", a=hi_c[1], arm=hi_c[0],
                                     b=lo_c[1], diff=acc[hi_c][0] - acc[lo_c][0], ci_lo=lo, ci_hi=hi, p=S.pval(d)))
        print(f"  scored: {sname}", flush=True)
    if mismatches:
        fail(f"{mismatches} BRSET test labels differ from the label rebuilt from this file -- not the source file")

    # ---- 3. does the image add anything beyond age and sex? ---------------
    rec = meta[meta.record & meta.patient_age.notna() & meta.male.notna()]
    rng = np.random.default_rng(a.seed)
    acc = {}
    for f in range(5):
        ref = None
        for c in cfgs:
            r = S.fold_runs(runs, *c, "hypertension")[f]
            iv = S.preds(r.inner_csv).merge(rec[["patient_age", "male"]], left_on="sample_id", right_index=True)
            te = S.preds(r.val_csv).merge(rec[["patient_age", "male"]], left_on="sample_id", right_index=True) \
                .reset_index(drop=True)
            mu, sd = float(iv.patient_age.mean()), float(iv.patient_age.std())
            if ref is None:
                ref = te
                W = S.patient_weights(te.patient_id.to_numpy(), a.boot, rng)
                yk = te.y_true.to_numpy()
                kinds = ["age+sex", "image+age+sex"]
            if not te.sample_id.equals(ref.sample_id):
                fail(f"fold {f}: {c} record-present test rows differ")
            for kind in kinds if c == cfgs[0] else ["image+age+sex"]:
                lr = LogisticRegression(C=1e6, max_iter=5000).fit(design(iv, kind, mu, sd), iv.y_true.astype(int))
                s = lr.predict_proba(design(te, kind, mu, sd))[:, 1]
                key = ("baseline", "age+sex") if kind == "age+sex" else c
                e = acc.setdefault(key, [0.0, np.zeros(a.boot)])
                e[0] += S.auc(yk, s) / 5
                e[1] += S.auc_weighted(yk, s, W) / 5
    for c in cfgs:
        d = acc[c][1] - acc[("baseline", "age+sex")][1]
        lo, hi = S.ci(d)
        cmp_rows.append(dict(subset="BRSET, record present", comparison="image+age+sex minus age+sex",
                             a=c[1], arm=c[0], b="age+sex", diff=acc[c][0] - acc[("baseline", "age+sex")][0],
                             ci_lo=lo, ci_hi=hi, p=S.pval(d)))
        ci_rows.append(dict(subset="BRSET, record present (stacked on inner-val)", arm=c[0],
                            config=c[1] + " + age + sex", auroc=acc[c][0], ci_lo=float(np.percentile(acc[c][1], 2.5)),
                            ci_hi=float(np.percentile(acc[c][1], 97.5))))
    pt, d = acc[("baseline", "age+sex")]
    ci_rows.append(dict(subset="BRSET, record present (stacked on inner-val)", arm="baseline", config="age + sex",
                        auroc=pt, ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5))))
    for lo_c, hi_c in (( ("resolution", "dinov2_large_res224"), ("core", "DINOv2-L")),
                       (("core", "ConvNeXtV2-B"), ("resolution", "convnextv2_res448"))):
        if lo_c in acc and hi_c in acc:
            dd = acc[hi_c][1] - acc[lo_c][1]
            lo, hi = S.ci(dd)
            cmp_rows.append(dict(subset="BRSET, record present", comparison="age-adjusted: larger minus smaller input",
                                 a=hi_c[1], arm=hi_c[0], b=lo_c[1], diff=acc[hi_c][0] - acc[lo_c][0],
                                 ci_lo=lo, ci_hi=hi, p=S.pval(dd)))

    ci = pd.DataFrame(ci_rows)
    cmpd = pd.DataFrame(cmp_rows)
    cmpd["p_holm"] = np.nan
    for _, idx in cmpd.groupby(["subset", "comparison"]).groups.items():
        cmpd.loc[idx, "p_holm"] = S.holm(cmpd.loc[idx, "p"].tolist())
    cam = pd.DataFrame(cam_rows).groupby(["arm", "config"]).mean(numeric_only=True).drop(columns="fold").reset_index()
    rho = pd.DataFrame(rho_rows).groupby(["arm", "config"]).mean(numeric_only=True).drop(columns="fold").reset_index()
    ci.to_csv(a.out / "auroc_by_subset.csv", index=False)
    cmpd.to_csv(a.out / "comparisons.csv", index=False)
    cam.to_csv(a.out / "camera_shortcut.csv", index=False)
    rho.to_csv(a.out / "score_age_correlation.csv", index=False)

    for sname in list(subsets) + ["BRSET, record present (stacked on inner-val)"]:
        t = ci[ci.subset == sname].copy()
        t["AUROC [95% CI]"] = [f"{v:.4f} [{l:.4f}, {h:.4f}]" if r != "(info)" else f"n={int(v)}, prevalence {l:.3f}"
                               for v, l, h, r in zip(t.auroc, t.ci_lo, t.ci_hi, t.arm)]
        md += [f"## {sname}\n", S.md_table(t[["arm", "config", "AUROC [95% CI]"]]), ""]
    c2 = cmpd.assign(**{"A−B": cmpd["diff"].map("{:+.4f}".format),
                        "95% CI": [f"[{l:+.4f}, {h:+.4f}]" for l, h in zip(cmpd.ci_lo, cmpd.ci_hi)],
                        "p (Holm)": cmpd.p_holm.map("{:.4f}".format)})
    md += ["## Paired comparisons\n", S.md_table(c2[["subset", "comparison", "a", "b", "A−B", "95% CI", "p (Holm)"]]), ""]
    md += ["## Camera shortcut: BRSET negatives (mean over folds)\n", S.md_table(cam.round(4)), ""]
    md += ["## Spearman(score, age), BRSET record present\n", S.md_table(rho.round(3)), ""]
    (a.out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
