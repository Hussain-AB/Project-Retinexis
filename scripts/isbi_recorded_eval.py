#!/usr/bin/env python3
"""Score the "recorded" arm (models trained on BRSET images with a clinical
record only) against age alone and against the original models, on identical
test images.

    python scripts/isbi_recorded_eval.py --root . --brset-labels ~/labels_brset.csv \
        --audit-dir label_audit_out --out recorded_eval_out --boot 2000

CPU only, a few minutes. Reads predictions under outputs_isbi*/ and
outputs_isbi_rec/; writes only to --out.

Every original hypertension configuration is re-scored on the record-present
BRSET images exactly as scripts/isbi_label_audit.py does (same subsets, same
patient-bootstrap seed and draw order), so its numbers reproduce the audit's.
With --audit-dir the script checks that they do, to 1e-9, and stops otherwise.
The record-only models are then added on the same draws, so every comparison is
paired:

  record-only minus age alone                         does the image beat age once
                                                      the shortcut cannot be learned?
  record-only minus original (same architecture)      does removing the shortcut from
                                                      TRAINING change the model?
  image+age+sex minus age+sex (stacked on inner-val)  does the image add to age and sex?

Each record-only test fold must hold exactly the images the original models'
fold holds after restricting to BRSET with a record, with the same labels;
otherwise the script stops.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isbi_stats as S  # noqa: E402
from isbi_label_audit import design, fail, load_brset, sha256  # noqa: E402

REC_TREE = "outputs_isbi_rec"
# record-only arch folder -> (label, the original core config it is paired with)
REC = {"convnextv2": ("ConvNeXtV2-B record-only", ("core", "ConvNeXtV2-B")),
       "efficientnet": ("EfficientNet-B2 record-only", ("core", "EfficientNet-B2"))}
BEST = ("core", "DINOv2-L")


def rec_runs(root: Path) -> dict[tuple[str, str], dict[int, pd.Series]]:
    out = {}
    for arch, (label, _) in REC.items():
        folds = {}
        for mf in sorted((root / REC_TREE / "hypertension_combined" / arch).glob("fold_*/run_manifest.json")):
            m = json.loads(mf.read_text())
            if int(m["seed"]) != 42:
                continue
            pdir = mf.parent / "predictions"
            folds[int(m["fold"])] = pd.Series(dict(
                run_dir=str(mf.parent.relative_to(root)), val_csv=str(pdir / "val_predictions.csv"),
                inner_csv=str(pdir / "inner_val_predictions.csv"), git_sha=(m.get("git_sha") or "")[:12],
                epoch_of_best=m.get("epoch_of_best"), hours=(m.get("wall_clock_sec") or 0) / 3600))
        if sorted(folds) != [0, 1, 2, 3, 4]:
            fail(f"{REC_TREE}/hypertension_combined/{arch}: folds {sorted(folds)} finished, need 0-4")
        out[("recorded", label)] = folds
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--brset-labels", type=Path, required=True)
    ap.add_argument("--audit-dir", type=Path, default=None,
                    help="label_audit_out from isbi_label_audit.py, to check the original models reproduce")
    ap.add_argument("--out", type=Path, default=Path("recorded_eval_out"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression

    meta = load_brset(a.brset_labels)
    runs = S.load_runs(a.root)
    cfgs = [(arm, lab) for arm, lab in runs[(runs.task == "hypertension") & (runs.seed == 42) & (~runs.replicate)]
            [["arm", "label"]].drop_duplicates().itertuples(index=False)
            if S.fold_runs(runs, arm, lab, "hypertension")]
    for need in [p for _, p in REC.values()] + [BEST]:
        if need not in cfgs:
            fail(f"original configuration {need} not found with all 5 folds under --root")
    rec = rec_runs(a.root)
    md = [f"# Record-only retraining: BRSET hypertension\n",
          f"Labels file `{a.brset_labels}` (sha256 {sha256(a.brset_labels)[:12]}). "
          f"Record-only runs under `{REC_TREE}/`.\n"]
    run_rows = [dict(config=c[1], fold=f, **r.to_dict()) for c, fr in rec.items() for f, r in fr.items()]

    subsets = {
        "BRSET, record present": lambda d: (d.source_dataset == "brset") & d.record.eq(True),
        "BRSET, record present, Canon CR": lambda d: (d.source_dataset == "brset") & d.record.eq(True)
                                                     & (d.camera == "Canon CR"),
    }
    cols = ["record", "camera", "patient_age", "male", "label_rebuilt"]
    ci_rows, cmp_rows, rho_rows = [], [], []
    for sname, sel in subsets.items():
        rng = np.random.default_rng(a.seed)   # same seed and draw order as the audit
        acc = {}
        for f in range(5):
            scores = {}
            ref = None
            for c in cfgs:
                r = S.fold_runs(runs, *c, "hypertension")[f]
                te = S.preds(r.val_csv).merge(meta[cols], left_on="sample_id", right_index=True, how="left")
                keep = sel(te).to_numpy()
                if ref is None:
                    ref = te
                    k = te[keep].reset_index(drop=True)
                    W = S.patient_weights(k.patient_id.to_numpy(), a.boot, rng)
                    yk = k.y_true.to_numpy()
                    if not np.isfinite(k.patient_age.to_numpy(float)).all():
                        fail(f"{sname} fold {f}: missing age among record-present images")
                    scores[("baseline", "age alone")] = k.patient_age.to_numpy(float)
                if not te.sample_id.equals(ref.sample_id):
                    fail(f"fold {f}: {c} was scored on different images")
                scores[c] = te.y_prob.to_numpy()[keep]
            for c, fr in rec.items():
                rt = S.preds(fr[f].val_csv).merge(meta[cols], left_on="sample_id", right_index=True, how="left")
                if rt.source_dataset.ne("brset").any() or not rt.record.eq(True).all():
                    fail(f"{fr[f].run_dir}: test set holds images that are not BRSET with a record")
                if sname == "BRSET, record present" and not rt.sample_id.equals(k.sample_id):
                    fail(f"{fr[f].run_dir}: test images differ from the original models' record-present "
                         f"images of fold {f} ({len(rt)} vs {len(k)})")
                rt = rt[sel(rt).to_numpy()].reset_index(drop=True)
                if not rt.sample_id.equals(k.sample_id):
                    fail(f"{fr[f].run_dir}: {sname} images differ from the original models'")
                if not (rt.y_true.to_numpy() == yk).all() or not (rt.label_rebuilt.to_numpy() == yk).all():
                    fail(f"{fr[f].run_dir}: labels differ from the original models' / the rebuilt label")
                scores[c] = rt.y_prob.to_numpy()
                if sname == "BRSET, record present":
                    rho_rows.append(dict(config=c[1], fold=f,
                                         spearman_score_age=spearmanr(rt.y_logit, rt.patient_age).statistic))
            for c, s in scores.items():
                e = acc.setdefault(c, [0.0, np.zeros(a.boot)])
                e[0] += S.auc(yk, s) / 5
                e[1] += S.auc_weighted(yk, s, W) / 5
            if f == 0:
                ci_rows.append(dict(subset=sname, arm="(info)", config="fold-0 images / prevalence",
                                    auroc=len(yk), ci_lo=float(yk.mean()), ci_hi=np.nan))
        for c, (pt, d) in acc.items():
            lo, hi = S.ci(d)
            ci_rows.append(dict(subset=sname, arm=c[0], config=c[1], auroc=pt, ci_lo=lo, ci_hi=hi))

        def add(comp, A, B):
            d = acc[A][1] - acc[B][1]
            lo, hi = S.ci(d)
            cmp_rows.append(dict(subset=sname, comparison=comp, a=A[1], b=B[1], diff=acc[A][0] - acc[B][0],
                                 ci_lo=lo, ci_hi=hi, p=S.pval(d)))
        for label, orig in REC.values():
            c = ("recorded", label)
            add("record-only minus age alone", c, ("baseline", "age alone"))
            add("original minus age alone", orig, ("baseline", "age alone"))
            add("record-only minus original, same architecture", c, orig)
            add(f"record-only minus original {BEST[1]}", c, BEST)
        print(f"  scored: {sname}", flush=True)

    # ---- image+age+sex vs age+sex, logistic fitted on inner-val -------------
    rmeta = meta[meta.record & meta.patient_age.notna() & meta.male.notna()]
    rng = np.random.default_rng(a.seed)       # same as the audit's section 3
    acc = {}
    first = cfgs[0]
    for f in range(5):
        r0 = S.fold_runs(runs, *first, "hypertension")[f]
        te0 = S.preds(r0.val_csv).merge(rmeta[["patient_age", "male"]], left_on="sample_id",
                                        right_index=True).reset_index(drop=True)
        W = S.patient_weights(te0.patient_id.to_numpy(), a.boot, rng)
        yk = te0.y_true.to_numpy()
        todo = [(("baseline", "age + sex"), "age+sex", r0)] + [(c, "image+age+sex", S.fold_runs(runs, *c, "hypertension")[f])
                                                 for c in (BEST, *[p for _, p in REC.values()])] \
            + [(c, "image+age+sex", fr[f]) for c, fr in rec.items()]
        for key, kind, r in todo:
            iv = S.preds(r.inner_csv).merge(rmeta[["patient_age", "male"]], left_on="sample_id", right_index=True)
            te = S.preds(r.val_csv).merge(rmeta[["patient_age", "male"]], left_on="sample_id",
                                          right_index=True).reset_index(drop=True)
            if not te.sample_id.equals(te0.sample_id):
                fail(f"fold {f}: {key} record-present test rows differ")
            mu, sd = float(iv.patient_age.mean()), float(iv.patient_age.std())
            lr = LogisticRegression(C=1e6, max_iter=5000).fit(design(iv, kind, mu, sd), iv.y_true.astype(int))
            s = lr.predict_proba(design(te, kind, mu, sd))[:, 1]
            e = acc.setdefault(key, [0.0, np.zeros(a.boot)])
            e[0] += S.auc(yk, s) / 5
            e[1] += S.auc_weighted(yk, s, W) / 5
            if f == 0:
                run_rows.append(dict(config=f"{key[1]} [{kind}]", fold=0, inner_val_rows_used=len(iv)))
    sn = "BRSET, record present (stacked on inner-val)"
    for key, (pt, d) in acc.items():
        lo, hi = S.ci(d)
        ci_rows.append(dict(subset=sn, arm=key[0],
                            config=key[1] + ("" if key[0] == "baseline" else " + age + sex"),
                            auroc=pt, ci_lo=lo, ci_hi=hi))
        if key[0] != "baseline":
            dd = d - acc[("baseline", "age + sex")][1]
            lo, hi = S.ci(dd)
            cmp_rows.append(dict(subset=sn, comparison="image+age+sex minus age+sex", a=key[1], b="age+sex",
                                 diff=pt - acc[("baseline", "age + sex")][0], ci_lo=lo, ci_hi=hi, p=S.pval(dd)))

    ci = pd.DataFrame(ci_rows)
    cmpd = pd.DataFrame(cmp_rows)
    cmpd["p_holm"] = np.nan
    for _, idx in cmpd.groupby(["subset", "comparison"]).groups.items():
        cmpd.loc[idx, "p_holm"] = S.holm(cmpd.loc[idx, "p"].tolist())

    # ---- the original models must reproduce the label audit -----------------
    if a.audit_dir is not None:
        au = pd.read_csv(a.audit_dir / "auroc_by_subset.csv")
        m = ci.merge(au, on=["subset", "arm", "config"], suffixes=("", "_audit"))
        m = m[m.arm != "(info)"]
        if m.empty:
            fail("no rows in common with the audit's auroc_by_subset.csv")
        worst = float(np.nanmax(np.abs(m[["auroc", "ci_lo", "ci_hi"]].to_numpy()
                                       - m[["auroc_audit", "ci_lo_audit", "ci_hi_audit"]].to_numpy())))
        if worst > 1e-9:
            bad = m.assign(d=(m.auroc - m.auroc_audit).abs()).sort_values("d").tail(3)
            fail(f"original models do not reproduce the label audit (max |diff| {worst:.2e}):\n{bad}")
        md.append(f"Original models reproduce `{a.audit_dir}/auroc_by_subset.csv`: {len(m)} rows, "
                  f"max |difference| {worst:.1e}.\n")

    rho = pd.DataFrame(rho_rows).groupby("config").mean(numeric_only=True).drop(columns="fold").reset_index()
    ci.to_csv(a.out / "auroc.csv", index=False)
    cmpd.to_csv(a.out / "comparisons.csv", index=False)
    rho.to_csv(a.out / "score_age_correlation.csv", index=False)
    pd.DataFrame(run_rows).to_csv(a.out / "runs.csv", index=False)

    keep_cfg = {BEST[1], *[p[1] for _, p in REC.values()]}
    for sname in list(subsets) + [sn]:
        t = ci[(ci.subset == sname) & (ci.arm.isin(["(info)", "baseline", "recorded"])
                                        | ci.config.str.replace(" + age + sex", "", regex=False).isin(keep_cfg))]
        t = t.assign(**{"AUROC [95% CI]": [f"{v:.4f} [{l:.4f}, {h:.4f}]" if r != "(info)"
                                           else f"n={int(v)}, prevalence {l:.3f}"
                                           for v, l, h, r in zip(t.auroc, t.ci_lo, t.ci_hi, t.arm)]})
        md += [f"## {sname}\n", S.md_table(t[["arm", "config", "AUROC [95% CI]"]]), ""]
    c2 = cmpd.assign(**{"A−B": cmpd["diff"].map("{:+.4f}".format),
                        "95% CI": [f"[{l:+.4f}, {h:+.4f}]" for l, h in zip(cmpd.ci_lo, cmpd.ci_hi)],
                        "p (Holm)": cmpd.p_holm.map("{:.4f}".format)})
    md += ["## Paired comparisons (Holm within subset x comparison)\n",
           S.md_table(c2[["subset", "comparison", "a", "b", "A−B", "95% CI", "p (Holm)"]]), ""]
    md += ["## Spearman(score, age), BRSET record present, record-only models\n", S.md_table(rho.round(3)), ""]
    md += ["All original configurations are in auroc.csv; the tables above show the paired ones.\n"]
    (a.out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
