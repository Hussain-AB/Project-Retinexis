"""
Turn the outputs of xai.run_maps into the tables and figures for the paper.

    python -m xai.summarize_maps --out-root outputs/xai_full --boot 1000

Reads every finished run under --out-root (a run counts once its DONE file
exists; unfinished runs are listed and skipped) and writes, to {out_root}/summary/:

  runs.csv               one row per run: layer used, fidelity (canonical vs
                         recorded logit), IG completeness
  enrichment.csv         per task x method x model x ROI x subgroup: mean
                         enrichment, mean null enrichment, difference with a
                         patient-level bootstrap CI and Holm-adjusted p
  model_contrasts.csv    per task x method x ROI: every pair of models,
                         paired on the same images
  method_agreement.csv   per task x model: Grad-CAM vs IG (pixel Spearman inside
                         the FOV, and whether both put the same ROIs above null)
  agreement.csv          fold 0: cross-model agreement vs the SAME model
                         retrained with another seed (the ceiling) and re-run
                         with the same seed (GPU non-determinism)
  radial.csv             mean radial profile per model (all / positives / negatives)
  site.csv               hypertension: periphery and disc enrichment, BRSET vs ODIR
  summary.md             the headline tables, plus a plain-language verdict
  figures/               mean maps per model, radial profiles, example overlays

Subgroups: all images, positives, negatives, and for hypertension BRSET / ODIR.
Main analysis = seed-42 runs in outputs_isbi (every held-out image once).
Extra-seed and exact-repeat runs (fold 0) are used ONLY for agreement.csv.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .paths import PROJECT_ROOT

KEY_ROIS = ["disc", "zone_b", "zone_c", "vessel", "artery", "vein", "macula", "background", "periphery"]
NAMES = {"retfound_dinov2": "RETFound-DINOv2", "retfound_mae": "RETFound-MAE", "dinov2_large": "DINOv2-L",
         "dinov3_large": "DINOv3-L", "convnextv2": "ConvNeXtV2-B", "efficientnet_b2": "EfficientNet-B2"}
ORDER = list(NAMES)


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def cluster_boot(values: np.ndarray, patients: np.ndarray, B: int, seed: int):
    """Mean with a patient-level bootstrap. Returns mean, lo, hi, two-sided p (vs 0)."""
    ok = np.isfinite(values)
    v, pid = values[ok], patients[ok]
    if v.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    codes, uniq = pd.factorize(pid)
    P = len(uniq)
    s = np.bincount(codes, weights=v, minlength=P)
    n = np.bincount(codes, minlength=P).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, P, size=(B, P)) + (np.arange(B) * P)[:, None]
    W = np.bincount(draws.ravel(), minlength=B * P).reshape(B, P).astype(float)
    bm = (W @ s) / (W @ n)
    p = 2 * min((bm <= 0).mean(), (bm >= 0).mean())
    return float(v.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)), float(max(p, 1 / B))


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    adj = np.empty(ok.sum())
    run = 0.0
    m = ok.sum()
    for r, i in enumerate(idx):
        run = max(run, (m - r) * p[ok][i])
        adj[i] = min(run, 1.0)
    out[ok] = adj
    return out


def mdt(df, index=True) -> str:
    """Markdown table; plain fallback when tabulate is not installed."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        d = df.reset_index() if index else df
        cols = [str(c) for c in d.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in d.itertuples(index=False):
            lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                           (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
        return "\n".join(lines)


def pos(method: str, m: np.ndarray) -> np.ndarray:
    """The map as the region metrics see it: Grad-CAM is already >= 0; for IG the
    positive part (evidence FOR the positive class), as in normalise_map."""
    m = m.astype(np.float32)
    return m if method == "gradcam" else np.clip(m, 0, None)


def pix_spearman(a: np.ndarray, b: np.ndarray, fov: np.ndarray) -> float:
    x, y = a[fov].astype(np.float64), b[fov].astype(np.float64)
    if x.size < 50 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


# ----------------------------------------------------------------------------
def load(out_root: Path):
    runs, regs, imgs, rads, maps = [], [], [], [], {}
    pending = []
    for info_f in sorted(out_root.glob("*/*/*/run_info.json")):
        d = info_f.parent
        info = json.loads(info_f.read_text())
        if not (d / "DONE").exists():
            pending.append(str(d.relative_to(out_root)))
            continue
        fid = json.loads((d / "DONE").read_text())
        role = ("random init" if info.get("random_init")
                else "exact repeat" if info["tree"].endswith("_rep")
                else "fold" if int(info["seed"]) == 42 else "extra seed")
        key = dict(task=info["task"], model=info["model"], fold=int(info["fold"]),
                   seed=int(info["seed"]), role=role)
        runs.append(dict(key, run_dir=info["run_dir"], input_size=info["input_size"],
                         layer=(info.get("gradcam_layer") or {}).get("name"),
                         layer_grid=(info.get("gradcam_layer") or {}).get("grid"),
                         n_images=info["n_images"], limit=info.get("limit", 0), **fid))
        for f in sorted(d.glob("chunk_*_images.csv")):
            imgs.append(pd.read_csv(f, dtype={"sample_id": str, "patient_id": str}).assign(**key))
        for f in sorted(d.glob("chunk_*_regions.csv")):
            x = pd.read_csv(f, dtype={"sample_id": str})
            if len(x):
                regs.append(x.assign(**key))
        for f in sorted(d.glob("chunk_*_radial.csv")):
            x = pd.read_csv(f, dtype={"sample_id": str})
            if len(x):
                rads.append(x.assign(**key))
        # several folders can belong to one run (e.g. Grad-CAM on all images,
        # IG on a subset); their map files are pooled and read per method
        maps.setdefault((info["task"], info["model"], int(info["fold"]), int(info["seed"]), role), []).extend(
            sorted(d.glob("chunk_*_maps.npz")))
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()  # noqa: E731
    return pd.DataFrame(runs), cat(regs), cat(imgs), cat(rads), maps, pending


def read_maps(files, method):
    ids, m, fov = [], [], []
    for f in files:
        z = np.load(f)
        if method not in z.files:
            continue
        ids += list(z["sample_id"]); m.append(z[method]); fov.append(z["fov"])
    if not m:
        return None
    return dict(ids=np.array(ids), maps=np.concatenate(m), fov=np.concatenate(fov))   # float16 in memory


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/xai_full")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--agreement-images", type=int, default=0,
                    help="cap images per fold-0 comparison (0 = all)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    root = Path(a.out_root)
    root = root if root.is_absolute() else PROJECT_ROOT / root
    out = root / "summary"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    runs, regs, imgs, rads, maps, pending = load(root)
    if runs.empty:
        raise SystemExit(f"no finished runs under {root}")
    print(f"{len(runs)} finished runs, {len(pending)} unfinished: {pending[:6]}")
    runs.to_csv(out / "runs.csv", index=False)
    md = ["# Grad-CAM + Integrated Gradients — summary\n",
          f"{len(runs)} finished runs; unfinished: {len(pending)}.\n"]

    # ---- fidelity ---------------------------------------------------------
    fid_rows = []
    for (task, model, fold, seed, role), g in imgs.drop_duplicates(
            ["task", "model", "fold", "seed", "role", "sample_id"]).groupby(
            ["task", "model", "fold", "seed", "role"]):
        y = g.label.to_numpy()
        from sklearn.metrics import roc_auc_score
        fid_rows.append(dict(task=task, model=model, fold=fold, seed=seed, role=role, n=len(g),
                             auroc_recorded=roc_auc_score(y, g.recorded_logit),
                             auroc_canonical=roc_auc_score(y, g.canonical_logit),
                             median_abs_logit_diff=float((g.canonical_logit - g.recorded_logit).abs().median())))
    fid = pd.DataFrame(fid_rows)
    fid["auroc_delta"] = fid.auroc_canonical - fid.auroc_recorded
    fid.to_csv(out / "fidelity.csv", index=False)
    md += ["## Fidelity — does the shared 512 px frame change what the models do?\n",
           mdt(fid[fid.role == "fold"].groupby(["task", "model"]).agg(
               runs=("fold", "size"), auroc_delta_mean=("auroc_delta", "mean"),
               auroc_delta_maxabs=("auroc_delta", lambda s: s.abs().max()),
               median_abs_logit_diff=("median_abs_logit_diff", "median")).round(4)), ""]

    ig = imgs[imgs.method == "ig"]
    if len(ig) and "ig_gap" in ig.columns:
        ig = ig[ig.ig_gap.abs() >= 0.5]          # ratio is unstable for tiny logit gaps
    if len(ig):
        comp = ig.groupby(["task", "model"]).ig_completeness.describe(percentiles=[.1, .5, .9])
        comp[["10%", "50%", "90%"]].round(3).to_csv(out / "ig_completeness.csv")
        md += ["## IG completeness, images with |logit gap| >= 0.5 (1.0 = enough integration steps)\n",
               mdt(comp[["10%", "50%", "90%"]].round(3)), ""]

    # ---- enrichment vs null ---------------------------------------------
    main_runs = regs[regs.role == "fold"] if len(regs) else regs
    if len(main_runs):
        meta = imgs.drop_duplicates(["task", "sample_id"])[["task", "sample_id", "patient_id",
                                                            "source_dataset", "label"]]
        R = main_runs.merge(meta, on=["task", "sample_id"], how="left")
        R["diff"] = R.enrichment - R.null_enrichment
        subgroups = {"all": lambda d: d, "positives": lambda d: d[d.label == 1],
                     "negatives": lambda d: d[d.label == 0],
                     "brset": lambda d: d[d.source_dataset == "brset"],
                     "odir": lambda d: d[d.source_dataset == "odir"]}
        rows = []
        for (task, method, model, roi), g in R.groupby(["task", "method", "model", "roi"]):
            for sg, f in subgroups.items():
                if task != "hypertension" and sg in ("brset", "odir"):
                    continue
                x = f(g)
                if not len(x):
                    continue
                m, lo, hi, p = cluster_boot(x["diff"].to_numpy(), x.patient_id.to_numpy(), a.boot, a.seed)
                rows.append(dict(task=task, method=method, model=model, roi=roi, subgroup=sg,
                                 images=len(x), area_frac=float(x.area_frac.mean()),
                                 enrichment=float(x.enrichment.mean()),
                                 null_enrichment=float(x.null_enrichment.mean()),
                                 diff=m, ci_lo=lo, ci_hi=hi, p=p,
                                 pointing=float(x.pointing.mean()), iou10=float(x.iou10.mean())))
        E = pd.DataFrame(rows)
        E["p_holm"] = np.nan
        for _, idx in E.groupby(["task", "method", "subgroup"]).groups.items():
            E.loc[idx, "p_holm"] = holm(E.loc[idx, "p"])
        E.to_csv(out / "enrichment.csv", index=False)
        for task in sorted(E.task.unique()):
            for method in sorted(E.method.unique()):
                t = E[(E.task == task) & (E.method == method) & (E.subgroup == "all") & E.roi.isin(KEY_ROIS)]
                if not len(t):
                    continue
                pv = t.assign(cell=[f"{e:.2f}{'*' if ph < 0.05 else ''}" for e, ph in zip(t.enrichment, t.p_holm)]) \
                    .pivot(index="model", columns="roi", values="cell").reindex(
                    [m for m in ORDER if m in set(t.model)])[[r for r in KEY_ROIS if r in set(t.roi)]]
                md += [f"## Enrichment — {task}, {method} (1.0 = uniform; * = differs from the "
                       "smoothness-matched null, Holm p<0.05)\n", mdt(pv), ""]

        # ---- trained vs random-init null (model randomisation) -----------------
        RI = regs[regs.role == "random init"] if len(regs) else regs
        if len(RI):
            tr = R[["task", "method", "model", "roi", "sample_id", "patient_id", "enrichment"]]
            ri = RI[["task", "method", "model", "roi", "sample_id", "enrichment"]].rename(
                columns={"enrichment": "enrichment_random"})
            J = tr.merge(ri, on=["task", "method", "model", "roi", "sample_id"])
            rrows = []
            for (task, method, model, roi), g in J.groupby(["task", "method", "model", "roi"]):
                d = (g.enrichment - g.enrichment_random).to_numpy()
                m, lo, hi, p = cluster_boot(d, g.patient_id.to_numpy(), a.boot, a.seed)
                rrows.append(dict(task=task, method=method, model=model, roi=roi, images=len(g),
                                  trained=float(g.enrichment.mean()), random_init=float(g.enrichment_random.mean()),
                                  diff=m, ci_lo=lo, ci_hi=hi, p=p))
            RR = pd.DataFrame(rrows)
            RR["p_holm"] = np.nan
            for _, idx in RR.groupby(["task", "method"]).groups.items():
                RR.loc[idx, "p_holm"] = holm(RR.loc[idx, "p"])
            RR.to_csv(out / "vs_random_init.csv", index=False)
            for (task, method), t in RR[RR.roi.isin(KEY_ROIS)].groupby(["task", "method"]):
                pv = t.assign(cell=[f"{tr_:.2f} vs {ri_:.2f}{'*' if ph < 0.05 else ''}"
                                    for tr_, ri_, ph in zip(t.trained, t.random_init, t.p_holm)]) \
                    .pivot(index="model", columns="roi", values="cell")
                md += [f"## Trained vs randomly initialised model — {task}, {method} "
                       "(enrichment, same images; * = Holm p<0.05). A region only counts as learned if "
                       "the trained model differs from its random-weight twin.\n", mdt(pv), ""]

        # ---- model contrasts (paired on the same images) ----------------------
        crow = []
        for (task, method, roi), g in R[R.roi.isin(KEY_ROIS)].groupby(["task", "method", "roi"]):
            wide = g.pivot_table(index=["sample_id", "patient_id"], columns="model", values="enrichment")
            for m1, m2 in itertools.combinations([m for m in ORDER if m in wide.columns], 2):
                d = (wide[m1] - wide[m2]).dropna()
                mm, lo, hi, p = cluster_boot(d.to_numpy(), d.index.get_level_values(1).to_numpy(), a.boot, a.seed)
                crow.append(dict(task=task, method=method, roi=roi, a=m1, b=m2, images=len(d),
                                 enrichment_a=float(wide[m1].mean()), enrichment_b=float(wide[m2].mean()),
                                 diff=mm, ci_lo=lo, ci_hi=hi, p=p))
        C = pd.DataFrame(crow)
        if len(C):
            C["p_holm"] = np.nan
            for _, idx in C.groupby(["task", "method", "roi"]).groups.items():
                C.loc[idx, "p_holm"] = holm(C.loc[idx, "p"])
            # does the other method agree on the direction of each contrast?
            other = C.set_index(["task", "roi", "a", "b", "method"])["diff"].unstack("method")
            if {"gradcam", "ig"} <= set(other.columns):
                agree = (np.sign(other.gradcam) == np.sign(other.ig)).rename("same_sign_both_methods")
                C = C.merge(agree.reset_index(), on=["task", "roi", "a", "b"], how="left")
            C.to_csv(out / "model_contrasts.csv", index=False)

        # ---- site: periphery / disc, BRSET vs ODIR -----------------------------
        srow = []
        H = R[(R.task == "hypertension") & R.roi.isin(["periphery", "disc", "background"])]
        for (method, model, roi), g in H.groupby(["method", "model", "roi"]):
            b, o = g[g.source_dataset == "brset"], g[g.source_dataset == "odir"]
            if not len(b) or not len(o):
                continue
            mb = cluster_boot(b.enrichment.to_numpy(), b.patient_id.to_numpy(), a.boot, a.seed)
            mo = cluster_boot(o.enrichment.to_numpy(), o.patient_id.to_numpy(), a.boot, a.seed + 1)
            srow.append(dict(method=method, model=model, roi=roi, brset=mb[0], brset_lo=mb[1], brset_hi=mb[2],
                             odir=mo[0], odir_lo=mo[1], odir_hi=mo[2], odir_minus_brset=mo[0] - mb[0],
                             separated=bool(mo[1] > mb[2] or mo[2] < mb[1])))
        S = pd.DataFrame(srow)
        if len(S):
            S.to_csv(out / "site.csv", index=False)
            md += ["## Hypertension: attribution by dataset (enrichment; 'separated' = non-overlapping CIs)\n",
                   mdt(S[["method", "model", "roi", "brset", "odir", "odir_minus_brset", "separated"]]
                       .round(3), index=False), ""]

    # ---- radial ------------------------------------------------------------
    if len(rads):
        rcols = [c for c in rads.columns if c.startswith("r") and c[1:].isdigit()]
        Rd = rads[rads.role == "fold"].merge(imgs.drop_duplicates(["task", "sample_id"])[["task", "sample_id", "label"]],
                                             on=["task", "sample_id"])
        rr = []
        for sg, f in {"all": lambda d: d, "positives": lambda d: d[d.label == 1],
                      "negatives": lambda d: d[d.label == 0]}.items():
            x = f(Rd).groupby(["task", "method", "model"])[rcols].mean().reset_index().assign(subgroup=sg)
            rr.append(x)
        radial = pd.concat(rr)
        radial.to_csv(out / "radial.csv", index=False)

    # ---- method agreement and seed ceiling (low-res maps) ------------------
    arow = []
    for (task, model, fold, seed, role), files in maps.items():
        if role != "fold":
            continue
        gc, igm = read_maps(files, "gradcam"), read_maps(files, "ig")
        if gc is None or igm is None:
            continue
        jx = {sid: j for j, sid in enumerate(igm["ids"])}       # align by image, not by position
        for i, sid in enumerate(gc["ids"]):
            if sid not in jx:
                continue
            arow.append(dict(task=task, model=model, fold=fold, sample_id=sid,
                             spearman=pix_spearman(pos("gradcam", gc["maps"][i]), pos("ig", igm["maps"][jx[sid]]),
                                                   gc["fov"][i])))
    MA = pd.DataFrame(arow)
    if len(MA):
        mrow = MA.groupby(["task", "model"]).spearman.describe(percentiles=[.25, .5, .75])
        if len(main_runs):
            e = E[E.subgroup == "all"].assign(above=lambda d: d.p_holm < 0.05)
            both = e.pivot_table(index=["task", "model", "roi"], columns="method", values="above", aggfunc="first")
            if {"gradcam", "ig"} <= set(both.columns):
                both = both.dropna()
                rois_agree = (both.gradcam == both.ig).groupby(["task", "model"]).mean().rename("roi_verdicts_agree")
                mrow = mrow.join(rois_agree)
        mrow.to_csv(out / "method_agreement.csv")
        md += ["## Grad-CAM vs IG on the same image (pixel Spearman inside the FOV; IG positive part)\n",
               mdt(mrow.round(3)), ""]

    agr = []
    for method in ("gradcam", "ig"):
        for task in sorted({k[0] for k in maps}):
            fold0 = {k: v for k, v in maps.items() if k[0] == task and k[2] == 0}
            loaded = {k: read_maps(v, method) for k, v in fold0.items()}
            loaded = {k: v for k, v in loaded.items() if v is not None}
            prim = {k[1]: v for k, v in loaded.items() if k[4] == "fold"}
            if not prim:
                continue
            ids = sorted(set.intersection(*[set(v["ids"]) for v in prim.values()]))
            if a.agreement_images:
                ids = ids[: a.agreement_images]
            idx = {m: {s: i for i, s in enumerate(v["ids"])} for m, v in prim.items()}
            for m1, m2 in itertools.combinations(sorted(prim), 2):
                for s in ids:
                    i, j = idx[m1][s], idx[m2][s]
                    agr.append(dict(task=task, method=method, kind="cross-model", a=m1, b=m2, sample_id=s,
                                    spearman=pix_spearman(pos(method, prim[m1]["maps"][i]),
                                                          pos(method, prim[m2]["maps"][j]), prim[m1]["fov"][i])))
            for k, v in loaded.items():
                if k[4] == "fold" or k[1] not in prim:
                    continue
                kind = {"extra seed": "same model, other seed",
                        "exact repeat": "same model, same seed (repeat)",
                        "random init": "same architecture, random weights"}[k[4]]
                iv = {s: i for i, s in enumerate(v["ids"])}
                for s in ids:
                    if s in iv:
                        agr.append(dict(task=task, method=method, kind=kind, a=k[1], b=f"{k[1]}@{k[3]}",
                                        sample_id=s, spearman=pix_spearman(pos(method, prim[k[1]]["maps"][idx[k[1]][s]]),
                                                                           pos(method, v["maps"][iv[s]]), v["fov"][iv[s]])))
    AG = pd.DataFrame(agr)
    if len(AG):
        AG.to_csv(out / "agreement_per_image.csv", index=False)
        ag = AG.groupby(["task", "method", "kind"]).spearman.agg(["count", "mean", "median"]).reset_index()
        ag.to_csv(out / "agreement.csv", index=False)
        md += ["## Agreement between maps, fold 0 (pixel Spearman)\n",
               "If 'cross-model' is about as high as 'same model, other seed', the models do NOT look at "
               "measurably different places -- differences between them are retraining noise.\n",
               mdt(ag.round(3), index=False), ""]

    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))

    if not a.no_figures:
        try:
            figures(root, out, imgs, maps, rads if len(rads) else None)
        except Exception as exc:  # noqa: BLE001
            print(f"[figures skipped: {exc}]")
    print(f"\nwritten: {out}")
    return 0


def figures(root, out, imgs, maps, rads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    disc = imgs.drop_duplicates(["task", "sample_id"]).set_index(["task", "sample_id"])
    for task in sorted({k[0] for k in maps}):
        models = [m for m in ORDER if any(k[0] == task and k[1] == m and k[4] == "fold" for k in maps)]
        fig, axs = plt.subplots(2, len(models), figsize=(2.2 * len(models), 4.6), squeeze=False)
        for col, model in enumerate(models):
            for row, method in enumerate(("gradcam", "ig")):
                acc, n = None, 0
                for k, files in maps.items():
                    if k[0] != task or k[1] != model or k[4] != "fold":
                        continue
                    m = read_maps(files, method)
                    if m is None:
                        continue
                    for i, s in enumerate(m["ids"]):
                        x = np.where(m["fov"][i], pos(method, m["maps"][i]), 0).astype(np.float64)
                        if x.sum() <= 0:
                            continue
                        x = x / x.sum()
                        dx = disc.loc[(task, s), "disc_x"] if (task, s) in disc.index else np.nan
                        if np.isfinite(dx) and dx < 256:      # flip so the disc is on the right
                            x = x[:, ::-1]
                        acc = x if acc is None else acc + x
                        n += 1
                ax = axs[row, col]
                if acc is not None:
                    ax.imshow(acc / n, cmap="magma")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(NAMES.get(model, model), fontsize=8)
                if col == 0:
                    ax.set_ylabel({"gradcam": "Grad-CAM", "ig": "IG (positive)"}[method], fontsize=8)
        fig.suptitle(f"{task}: mean attribution over all held-out images (disc flipped to the right)", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "figures" / f"mean_maps_{task}.png", dpi=200); plt.close(fig)

    if rads is not None and (out / "radial.csv").exists():
        rad = pd.read_csv(out / "radial.csv")
        rad = rad[rad.subgroup == "all"]
        rcols = [c for c in rad.columns if c.startswith("r") and c[1:].isdigit()]
        x = (np.arange(len(rcols)) + 0.5) * 4.0 / len(rcols)
        for task in sorted(rad.task.unique()):
            fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
            for ax, method in zip(axs, ("gradcam", "ig")):
                for model in ORDER:
                    r = rad[(rad.task == task) & (rad.method == method) & (rad.model == model)]
                    if len(r):
                        ax.plot(x, r[rcols].iloc[0].to_numpy(), label=NAMES[model], lw=1.4)
                ax.axhline(1, color="grey", ls="--", lw=0.8)
                ax.set_title(f"{task}: {method}", fontsize=9); ax.set_xlabel("distance from optic disc (disc diameters)")
            axs[0].set_ylabel("attribution density (1 = uniform)"); axs[0].legend(fontsize=7, frameon=False)
            fig.tight_layout(); fig.savefig(out / "figures" / f"radial_{task}.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
