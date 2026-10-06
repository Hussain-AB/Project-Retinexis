#!/usr/bin/env python3
"""Statistics for the ISBI grid, computed from the per-image prediction CSVs.

    python scripts/isbi_stats.py                          # repo root, all arms
    python scripts/isbi_stats.py --root <dir> --out stats_out --boot 2000

Reads every run_manifest.json + predictions/{val,inner_val}_predictions.csv
under outputs_isbi*/ and writes CSV tables plus summary.md to --out.

Method
------
* Uncertainty is a PATIENT-LEVEL cluster bootstrap. Each patient contributes
  ~2.4 images; resampling images would treat correlated eyes as independent
  and make every interval too narrow.
* A configuration's score is the MEAN of its five per-fold AUROCs (seed 42).
  In each bootstrap replicate, patients are resampled within every fold and the
  five fold AUROCs are recomputed and averaged.
* Comparisons are PAIRED: both configurations are scored on the same resampled
  patients in every replicate, fold by fold. The difference CI and a two-sided
  bootstrap p-value come from that paired distribution. p-values are reported
  raw and Holm-adjusted within each family of comparisons.
* These intervals capture patient-sampling uncertainty ONLY. Training
  randomness (seed, GPU non-determinism) is measured separately from the fold-0
  seed runs and the seed-42 replicate, and reported in its own table.
* Operating thresholds are chosen on INNER-VALIDATION predictions only -- the
  largest threshold reaching the target sensitivity there, and Youden's J --
  then applied unchanged to the held-out fold.
* ECE uses the trainer's own definition (10 equal-width bins, right-closed)
  and is cross-checked against the value each manifest recorded.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TREES = {
    "outputs_isbi": "core", "outputs_isbi_rep": "core",
    "outputs_isbi_vessel": "vessel", "outputs_isbi_vessel_rep": "vessel",
    "outputs_isbi_res": "resolution", "outputs_isbi_res_rep": "resolution",
    "outputs_isbi_wbce": "wbce",
}
CORE_NAMES = {
    "retfound_dinov2": "RETFound-DINOv2", "retfound_mae": "RETFound-MAE",
    "dinov2": "DINOv2-L", "dinov3": "DINOv3-L",
    "convnextv2": "ConvNeXtV2-B", "efficientnet": "EfficientNet-B2",
}
def md_table(df: pd.DataFrame) -> str:
    """Markdown table. Uses tabulate (via pandas) when installed, else a plain
    fallback, so the script needs nothing beyond numpy and pandas."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = [str(c) for c in df.columns]
        rows = [["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r]
                for r in df.itertuples(index=False)]
        return "\n".join(["| " + " | ".join(cols) + " |",
                          "|" + "|".join("---" for _ in cols) + "|"]
                         + ["| " + " | ".join(r) + " |" for r in rows])


TASK_NAMES = {"cimt": "cimt", "hypertension_combined": "hypertension"}


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for tree, arm in TREES.items():
        base = root / tree
        if not base.exists():
            continue
        for mf in sorted(base.rglob("run_manifest.json")):
            rel = mf.relative_to(root).parts
            if arm == "resolution":
                _, cfg, dataset, arch, _rundir, _ = rel
                label = cfg
            else:
                _, dataset, arch, _rundir, _ = rel
                label = (CORE_NAMES.get(arch, arch) if arm == "core"
                         else "EfficientNet-B2 wBCE" if arm == "wbce" else arch)
            m = json.loads(mf.read_text())
            pdir = mf.parent / "predictions"
            rows.append(dict(
                arm=arm, label=label, task=TASK_NAMES.get(dataset, dataset),
                arch=arch, tree=tree, replicate=tree.endswith("_rep"),
                fold=int(m["fold"]), seed=int(m["seed"]),
                input_size=m.get("input_size"), git_sha=(m.get("git_sha") or "")[:12],
                git_dirty=m.get("git_dirty"), wbce=m.get("use_weighted_bce"),
                epoch_of_best=m.get("epoch_of_best"),
                hours=(m.get("wall_clock_sec") or 0) / 3600,
                manifest_ece=(m.get("final_metrics") or {}).get("ece"),
                manifest_auroc=(m.get("final_metrics") or {}).get("auroc"),
                val_csv=str(pdir / "val_predictions.csv"),
                inner_csv=str(pdir / "inner_val_predictions.csv"),
                run_dir=str(mf.parent.relative_to(root)),
            ))
    return pd.DataFrame(rows)


_cache: dict[str, pd.DataFrame] = {}


def preds(path: str) -> pd.DataFrame:
    if path not in _cache:
        d = pd.read_csv(path, dtype={"sample_id": str, "patient_id": str})
        _cache[path] = d.sort_values("sample_id").reset_index(drop=True)
    return _cache[path]


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def auc_weighted(y: np.ndarray, s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AUROC for each row of weights W (B, n). Ties count one half.

    Sort once, then per replicate accumulate negative weight below each tie
    group -- O(n) per replicate, so 2,000 patient-bootstrap replicates of a
    4,205-image fold take well under a second.
    """
    order = np.argsort(s, kind="mergesort")
    ss, yy, Ws = s[order], y[order].astype(float), W[:, order]
    _, start = np.unique(ss, return_index=True)
    gp = np.add.reduceat(Ws * yy, start, axis=1)
    gn = np.add.reduceat(Ws * (1.0 - yy), start, axis=1)
    below = np.cumsum(gn, axis=1) - gn
    num = (gp * (below + 0.5 * gn)).sum(1)
    den = gp.sum(1) * gn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def auc(y, s) -> float:
    return float(auc_weighted(np.asarray(y), np.asarray(s, float),
                              np.ones((1, len(y))))[0])


def patient_weights(pids: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    codes, uniq = pd.factorize(pids)
    P = len(uniq)
    draws = rng.integers(0, P, size=(B, P))
    counts = np.bincount((draws + np.arange(B)[:, None] * P).ravel(),
                         minlength=B * P).reshape(B, P).astype(np.float32)
    return counts[:, codes]


def ece(y, p, bins: int = 10) -> float:
    """Equal-width ECE, defined exactly as src/evaluation/metrics.py does it:
    10 bins, right-closed intervals (lo, hi]. Reproducing the trainer's own
    definition means every recomputed value can be checked against the
    manifest rather than silently differing from it."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    tot, n = 0.0, len(y)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.any():
            tot += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(tot)


def ece_weighted(y: np.ndarray, p: np.ndarray, W: np.ndarray, bins: int = 10) -> np.ndarray:
    """ece() for each row of weights W (B, n). Uses the identity
    sum_b (n_b/n)|mean_y - mean_p| = sum_b |sum_y - sum_p| / n, so W = 1
    gives exactly ece()."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    tot = np.zeros(W.shape[0])
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.any():
            tot += np.abs(W[:, m] @ (y[m] - p[m]))
    return tot / W.sum(1)


def brier_weighted(y: np.ndarray, p: np.ndarray, W: np.ndarray) -> np.ndarray:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return (W @ (p - y) ** 2) / W.sum(1)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def thresholds(inner: pd.DataFrame, target_sens: float) -> tuple[float, float]:
    y, p = inner.y_true.to_numpy(), inner.y_prob.to_numpy()
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target_sens * len(pos))) - 1
    t_sens = float(pos[max(k, 0)]) if len(pos) else 0.5
    cand = np.unique(p)
    tp = np.array([(p[y == 1] >= t).mean() for t in cand])
    tn = np.array([(p[y == 0] < t).mean() for t in cand])
    t_youden = float(cand[np.argmax(tp + tn - 1)])
    return t_sens, t_youden


def sens_spec(df: pd.DataFrame, t: float) -> tuple[float, float]:
    y, p = df.y_true.to_numpy(), df.y_prob.to_numpy()
    return float((p[y == 1] >= t).mean()), float((p[y == 0] < t).mean())


# ----------------------------------------------------------------------------
# bootstrap engine
# ----------------------------------------------------------------------------
def fold_runs(runs: pd.DataFrame, arm: str, label: str, task: str) -> dict[int, pd.Series]:
    sel = runs[(runs.arm == arm) & (runs.label == label) & (runs.task == task)
               & (runs.seed == 42) & (~runs.replicate)]
    out = {int(r.fold): r for r in sel.itertuples()}
    return out if sorted(out) == [0, 1, 2, 3, 4] else {}


def boot_configs(runs, task, configs, cohort, B, seed):
    """Point and bootstrap mean-fold AUROC for several configs on SHARED draws."""
    rng = np.random.default_rng(seed)
    point = {c: [] for c in configs}
    dist = {c: np.zeros(B) for c in configs}
    folds = {c: fold_runs(runs, *c, task) for c in configs}
    for f in range(5):
        dfs = {c: preds(folds[c][f].val_csv) for c in configs}
        ref = dfs[configs[0]]
        for c, d in dfs.items():
            if not d.sample_id.equals(ref.sample_id):
                raise ValueError(f"fold {f}: {c} was evaluated on different images")
        keep = (np.ones(len(ref), bool) if cohort == "pooled"
                else (ref.source_dataset == cohort).to_numpy())
        W = patient_weights(ref.patient_id.to_numpy()[keep], B, rng)
        y = ref.y_true.to_numpy()[keep]
        for c, d in dfs.items():
            s = d.y_prob.to_numpy()[keep]
            point[c].append(auc(y, s))
            dist[c] += auc_weighted(y, s, W) / 5
    return {c: float(np.mean(point[c])) for c in configs}, dist


def ci(x) -> tuple[float, float]:
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def pval(d) -> float:
    d = d[~np.isnan(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(max(p, 1 / len(d)))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj.tolist()


def compare(runs, task, pairs, cohort, B, seed, family):
    configs = sorted({c for p in pairs for c in p})
    point, dist = boot_configs(runs, task, configs, cohort, B, seed)
    rows = []
    for a, b in pairs:
        d = dist[a] - dist[b]
        lo, hi = ci(d)
        rows.append(dict(family=family, task=task, cohort=cohort,
                         a=a[1], b=b[1], auroc_a=point[a], auroc_b=point[b],
                         diff=point[a] - point[b], ci_lo=lo, ci_hi=hi, p=pval(d)))
    for r, adj in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = adj
    return rows


def compare_calibration(runs, task, a_cfg, b_cfg, cohort, B, seed, family):
    """Paired patient-bootstrap difference in mean-fold ECE and Brier (A - B)."""
    rng = np.random.default_rng(seed)
    fa, fb = fold_runs(runs, *a_cfg, task), fold_runs(runs, *b_cfg, task)
    pt = {k: [] for k in ("ece_a", "ece_b", "brier_a", "brier_b")}
    dist = {k: np.zeros(B) for k in pt}
    for f in range(5):
        da, db = preds(fa[f].val_csv), preds(fb[f].val_csv)
        if not da.sample_id.equals(db.sample_id):
            raise ValueError(f"fold {f}: {a_cfg} and {b_cfg} were evaluated on different images")
        keep = (np.ones(len(da), bool) if cohort == "pooled"
                else (da.source_dataset == cohort).to_numpy())
        W = patient_weights(da.patient_id.to_numpy()[keep], B, rng).astype(float)
        y = da.y_true.to_numpy()[keep]
        for tag, d in (("a", da), ("b", db)):
            pr = d.y_prob.to_numpy()[keep]
            pt[f"ece_{tag}"].append(ece(y, pr)); pt[f"brier_{tag}"].append(brier(y, pr))
            dist[f"ece_{tag}"] += ece_weighted(y, pr, W) / 5
            dist[f"brier_{tag}"] += brier_weighted(y, pr, W) / 5
    rows = []
    for metric in ("ece", "brier"):
        d = dist[f"{metric}_a"] - dist[f"{metric}_b"]
        lo, hi = ci(d)
        va, vb = float(np.mean(pt[f"{metric}_a"])), float(np.mean(pt[f"{metric}_b"]))
        rows.append(dict(family=family, task=task, cohort=cohort, metric=metric,
                         a=a_cfg[1], b=b_cfg[1], value_a=va, value_b=vb, diff=va - vb,
                         ci_lo=lo, ci_hi=hi, p=pval(d)))
    return rows


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=Path("stats_out"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--target-sens", type=float, default=0.80)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    B, S = a.boot, a.seed

    runs = load_runs(a.root)
    md: list[str] = [f"# ISBI statistics\n\n{len(runs)} runs, {B} patient-bootstrap replicates, "
                     f"seed {S}.\n"]

    # ---- 1. provenance ----------------------------------------------------
    prov = (runs.groupby(["arm", "tree"])
            .agg(runs=("run_dir", "size"),
                 shas=("git_sha", lambda s: ",".join(sorted(set(s)))),
                 dirty=("git_dirty", lambda s: int(sum(bool(x) for x in s))))
            .reset_index())
    prov.to_csv(a.out / "provenance.csv", index=False)
    md += ["## Provenance\n", md_table(prov), ""]

    # identical evaluation images across every config of a task/fold
    bad = 0
    for (task, fold), g in runs[~runs.replicate].groupby(["task", "fold"]):
        ids = {tuple(preds(p).sample_id) for p in g.val_csv}
        bad += len(ids) > 1
    md.append(f"Evaluation sets identical across all configurations within every task/fold: "
              f"**{'yes' if bad == 0 else f'NO ({bad} mismatches)'}**\n")

    # ---- 2. per-run metrics (thresholds, calibration) ----------------------
    per = []
    for r in runs.itertuples():
        v, iv = preds(r.val_csv), preds(r.inner_csv)
        t_s, t_y = thresholds(iv, a.target_sens)
        se_s, sp_s = sens_spec(v, t_s)
        se_y, sp_y = sens_spec(v, t_y)
        row = dict(arm=r.arm, label=r.label, task=r.task, fold=r.fold, seed=r.seed,
                   replicate=r.replicate, input_size=r.input_size,
                   auroc=auc(v.y_true, v.y_prob), ece=ece(v.y_true, v.y_prob),
                   brier=brier(v.y_true, v.y_prob), manifest_ece=r.manifest_ece,
                   manifest_auroc=r.manifest_auroc,
                   thr_sens=t_s, sens_at_target=se_s, spec_at_target=sp_s,
                   thr_youden=t_y, sens_youden=se_y, spec_youden=sp_y,
                   epoch_of_best=r.epoch_of_best, hours=r.hours)
        if r.task == "hypertension":
            for c in ("brset", "odir"):
                sub = v[v.source_dataset == c]
                row[f"auroc_{c}"] = auc(sub.y_true, sub.y_prob)
        per.append(row)
    per = pd.DataFrame(per)
    per.to_csv(a.out / "per_run.csv", index=False)
    d_auc = (per.auroc - per.manifest_auroc).abs().max()
    d_ece = (per.ece - per.manifest_ece).abs().max()
    md.append(f"Recomputed vs manifest: max |AUROC diff| {d_auc:.2e}, max |ECE diff| {d_ece:.2e}\n")

    # ---- 3. headline table with CIs ----------------------------------------
    head = []
    configs = (runs[~runs.replicate & (runs.seed == 42)]
               [["arm", "label", "task"]].drop_duplicates())
    for (arm, label, task) in configs.itertuples(index=False):
        if not fold_runs(runs, arm, label, task):
            continue
        cohorts = ["pooled", "brset", "odir"] if task == "hypertension" else ["pooled"]
        row = dict(arm=arm, label=label, task=task)
        for c in cohorts:
            pt, dist = boot_configs(runs, task, [(arm, label)], c, B, S)
            lo, hi = ci(dist[(arm, label)])
            row[f"{c}"] = pt[(arm, label)]
            row[f"{c}_lo"], row[f"{c}_hi"] = lo, hi
        sub = per[(per.arm == arm) & (per.label == label) & (per.task == task)
                  & (per.seed == 42) & (~per.replicate)]
        row.update(ece=sub.ece.mean(), brier=sub.brier.mean(),
                   sens_at_target=sub.sens_at_target.mean(),
                   spec_at_target=sub.spec_at_target.mean(),
                   sens_youden=sub.sens_youden.mean(), spec_youden=sub.spec_youden.mean(),
                   input_size=int(sub.input_size.iloc[0]))
        # training-randomness from fold 0
        f0 = per[(per.arm == arm) & (per.label == label) & (per.task == task) & (per.fold == 0)]
        seeds = f0[~f0.replicate].sort_values("seed").auroc
        rep = f0[f0.replicate].auroc
        row["sd_seed"] = float(seeds.std(ddof=1)) if len(seeds) == 3 else np.nan
        row["nondet"] = (float(abs(rep.iloc[0] - f0[(~f0.replicate) & (f0.seed == 42)].auroc.iloc[0]))
                         if len(rep) else np.nan)
        row["sd_fold"] = float(sub.auroc.std(ddof=1))
        head.append(row)
    head = pd.DataFrame(head)
    head.to_csv(a.out / "headline.csv", index=False)

    def fmt(r, c):
        return f"{r[c]:.4f} [{r[c + '_lo']:.4f}, {r[c + '_hi']:.4f}]"

    for task in ("hypertension", "cimt"):
        t = head[head.task == task].sort_values("pooled", ascending=False)
        if t.empty:
            continue
        cols = ["pooled", "brset", "odir"] if task == "hypertension" else ["pooled"]
        tab = pd.DataFrame({
            "arm": t.arm, "config": t.label, "px": t.input_size,
            **{c: [fmt(r, c) for _, r in t.iterrows()] for c in cols},
            "sd_fold": t.sd_fold.round(4), "sd_seed": t.sd_seed.round(4),
            "nondet": t.nondet.round(4), "ECE": t.ece.round(4),
            f"sens@{a.target_sens:.2f}": t.sens_at_target.round(3),
            "spec": t.spec_at_target.round(3)})
        md += [f"## {task}: mean 5-fold AUROC [95% patient-bootstrap CI]\n",
               md_table(tab), ""]

    # ---- 4. comparison families --------------------------------------------
    C = lambda lbl, arm="core": (arm, lbl)  # noqa: E731
    fams = []
    for task in ("hypertension", "cimt"):
        cohorts = ["pooled", "brset"] if task == "hypertension" else ["pooled"]
        core = head[(head.arm == "core") & (head.task == task)].sort_values(
            "brset" if task == "hypertension" else "pooled", ascending=False)
        best = C(core.label.iloc[0])
        for coh in cohorts:
            fams += compare(runs, task, [(best, C(l)) for l in core.label.iloc[1:]],
                            coh, B, S, "best-vs-rest")
            fams += compare(runs, task, [(C("RETFound-DINOv2"), C("DINOv2-L")),
                                         (C("RETFound-MAE"), C("DINOv3-L"))],
                            coh, B, S, "retinal-vs-generalist (matched)")
            res = [(("resolution", "dinov2_large_res224"), C("DINOv2-L")),
                   (("resolution", "dinov2_large_res336"), C("DINOv2-L")),
                   (("resolution", "dinov2_large_res224"), ("resolution", "dinov2_large_res336")),
                   (("resolution", "convnextv2_res448"), C("ConvNeXtV2-B"))]
            if all(fold_runs(runs, *x, task) for p in res for x in p):
                fams += compare(runs, task, res, coh, B, S, "resolution")
            if fold_runs(runs, "wbce", "EfficientNet-B2 wBCE", task):
                fams += compare(runs, task, [(("wbce", "EfficientNet-B2 wBCE"),
                                              C("EfficientNet-B2"))], coh, B, S, "weighted-BCE")
        if task == "hypertension" and fold_runs(runs, "vessel", "efficientnet_vessel", task):
            V = lambda n: ("vessel", n)  # noqa: E731
            for coh in cohorts:
                fams += compare(runs, task, [
                    (V("efficientnet_vessel"), C("EfficientNet-B2")),
                    (V("efficientnet_vessel"), V("efficientnet_vessel_shuffled")),
                    (V("efficientnet_vessel_image"), V("efficientnet_vessel")),
                    (V("efficientnet_vessel_image"), C("EfficientNet-B2"))],
                    coh, B, S, "vessel")
    fams = pd.DataFrame(fams)
    fams.to_csv(a.out / "comparisons.csv", index=False)
    for (fam, task, coh), g in fams.groupby(["family", "task", "cohort"], sort=False):
        tab = pd.DataFrame({
            "A": g.a, "B": g.b, "A−B": g["diff"].map("{:+.4f}".format),
            "95% CI": [f"[{l:+.4f}, {h:+.4f}]" for l, h in zip(g.ci_lo, g.ci_hi)],
            "p": g.p.map("{:.4f}".format), "p (Holm)": g.p_holm.map("{:.4f}".format)})
        md += [f"## {fam} — {task}, {coh}\n", md_table(tab), ""]

    # ---- 4b. calibration: weighted-BCE vs plain BCE ------------------------
    cal = []
    for task in ("hypertension", "cimt"):
        if not fold_runs(runs, "wbce", "EfficientNet-B2 wBCE", task):
            continue
        for coh in (["pooled", "brset"] if task == "hypertension" else ["pooled"]):
            cal += compare_calibration(runs, task, ("wbce", "EfficientNet-B2 wBCE"),
                                       ("core", "EfficientNet-B2"), coh, B, S, "weighted-BCE")
    if cal:
        cal = pd.DataFrame(cal)
        cal.to_csv(a.out / "calibration.csv", index=False)
        tab = pd.DataFrame({
            "task": cal.task, "cohort": cal.cohort, "metric": cal.metric,
            "wBCE": cal.value_a.round(4), "plain BCE": cal.value_b.round(4),
            "A−B": cal["diff"].map("{:+.4f}".format),
            "95% CI": [f"[{l:+.4f}, {h:+.4f}]" for l, h in zip(cal.ci_lo, cal.ci_hi)],
            "p": cal.p.map("{:.4f}".format)})
        md += ["## Calibration — EfficientNet-B2 weighted-BCE minus plain BCE "
               "(mean 5-fold, paired patient bootstrap)\n", md_table(tab), ""]

    # ---- 5. site inflation by configuration --------------------------------
    hy = head[head.task == "hypertension"].copy()
    hy["inflation"] = hy.pooled - hy.brset
    hy[["arm", "label", "input_size", "pooled", "brset", "odir", "inflation"]] \
        .sort_values(["arm", "input_size"]).to_csv(a.out / "site_inflation.csv", index=False)
    md += ["## Pooled minus BRSET-only AUROC, hypertension\n",
           md_table(hy.sort_values(["arm", "input_size"])[
               ["arm", "label", "input_size", "pooled", "brset", "inflation"]].round(4)), ""]

    (a.out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
