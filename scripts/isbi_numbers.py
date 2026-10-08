#!/usr/bin/env python3
"""Every number from the ISBI grid, as flat tables.

    python scripts/isbi_numbers.py --root . --out numbers_out --boot 2000

Companion to scripts/isbi_stats.py (whose loaders and metric definitions it
reuses unchanged). Where isbi_stats.py writes the tables the paper argues from,
this writes EVERYTHING: one row per run with every metric, fold-mean summaries
with patient-bootstrap CIs for every metric, every paired comparison for every
metric, training noise, dataset counts and cost; and, when the run folders carry them (they do on IBEX), the
per-epoch training logs, SLURM job logs and resolved configs. CSV files only.

Bootstrap
---------
For each task x cohort, one set of patient-level resampling weights is drawn
per fold -- exactly the draws isbi_stats.py makes (same seed, same order) -- and
EVERY configuration is scored on those same draws. Every comparison is therefore
paired, and AUROC intervals reproduce isbi_stats.py exactly.

Thresholds are fixed per run (chosen on that run's inner-validation
predictions); the bootstrap resamples test patients only.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isbi_stats as S  # noqa: E402

COHORTS = {"hypertension": ["pooled", "brset", "odir"], "cimt": ["pooled"]}
BOOT_METRICS = ["auroc", "auprc", "ece", "brier", "sens_at_target", "spec_at_target"]
LOWER_IS_BETTER = {"ece", "brier"}


# ----------------------------------------------------------------------------
# point metrics
# ----------------------------------------------------------------------------
def ap_weighted(y: np.ndarray, s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Average precision (sklearn's definition: sum over distinct thresholds of
    delta-recall x precision) for each row of weights W (B, n)."""
    order = np.argsort(-s, kind="mergesort")
    ss, yy, Ws = s[order], y[order].astype(float), W[:, order]
    _, start = np.unique(-ss, return_index=True)          # tie groups, descending
    tp = np.cumsum(np.add.reduceat(Ws * yy, start, axis=1), axis=1)
    fp = np.cumsum(np.add.reduceat(Ws * (1.0 - yy), start, axis=1), axis=1)
    P = tp[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tp / (tp + fp)
        rec = tp / P
    drec = np.diff(np.concatenate([np.zeros((W.shape[0], 1)), rec], axis=1), axis=1)
    return np.nansum(drec * np.nan_to_num(prec), axis=1)


def ap(y, s) -> float:
    return float(ap_weighted(np.asarray(y), np.asarray(s, float), np.ones((1, len(y))))[0])


def sens_spec_weighted(y, p, t, W):
    pos, neg = (y == 1).astype(float), (y == 0).astype(float)
    hit = (p >= t).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ (pos * hit)) / (W @ pos), (W @ (neg * (1 - hit))) / (W @ neg)


def confusion(y, p, t) -> dict:
    y = np.asarray(y).astype(int); yhat = (np.asarray(p) >= t).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    ppv, npv = div(tp, tp + fp), div(tn, tn + fn)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, sens=sens, spec=spec, ppv=ppv, npv=npv,
                f1=div(2 * tp, 2 * tp + fp + fn), acc=div(tp + tn, len(y)),
                bal_acc=(sens + spec) / 2, flagged=div(tp + fp, len(y)))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_fit(y, p) -> tuple[float, float]:
    """Calibration intercept and slope: logistic regression of y on logit(p)
    (Cox). Slope 1 / intercept 0 = perfect; slope < 1 = over-confident."""
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7); y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(p), np.log(p / (1 - p))])
    b = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - mu); H = X.T @ (X * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-10:
            break
    return float(b[0]), float(b[1])


def mce(y, p, bins=10) -> float:
    """Maximum calibration error, same bins as the trainer's ECE."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = np.linspace(0, 1, bins + 1); out = 0.0
    for i in range(bins):
        m = (p > e[i]) & (p <= e[i + 1])
        if m.any():
            out = max(out, abs(y[m].mean() - p[m].mean()))
    return float(out)


def block(prefix: str, y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p, float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {}
    c0, c1 = calibration_fit(y, p)
    return {f"{prefix}n": len(y), f"{prefix}n_pos": int(y.sum()), f"{prefix}prevalence": float(y.mean()),
            f"{prefix}auroc": S.auc(y, p), f"{prefix}auprc": ap(y, p),
            f"{prefix}ece": S.ece(y, p), f"{prefix}mce": mce(y, p), f"{prefix}brier": S.brier(y, p),
            f"{prefix}log_loss": log_loss(y, p), f"{prefix}mean_pred": float(p.mean()),
            f"{prefix}cal_intercept": c0, f"{prefix}cal_slope": c1}


# ----------------------------------------------------------------------------
def per_run_table(runs: pd.DataFrame, target: float) -> pd.DataFrame:
    import json
    rows = []
    for r in runs.itertuples():
        m = json.loads((Path(r.val_csv).parents[1] / "run_manifest.json").read_text())
        fm, im = m.get("final_metrics") or {}, m.get("inner_val_metrics") or {}
        v, iv = S.preds(r.val_csv), S.preds(r.inner_csv)
        t_s, t_y = S.thresholds(iv, target)
        row = dict(arm=r.arm, config=r.label, task=r.task, fold=r.fold, seed=r.seed,
                   role="exact repeat" if r.replicate else ("fold" if r.seed == 42 else "extra seed"),
                   git_sha=r.git_sha, git_dirty=bool(r.git_dirty), input_size=r.input_size,
                   phase_of_best=m.get("phase_of_best"), epoch_of_best=r.epoch_of_best,
                   hours=r.hours, gpu=m.get("gpu"), n_train=m.get("n_train"),
                   n_inner_val=m.get("n_inner_val"), n_test=m.get("n_val"),
                   test_patients=v.patient_id.nunique(),
                   labels_csv_sha256=(m.get("labels_csv_sha256") or "")[:12],
                   splits_sha256=(m.get("splits_file_sha256") or "")[:12],
                   run_dir=r.run_dir)
        row.update(block("", v.y_true, v.y_prob))
        for name, t in (("t050", 0.5), ("tsens", t_s), ("tyouden", t_y)):
            row[f"{name}_threshold"] = t
            for k, val in confusion(v.y_true, v.y_prob, t).items():
                row[f"{name}_{k}"] = val
        if r.task == "hypertension":
            for c in ("brset", "odir"):
                sub = v[v.source_dataset == c]
                b = block(f"{c}_", sub.y_true, sub.y_prob)
                row.update({k: b[k] for k in b if not k.endswith(("mce", "log_loss", "cal_intercept"))})
                cs = confusion(sub.y_true, sub.y_prob, t_s)
                row[f"{c}_tsens_sens"], row[f"{c}_tsens_spec"] = cs["sens"], cs["spec"]
        ib = block("inner_", iv.y_true, iv.y_prob)
        row.update({k: ib[k] for k in ("inner_n", "inner_n_pos", "inner_prevalence",
                                       "inner_auroc", "inner_auprc", "inner_ece", "inner_brier")})
        # agreement with what the trainer itself recorded
        row["chk_auroc"] = abs(row["auroc"] - fm.get("auroc", np.nan))
        row["chk_auprc"] = abs(row["auprc"] - fm.get("auprc", np.nan))
        row["chk_ece"] = abs(row["ece"] - fm.get("ece", np.nan))
        row["chk_f1_at_0.5"] = abs(row["t050_f1"] - fm.get("f1", np.nan))
        row["chk_sens_at_0.5"] = abs(row["t050_sens"] - fm.get("sensitivity", np.nan))
        row["chk_spec_at_0.5"] = abs(row["t050_spec"] - fm.get("specificity", np.nan))
        row["chk_inner_auroc"] = abs(row["inner_auroc"] - im.get("auroc", np.nan))
        row["chk_inner_auprc"] = abs(row["inner_auprc"] - im.get("auprc", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_all(runs, per, B, seed):
    """{(task, cohort): {config: {metric: (point, dist)}}} on shared draws."""
    out = {}
    cfgs = (runs[~runs.replicate & (runs.seed == 42)][["arm", "label", "task"]]
            .drop_duplicates())
    for task, cohorts in COHORTS.items():
        tc = [(a, l) for a, l, t in cfgs.itertuples(index=False)
              if t == task and S.fold_runs(runs, a, l, task)]
        folds = {c: S.fold_runs(runs, *c, task) for c in tc}
        for coh in cohorts:
            rng = np.random.default_rng(seed)              # same draws as isbi_stats
            acc = {c: {m: [0.0, np.zeros(B)] for m in BOOT_METRICS} for c in tc}
            for f in range(5):
                ref = S.preds(folds[tc[0]][f].val_csv)
                keep = (np.ones(len(ref), bool) if coh == "pooled"
                        else (ref.source_dataset == coh).to_numpy())
                W = S.patient_weights(ref.patient_id.to_numpy()[keep], B, rng)
                Wd = W.astype(float)
                y = ref.y_true.to_numpy()[keep]
                for c in tc:
                    r = folds[c][f]
                    d = S.preds(r.val_csv)
                    if not d.sample_id.equals(ref.sample_id):
                        raise ValueError(f"{task} fold {f}: {c} evaluated on different images")
                    p = d.y_prob.to_numpy()[keep]
                    t = float(per.loc[(per.run_dir == r.run_dir), "tsens_threshold"].iloc[0])
                    se_p, sp_p = sens_spec_weighted(y, p, t, np.ones((1, len(y))))
                    se_w, sp_w = sens_spec_weighted(y, p, t, Wd)
                    vals = {"auroc": (S.auc(y, p), S.auc_weighted(y, p, W)),
                            "auprc": (ap(y, p), ap_weighted(y, p, Wd)),
                            "ece": (S.ece(y, p), S.ece_weighted(y, p, Wd)),
                            "brier": (S.brier(y, p), S.brier_weighted(y, p, Wd)),
                            "sens_at_target": (float(se_p[0]), se_w),
                            "spec_at_target": (float(sp_p[0]), sp_w)}
                    for m, (pt, dist) in vals.items():
                        acc[c][m][0] += pt / 5
                        acc[c][m][1] += dist / 5
            out[(task, coh)] = acc
            print(f"  bootstrap done: {task}/{coh} ({len(tc)} configs)", flush=True)
    return out


def comparison_pairs(task: str, labels: set) -> list[tuple[str, tuple, tuple]]:
    C = lambda l: ("core", l)  # noqa: E731
    core = sorted(l for a, l in labels if a == "core")
    pairs = [("all core pairs", C(a), C(b)) for a, b in itertools.combinations(core, 2)]
    pairs += [("retinal-vs-generalist (matched)", C("RETFound-DINOv2"), C("DINOv2-L")),
              ("retinal-vs-generalist (matched)", C("RETFound-MAE"), C("DINOv3-L"))]
    R = lambda n: ("resolution", n)  # noqa: E731
    pairs += [("resolution", C("DINOv2-L"), R("dinov2_large_res224")),
              ("resolution", C("DINOv2-L"), R("dinov2_large_res336")),
              ("resolution", R("dinov2_large_res336"), R("dinov2_large_res224")),
              ("resolution", R("convnextv2_res448"), C("ConvNeXtV2-B")),
              ("resolution (matched size)", R("convnextv2_res448"), C("DINOv2-L")),
              ("resolution (matched size)", R("dinov2_large_res224"), C("ConvNeXtV2-B")),
              ("resolution (matched size)", R("dinov2_large_res224"), C("DINOv3-L")),
              ("resolution (matched size)", R("dinov2_large_res224"), C("RETFound-MAE"))]
    pairs += [("weighted-BCE", ("wbce", "EfficientNet-B2 wBCE"), C("EfficientNet-B2"))]
    V = lambda n: ("vessel", n)  # noqa: E731
    pairs += [("vessel", V("efficientnet_vessel"), C("EfficientNet-B2")),
              ("vessel", V("efficientnet_vessel"), V("efficientnet_vessel_shuffled")),
              ("vessel", V("efficientnet_vessel_image"), V("efficientnet_vessel")),
              ("vessel", V("efficientnet_vessel_image"), C("EfficientNet-B2")),
              ("vessel", V("efficientnet_vessel_image"), V("efficientnet_vessel_shuffled")),
              ("vessel", V("efficientnet_vessel_shuffled"), C("EfficientNet-B2"))]
    return [p for p in pairs if p[1] in labels and p[2] in labels]


# ----------------------------------------------------------------------------
# training history, SLURM logs and resolved configs (present on IBEX; optional)
# ----------------------------------------------------------------------------
ID = ["arm", "config", "task", "fold", "seed", "role"]


def history_tables(root: Path, per: pd.DataFrame):
    """Per-epoch log for every run plus a per-run summary. 'val_*' in
    training_log.csv is the INNER-validation split (model selection)."""
    import json
    long, summ = [], []
    for r in per.itertuples():
        f = root / r.run_dir / "logs" / "training_log.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        m = json.loads((root / r.run_dir / "run_manifest.json").read_text())
        long.append(d.assign(**{k: getattr(r, k) for k in ID}, run_dir=r.run_dir))
        md = m.get("min_delta", 0.001) or 0.0
        best, be = -np.inf, None                       # trainer rule: > best + min_delta
        for e, a in zip(d.epoch, d.val_auroc):
            if a > best + md:
                best, be = a, e
        b = d[d.epoch == m["epoch_of_best"]].iloc[0]
        last = d.iloc[-1]
        imin = d.val_loss.idxmin()
        summ.append(dict(
            **{k: getattr(r, k) for k in ID}, input_size=r.input_size,
            epochs_run=len(d), epoch_of_best=int(m["epoch_of_best"]),
            epochs_after_best=int(last.epoch - m["epoch_of_best"]),
            patience=m.get("patience"), epoch_cap=m.get("phase2_epochs_cap"),
            stopped_by=("epoch cap" if len(d) >= (m.get("phase2_epochs_cap") or 10**9)
                        else "early stopping" if last.epoch - m["epoch_of_best"] >= (m.get("patience") or 0)
                        else "other"),
            inner_auroc_at_best=b.val_auroc, inner_auprc_at_best=b.val_auprc,
            inner_ece_at_best=b.val_ece, inner_loss_at_best=b.val_loss,
            train_loss_at_best=b.train_loss, lr_at_best=b.lr,
            inner_auroc_max=d.val_auroc.max(), epoch_of_max_inner_auroc=int(d.epoch[d.val_auroc.idxmax()]),
            inner_auroc_epoch0=d.val_auroc.iloc[0], inner_auroc_last=last.val_auroc,
            inner_loss_min=d.val_loss.min(), epoch_of_min_inner_loss=int(d.epoch[imin]),
            train_loss_first=d.train_loss.iloc[0], train_loss_last=last.train_loss,
            inner_loss_last=last.val_loss, loss_gap_last=last.val_loss - last.train_loss,
            lr_peak=d.lr.max(), lr_last=last.lr,
            epoch_sec_first=d.elapsed_sec.iloc[0], epoch_sec_median=d.elapsed_sec.median(),
            epoch_sec_total=d.elapsed_sec.sum(), wall_clock_sec=m.get("wall_clock_sec"),
            chk_epoch_of_best=int(be != m["epoch_of_best"]),
            chk_inner_auroc_vs_manifest=abs(b.val_auroc - (m.get("inner_val_metrics") or {}).get("auroc", np.nan))))
    if not summ:
        return None, None
    hl = pd.concat(long, ignore_index=True)
    hl = hl[ID + [c for c in hl.columns if c not in ID + ["run_dir"]] + ["run_dir"]]
    return hl, pd.DataFrame(summ)


def slurm_table(root: Path) -> pd.DataFrame | None:
    import re
    rows = []
    for f in sorted(root.glob("outputs_isbi*/_logs/*.out")):
        t = f.read_text(errors="replace")
        e = f.with_suffix(".err")
        et = e.read_text(errors="replace") if e.exists() else ""
        g = lambda pat: (re.search(pat, t, re.M).group(1).strip() if re.search(pat, t, re.M) else None)  # noqa: E731
        row = g(r"manifest row (\d+):")
        mrow = re.search(r"manifest row \d+:\s*(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)", t)
        err = re.findall(r"^(\w*(?:Error|Exception)[^\n]*)", et, re.M)
        params = g(r"Model:\s+\S+ \(([\d,]+) params\)")
        trainable = g(r"Trainable params: ([\d,]+)")
        done = re.search(r"^done: ", t, re.M) is not None
        rows.append(dict(
            log=f.relative_to(root).as_posix(), slurm_job=g(r"^ job (\d+)"), array_index=int(row) if row else None,
            task=mrow.group(1) if mrow else None, model=mrow.group(2) if mrow else None,
            fold=int(mrow.group(3)) if mrow else None, seed=int(mrow.group(4)) if mrow else None,
            out_tree=mrow.group(5) if mrow else None, config_file=mrow.group(6) if mrow else None,
            node=g(r"node (\S+)"), gpu=g(r"^ gpu\s+(.+)$"), git_sha=(g(r"^ git\s+(\S+)") or "")[:12],
            output=g(r"Output: (\S+)"), input_px=g(r"Input:\s+(\d+)px"),
            params=int(params.replace(",", "")) if params else None,
            trainable_params=int(trainable.replace(",", "")) if trainable else None,
            train_samples=g(r"Train:\s+(\d+) samples"), inner_samples=g(r"Inner val:\s+(\d+) samples"),
            test_samples=g(r"Val:\s+(\d+) samples"),
            minutes=float(g(r"Done in ([\d.]+) minutes")) if g(r"Done in ([\d.]+) minutes") else None,
            status="completed" if done else ("refused: run folder already had best.pt"
                                             if "already exists and no --checkpoint_path" in et
                                             else "failed"),
            error=err[-1][:200] if (err and not done) else None,
            warn_nondet_attention=int("Memory Efficient attention defaults to a non-deterministic" in et),
            warn_nondet_bicubic=int("upsample_bicubic2d_backward_out_cuda does not have a deterministic" in et)))
    return pd.DataFrame(rows) if rows else None


def config_table(root: Path, per: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import yaml
    except ImportError:
        return None
    vary = {"fold", "seed", "output_dir", "checkpoint_path"}
    out = []
    for (arm, cfg, task), g in per.groupby(["arm", "config", "task"]):
        cfgs = []
        for r in g.itertuples():
            f = root / r.run_dir / "config_resolved.yaml"
            if f.exists():
                c = yaml.safe_load(f.read_text()) or {}
                mk = c.pop("model_kwargs", {}) or {}
                c.update({f"model_kwargs.{k}": v for k, v in mk.items()})
                cfgs.append(c)
        if not cfgs:
            continue
        keys = sorted(set().union(*cfgs) - vary)
        differ = [k for k in keys if len({repr(c.get(k)) for c in cfgs}) > 1]
        row = dict(arm=arm, config=cfg, task=task, runs_with_config=len(cfgs),
                   fields_that_differ_between_runs=", ".join(differ) or "none")
        row.update({k: (repr(cfgs[0].get(k)) if isinstance(cfgs[0].get(k), (list, dict))
                        else cfgs[0].get(k)) for k in keys})
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap_.add_argument("--out", type=Path, default=Path("numbers_out"))
    ap_.add_argument("--boot", type=int, default=2000)
    ap_.add_argument("--seed", type=int, default=20261026)
    ap_.add_argument("--target-sens", type=float, default=0.80)
    a = ap_.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    runs = S.load_runs(a.root)
    print(f"{len(runs)} runs", flush=True)
    per = per_run_table(runs, a.target_sens)
    per.to_csv(a.out / "runs.csv", index=False)

    # ---- fold summaries (seed 42, folds 0-4) --------------------------------
    main_runs = per[per.role == "fold"]
    num = [c for c in per.columns if per[c].dtype.kind in "fi"
           and c not in ("fold", "seed") and not c.startswith("chk_")]
    g = main_runs.groupby(["arm", "config", "task"])
    summ = pd.concat({"mean": g[num].mean(), "sd": g[num].std(ddof=1),
                      "min": g[num].min(), "max": g[num].max()}, axis=1)
    summ.columns = [f"{c}__{s}" for s, c in summ.columns]
    summ = summ[[f"{c}__{s}" for c in num for s in ("mean", "sd", "min", "max")]]
    summ.insert(0, "n_folds", g.size())
    summ.reset_index().to_csv(a.out / "fold_summary.csv", index=False)

    # ---- bootstrap CIs for every config / cohort / metric -----------------
    boot = bootstrap_all(runs, per, a.boot, a.seed)
    rows = []
    for (task, coh), acc in boot.items():
        for (arm, label), ms in acc.items():
            for m, (pt, dist) in ms.items():
                lo, hi = S.ci(dist)
                rows.append(dict(task=task, cohort=coh, arm=arm, config=label, metric=m,
                                 value=pt, ci_lo=lo, ci_hi=hi, boot_se=float(np.nanstd(dist, ddof=1))))
    cis = pd.DataFrame(rows)
    cis.to_csv(a.out / "confidence_intervals.csv", index=False)

    # ---- every paired comparison --------------------------------------------
    rows = []
    for (task, coh), acc in boot.items():
        for fam, A, Bc in comparison_pairs(task, set(acc)):
            for m in BOOT_METRICS:
                d = acc[A][m][1] - acc[Bc][m][1]
                lo, hi = S.ci(d)
                rows.append(dict(family=fam, task=task, cohort=coh, metric=m, a=A[1], b=Bc[1],
                                 value_a=acc[A][m][0], value_b=acc[Bc][m][0],
                                 diff=acc[A][m][0] - acc[Bc][m][0], ci_lo=lo, ci_hi=hi,
                                 p=S.pval(d), better="lower" if m in LOWER_IS_BETTER else "higher"))
    comp = pd.DataFrame(rows)
    comp["p_holm"] = np.nan
    for _, idx in comp.groupby(["family", "task", "cohort", "metric"]).groups.items():
        comp.loc[idx, "p_holm"] = S.holm(comp.loc[idx, "p"].tolist())
    comp.to_csv(a.out / "comparisons_all.csv", index=False)

    # ---- training noise on fold 0 ------------------------------------------
    rows = []
    for (arm, cfg, task), f0 in per[per.fold == 0].groupby(["arm", "config", "task"]):
        r = dict(arm=arm, config=cfg, task=task)
        for m in ("auroc", "auprc", "ece", "brier"):
            for s in (42, 1337, 2024):
                x = f0[(f0.seed == s) & (f0.role != "exact repeat")][m]
                r[f"{m}_seed{s}"] = float(x.iloc[0]) if len(x) else np.nan
            rep = f0[f0.role == "exact repeat"][m]
            r[f"{m}_repeat42"] = float(rep.iloc[0]) if len(rep) else np.nan
            seeds = [r[f"{m}_seed{s}"] for s in (42, 1337, 2024)]
            r[f"{m}_sd_seed"] = float(np.std(seeds, ddof=1)) if not np.isnan(seeds).any() else np.nan
            r[f"{m}_range_seed"] = float(np.ptp(seeds)) if not np.isnan(seeds).any() else np.nan
            r[f"{m}_nondet"] = abs(r[f"{m}_repeat42"] - r[f"{m}_seed42"])
        rows.append(r)
    pd.DataFrame(rows).to_csv(a.out / "training_noise.csv", index=False)

    # ---- dataset counts (from the core EfficientNet-B2 fold runs) ----------
    rows = []
    ref = per[(per.arm == "core") & (per.config == "EfficientNet-B2") & (per.role == "fold")]
    for r in ref.sort_values(["task", "fold"]).itertuples():
        run = runs[runs.run_dir == r.run_dir].iloc[0]
        v, iv = S.preds(run.val_csv), S.preds(run.inner_csv)
        for split, d in (("test", v), ("inner_val", iv)):
            groups = [("all", d)] + ([(c, d[d.source_dataset == c]) for c in ("brset", "odir")]
                                    if r.task == "hypertension" else [])
            for coh, sub in groups:
                rows.append(dict(task=r.task, fold=r.fold, split=split, cohort=coh,
                                 images=len(sub), patients=sub.patient_id.nunique(),
                                 pos_images=int(sub.y_true.sum()),
                                 pos_patients=sub[sub.y_true == 1].patient_id.nunique(),
                                 prevalence_images=float(sub.y_true.mean()),
                                 images_per_patient=len(sub) / sub.patient_id.nunique()))
        rows.append(dict(task=r.task, fold=r.fold, split="train", cohort="all", images=r.n_train))
    pd.DataFrame(rows).to_csv(a.out / "dataset_counts.csv", index=False)

    # ---- cost ----------------------------------------------------------------
    cost = (per.groupby(["arm", "config", "task"])
            .agg(runs=("hours", "size"), hours_mean=("hours", "mean"), hours_min=("hours", "min"),
                 hours_max=("hours", "max"), gpu_hours_total=("hours", "sum"),
                 epoch_of_best_mean=("epoch_of_best", "mean"),
                 epoch_of_best_min=("epoch_of_best", "min"),
                 epoch_of_best_max=("epoch_of_best", "max"),
                 input_size=("input_size", "first"), gpu=("gpu", "first"))
            .reset_index())
    cost.to_csv(a.out / "cost.csv", index=False)

    hl, hs = history_tables(a.root, per)
    if hs is not None:
        hl.to_csv(a.out / "history_long.csv", index=False)
        hs.to_csv(a.out / "history_summary.csv", index=False)
        print(f"history: {len(hs)} runs, {len(hl)} epochs; epoch_of_best mismatches "
              f"{int(hs.chk_epoch_of_best.sum())}, max |inner AUROC - manifest| "
              f"{hs.chk_inner_auroc_vs_manifest.max():.1e}", flush=True)
    else:
        print("history: no logs/training_log.csv found -- skipped", flush=True)
    sl = slurm_table(a.root)
    if sl is not None:
        sl.to_csv(a.out / "slurm_jobs.csv", index=False)
        print(f"slurm: {len(sl)} job logs, " + ", ".join(f"{k}: {v}" for k, v in
              sl.status.value_counts().items()), flush=True)
    ct = config_table(a.root, per)
    if ct is not None and len(ct):
        ct.to_csv(a.out / "configs.csv", index=False)
        print(f"configs: {len(ct)} configurations; inconsistent within a configuration: "
              f"{int((ct.fields_that_differ_between_runs != 'none').sum())}", flush=True)

    chk = per.filter(like="chk_").max()
    print("max |recomputed - manifest|:\n" + chk.to_string(), flush=True)
    chk.rename("max_abs_diff").to_csv(a.out / "manifest_check.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
