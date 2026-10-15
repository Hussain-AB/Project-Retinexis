#!/usr/bin/env python3
"""Age / sex baselines on exactly the folds and patients the image models used.

    python scripts/isbi_age_baseline.py --root . --out baseline_out --boot 2000

CPU only, a few minutes. Reads nothing that training wrote except predictions,
manifests and resolved configs; writes only to --out. The labels CSV is read,
never modified (the gender/sex column is used as-is).

What it answers
---------------
1. How well do age and sex alone predict the label?            (age, sex, age+sex)
2. How much of the pooled hypertension score does dataset
   identity alone explain?                                     (dataset, age+sex+dataset)
3. Does the image add anything beyond age and sex?             (image+age+sex, per image model)
4. Is each image model's score just an age estimate?           (Spearman with age; AUROC within age bands)

Folds
-----
For each task and fold, the reference is the core EfficientNet-B2 seed-42 run:
  test      = the sample_ids in its predictions/val_predictions.csv
  inner-val = the sample_ids in its predictions/inner_val_predictions.csv
  train     = every other row of the labels CSV the run recorded in its manifest
The train count is checked against the manifest's n_train, the CSV's sha256
against the manifest, and every label against the prediction files' y_true.
Any mismatch stops the script.

Models (logistic regression, effectively unpenalised, fitted on TRAIN only)
  age            age (standardised), age^2, missing-age indicator
  sex            sex indicator, missing-sex indicator
  age+sex        age, age^2, sex, age x sex, missing indicators
  dataset        is-ODIR indicator                          (hypertension only)
  age+sex+data   age+sex features + is-ODIR                  (hypertension only)
  image+age+sex  the image model's logit + age+sex features, fitted on that
                 run's INNER-VAL predictions (the image model never trained on
                 them), evaluated on the test fold. One per core image model.

Thresholds (80% sensitivity) are chosen on inner-val, as for the image models.
Bootstrap: the same patient draws isbi_stats.py / isbi_numbers.py use (same
seed, same order), so every baseline-vs-image comparison is paired with the
image models' own intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isbi_stats as S  # noqa: E402
import isbi_numbers as N  # noqa: E402

AGE_CANDIDATES = ["patient_age", "age", "Age", "Patient Age", "age_years", "AGE"]
SEX_CANDIDATES = ["patient_sex", "sex", "gender", "Sex", "Gender", "Patient Sex", "SEX", "GENDER"]
TASK_DATASET = {"hypertension": "hypertension_combined", "cimt": "cimt"}
COHORTS = {"hypertension": ["pooled", "brset", "odir"], "cimt": ["pooled"]}


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}\nNothing was written beyond what is already in --out.", file=sys.stderr)
    raise SystemExit(2)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pick(cols, override, candidates, what):
    if override:
        if override not in cols:
            fail(f"--{what}-col {override!r} is not a column of the labels CSV. Columns: {list(cols)}")
        return override
    hits = [c for c in candidates if c in cols]
    if not hits:
        fail(f"no {what} column found (tried {candidates}). Columns are: {list(cols)}. "
             f"Pass --{what}-col <name>.")
    return hits[0]


def read_cfg(run_dir: Path) -> dict:
    f = run_dir / "config_resolved.yaml"
    if not f.exists():
        return {}
    import yaml
    return yaml.safe_load(f.read_text()) or {}


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
MALE = {"m", "male", "man", "masculino", "homem", "h"}
FEMALE = {"f", "female", "woman", "feminino", "mulher"}


def parse_codes(spec: str | None) -> dict[str, dict[str, str]]:
    """'brset:1=male,2=female' (several datasets separated by ';') ->
    {'brset': {'1': 'male', '2': 'female'}}. Values must be male/female."""
    out: dict[str, dict[str, str]] = {}
    for part in filter(None, (spec or "").split(";")):
        ds, _, pairs = part.partition(":")
        if not pairs:
            fail(f"bad --*-sex-codes entry {part!r}; expected e.g. brset:1=male,2=female")
        m = {}
        for kv in pairs.split(","):
            k, _, v = kv.partition("=")
            v = v.strip().lower()
            if v not in ("male", "female"):
                fail(f"bad --*-sex-codes value {kv!r}; the right-hand side must be male or female")
            m[k.strip().lower()] = v
        if sorted(m.values()) != ["female", "male"]:
            fail(f"--*-sex-codes for {ds!r} must name one male and one female code: {m}")
        out[ds.strip().lower()] = m
    return out


def encode_sex(s: pd.Series, source: pd.Series, codes: dict | None = None) -> tuple[pd.Series, dict]:
    """1 = male, 0 = female when the column uses words. When it uses numeric
    codes, the codes must be the same two values in every source dataset (then
    the direction is irrelevant to a logistic model and the sorted order is used).
    Mixed codings across datasets would silently turn sex into a dataset
    indicator, so they stop the script with the value counts per dataset."""
    raw = s.astype("string").str.strip().str.lower()
    raw = raw.mask(raw.isin(["", "nan", "none", "na", "n/a", "unknown", "<na>"]))
    raw = raw.str.replace(r"\.0$", "", regex=True)
    by_src = {str(k): v.value_counts().to_dict() for k, v in raw.groupby(source.astype(str))}
    for ds, m in (codes or {}).items():        # documented numeric codes -> words, per dataset
        rows = source.astype(str).str.lower() == ds
        if not rows.any():
            fail(f"--*-sex-codes names dataset {ds!r}, which has no rows (datasets: {sorted(by_src)})")
        unknown = set(raw[rows].dropna()) - set(m)
        if unknown:
            fail(f"--*-sex-codes for {ds!r} does not cover the values {sorted(unknown)}")
        raw = raw.where(~rows, raw.map(lambda v: m.get(v, v) if isinstance(v, str) else v))
    words = raw.isin(MALE | FEMALE)
    if words.all() or (words | raw.isna()).all():
        out = raw.map(lambda v: 1.0 if v in MALE else (0.0 if v in FEMALE else np.nan))
        return out.astype(float), {"coding": "words (1 = male, 0 = female)"
                                   + (f"; numeric codes translated per dataset: {codes}" if codes else ""),
                                   "by_dataset": by_src}
    vals = sorted(raw.dropna().unique().tolist())
    per = {k: sorted(v.dropna().unique().tolist()) for k, v in raw.groupby(source.astype(str))}
    if len(vals) == 2 and not words.any() and all(v == vals for v in per.values()):
        out = raw.map({vals[0]: 0.0, vals[1]: 1.0})
        return out.astype(float), {"coding": f"numeric codes {vals[0]} -> 0, {vals[1]} -> 1 "
                                             "(same two codes in every dataset)", "by_dataset": by_src}
    fail("the sex column is coded inconsistently, so it cannot be used without a decision. "
         f"Value counts per dataset: {by_src}. Pass --<task>-sex-col to choose another column, "
         "or state the coding, e.g. --hypertension-sex-codes 'brset:1=male,2=female'.")


def design(df: pd.DataFrame, kind: str, ref: dict, image_logit=None) -> np.ndarray:
    """Feature matrix. `ref` holds train-set age mean/sd/median so every split is
    transformed with train statistics only."""
    age = df["_age"].to_numpy(float)
    miss_a = np.isnan(age).astype(float)
    a = np.where(np.isnan(age), ref["age_median"], age)
    a = (a - ref["age_mean"]) / ref["age_sd"]
    sex = df["_sex"].to_numpy(float)
    miss_s = np.isnan(sex).astype(float)
    sx = np.nan_to_num(sex, nan=0.0)
    odir = (df["source_dataset"].astype(str).str.lower() == "odir").to_numpy(float)
    cols = {
        "age": [a, a ** 2, miss_a],
        "sex": [sx, miss_s],
        "age+sex": [a, a ** 2, sx, a * sx, miss_a, miss_s],
        "dataset": [odir],
        "age+sex+dataset": [a, a ** 2, sx, a * sx, miss_a, miss_s, odir],
        "image+age+sex": [image_logit, a, a ** 2, sx, a * sx, miss_a, miss_s] if image_logit is not None else None,
    }[kind]
    X = np.column_stack(cols)
    keep = X.std(0) > 0                      # drop constant columns (e.g. no missing values)
    return X[:, keep] if keep.any() else X


def fit_predict(Xtr, ytr, Xs):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)
    m.fit(Xtr, ytr)
    return [m.predict_proba(X)[:, 1] for X in Xs]


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=Path("baseline_out"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--target-sens", type=float, default=0.80)
    ap.add_argument("--tasks", nargs="+", default=["hypertension", "cimt"])
    for t in ("hypertension", "cimt"):
        ap.add_argument(f"--{t}-age-col"); ap.add_argument(f"--{t}-sex-col")
        ap.add_argument(f"--{t}-sex-codes", help="numeric sex codes per dataset, e.g. 'brset:1=male,2=female'")
    ap.add_argument("--allow-sha-mismatch", action="store_true",
                    help="testing only: do not stop when the labels CSV differs from the manifest's")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = a.root.resolve()

    runs = S.load_runs(root)
    if runs.empty:
        fail(f"no runs found under {root}")
    info, fold_rows, pred_store = {}, [], {}

    for task in a.tasks:
        ref_runs = S.fold_runs(runs, "core", "EfficientNet-B2", task)
        if not ref_runs:
            fail(f"{task}: core EfficientNet-B2 seed-42 folds 0-4 not found")
        man0 = json.loads((Path(ref_runs[0].val_csv).parents[1] / "run_manifest.json").read_text())
        cfg = read_cfg(Path(ref_runs[0].val_csv).parents[1])
        csv = root / man0["labels_csv"]
        if not csv.exists():
            fail(f"{task}: labels CSV {csv} (from the manifest) does not exist")
        got, want = sha256(csv), man0.get("labels_csv_sha256")
        if want and got != want:
            msg = f"{task}: labels CSV sha256 {got[:12]} != manifest {want[:12]}"
            if not a.allow_sha_mismatch:
                fail(msg + " -- this is not the file the models were trained on")
            print("WARNING (--allow-sha-mismatch): " + msg)
        sid_col = cfg.get("sample_id_col", "image_id")
        lab_col = cfg.get("label_col", "label")
        pid_col = cfg.get("patient_id_col", "patient_id")
        lab = pd.read_csv(csv, dtype={sid_col: str, pid_col: str}, low_memory=False)
        for c in (sid_col, lab_col, pid_col):
            if c not in lab.columns:
                fail(f"{task}: column {c!r} not in {csv.name}; columns: {list(lab.columns)}")
        if lab[sid_col].duplicated().any():
            fail(f"{task}: {int(lab[sid_col].duplicated().sum())} duplicate sample ids in {csv.name}")
        age_col = pick(lab.columns, getattr(a, f"{task}_age_col"), AGE_CANDIDATES, f"{task}-age")
        sex_col = pick(lab.columns, getattr(a, f"{task}_sex_col"), SEX_CANDIDATES, f"{task}-sex")
        lab = lab.set_index(sid_col)
        scol = cfg.get("source_col") or "source_dataset"
        if scol in lab.columns:
            lab["_src"] = lab[scol].astype(str).str.lower()
        else:                                   # e.g. "brset_img00021" -> brset
            pre = lab.index.to_series().str.split("_").str[0].str.lower()
            lab["_src"] = pre.where(pre.isin(["brset", "odir"]), "all") if task == "hypertension" else "all"
        if task == "hypertension" and not set(lab["_src"]) <= {"brset", "odir"}:
            fail(f"{task}: cannot tell BRSET from ODIR rows (source values {sorted(set(lab['_src']))[:5]})")
        lab["_age"] = pd.to_numeric(lab[age_col], errors="coerce")
        lab["_sex"], sexinfo = encode_sex(lab[sex_col], lab["_src"],
                                          parse_codes(getattr(a, f"{task}_sex_codes")))
        info[task] = dict(labels_csv=str(man0["labels_csv"]), sha256=got, sample_id_col=sid_col,
                          label_col=lab_col, patient_id_col=pid_col, age_col=age_col, sex_col=sex_col,
                          age_missing=int(lab["_age"].isna().sum()), rows=len(lab),
                          age_min=float(lab["_age"].min()), age_max=float(lab["_age"].max()),
                          age_mean=float(lab["_age"].mean()), sex=sexinfo,
                          sex_missing=int(lab["_sex"].isna().sum()),
                          by_dataset={k: dict(rows=len(g), age_missing=int(g["_age"].isna().sum()),
                                              age_mean=float(g["_age"].mean()),
                                              sex_missing=int(g["_sex"].isna().sum()),
                                              share_coded_1=float(g["_sex"].mean()))
                                      for k, g in lab.groupby("_src")})
        bd = info[task]["by_dataset"]
        if len(bd) > 1:
            miss = {k: v["age_missing"] / v["rows"] for k, v in bd.items()}
            if max(miss.values()) - min(miss.values()) > 0.05:
                print(f"WARNING: {task}: age missingness differs by dataset {miss}; the missing-age "
                      "indicator then partly encodes the dataset. Judge age+sex on BRSET-only rows.")
        print(f"{task}: {csv.name} ({len(lab)} rows)  age={age_col!r} "
              f"[{info[task]['age_min']:.0f}-{info[task]['age_max']:.0f}, {info[task]['age_missing']} missing]  "
              f"sex={sex_col!r} [{sexinfo['coding']}] ({info[task]['sex_missing']} missing)", flush=True)

        core = runs[(runs.arm == "core") & (runs.task == task) & (runs.seed == 42) & (~runs.replicate)]
        for f in range(5):
            r = ref_runs[f]
            m = json.loads((Path(r.val_csv).parents[1] / "run_manifest.json").read_text())
            te, iv = S.preds(r.val_csv), S.preds(r.inner_csv)
            missing = set(te.sample_id) | set(iv.sample_id)
            missing -= set(lab.index)
            if missing:
                fail(f"{task} fold {f}: {len(missing)} prediction ids not in the labels CSV, e.g. {sorted(missing)[:3]}")
            tr_ids = lab.index.difference(pd.Index(te.sample_id)).difference(pd.Index(iv.sample_id))
            if m.get("n_train") is not None and len(tr_ids) != m["n_train"]:
                fail(f"{task} fold {f}: rebuilt train set has {len(tr_ids)} rows, manifest n_train is {m['n_train']}")
            for name, d in (("test", te), ("inner", iv)):
                yl = lab.loc[d.sample_id, lab_col].to_numpy(float)
                if not np.array_equal(yl, d.y_true.to_numpy(float)):
                    fail(f"{task} fold {f}: labels in CSV disagree with {name} predictions")
            tr = lab.loc[tr_ids].assign(source_dataset=lab.loc[tr_ids, "_src"].to_numpy())
            tef = lab.loc[te.sample_id].assign(source_dataset=te.source_dataset.to_numpy())
            ivf = lab.loc[iv.sample_id].assign(source_dataset=iv.source_dataset.to_numpy())
            ref = dict(age_mean=float(tr["_age"].mean()), age_sd=float(tr["_age"].std()) or 1.0,
                       age_median=float(tr["_age"].median()))
            kinds = ["age", "sex", "age+sex"] + (["dataset", "age+sex+dataset"] if task == "hypertension" else [])
            outs = {}
            for k in kinds:
                p_iv, p_te = fit_predict(design(tr, k, ref), tr[lab_col].to_numpy(int),
                                         [design(ivf, k, ref), design(tef, k, ref)])
                outs[k] = (p_iv, p_te)
            # image + age + sex, stacked on inner-val, per core image model
            for rr in core[core.fold == f].itertuples():
                iv_img, te_img = S.preds(rr.inner_csv), S.preds(rr.val_csv)
                if not (iv_img.sample_id.equals(iv.sample_id) and te_img.sample_id.equals(te.sample_id)):
                    fail(f"{task} fold {f}: {rr.label} used different inner/test images")
                Xiv = design(ivf, "image+age+sex", ref, iv_img.y_logit.to_numpy(float))
                Xte = design(tef, "image+age+sex", ref, te_img.y_logit.to_numpy(float))
                (p_te,) = fit_predict(Xiv, iv.y_true.to_numpy(int), [Xte])
                (p_iv,) = fit_predict(Xiv, iv.y_true.to_numpy(int), [Xiv])   # in-sample, threshold only
                outs[f"{rr.label}+age+sex"] = (p_iv, p_te)
            for k, (p_iv, p_te) in outs.items():
                pred_store[(task, k, f)] = p_te
                y, yi = te.y_true.to_numpy(int), iv.y_true.to_numpy(int)
                t_s, _ = S.thresholds(pd.DataFrame({"y_true": yi, "y_prob": p_iv}), a.target_sens)
                c = N.confusion(y, p_te, t_s)
                row = dict(task=task, model=k, fold=f, n_train=len(tr_ids), auroc=S.auc(y, p_te),
                           auprc=N.ap(y, p_te), ece=S.ece(y, p_te), brier=S.brier(y, p_te),
                           tsens_threshold=t_s, tsens_sens=c["sens"], tsens_spec=c["spec"])
                for coh in COHORTS[task][1:]:
                    mk = (te.source_dataset == coh).to_numpy()
                    row[f"{coh}_auroc"] = S.auc(y[mk], p_te[mk]); row[f"{coh}_auprc"] = N.ap(y[mk], p_te[mk])
                fold_rows.append(row)
                d = pd.DataFrame({"sample_id": te.sample_id, "patient_id": te.patient_id,
                                  "source_dataset": te.source_dataset, "y_true": y, "y_prob": p_te})
                pdir = a.out / "predictions" / task / k.replace("+", "_plus_")
                pdir.mkdir(parents=True, exist_ok=True)
                d.to_csv(pdir / f"fold_{f}_test_predictions.csv", index=False)
        pred_store[(task, "_lab")] = lab
        print(f"  {task}: 5 folds fitted", flush=True)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(a.out / "baseline_folds.csv", index=False)
    (a.out / "columns_used.json").write_text(json.dumps(info, indent=2, default=str))

    # ---- bootstrap: baselines and image models on the SAME draws ------------
    ci_rows, cmp_rows = [], []
    for task in a.tasks:
        ref_runs = S.fold_runs(runs, "core", "EfficientNet-B2", task)
        core_labels = sorted(runs[(runs.arm == "core") & (runs.task == task)].label.unique())
        base = sorted({k for (t, k, *_) in pred_store if t == task and k != "_lab"})
        for coh in COHORTS[task]:
            rng = np.random.default_rng(a.seed)
            acc = {}
            for f in range(5):
                ref = S.preds(ref_runs[f].val_csv)
                keep = (np.ones(len(ref), bool) if coh == "pooled"
                        else (ref.source_dataset == coh).to_numpy())
                W = S.patient_weights(ref.patient_id.to_numpy()[keep], a.boot, rng)
                Wd = W.astype(float)
                y = ref.y_true.to_numpy()[keep]
                scores = {k: pred_store[(task, k, f)][keep] for k in base}
                for lbl in core_labels:
                    rr = S.fold_runs(runs, "core", lbl, task)[f]
                    scores[f"image: {lbl}"] = S.preds(rr.val_csv).y_prob.to_numpy()[keep]
                for k, s in scores.items():
                    e = acc.setdefault(k, {"auroc": [0.0, np.zeros(a.boot)], "auprc": [0.0, np.zeros(a.boot)]})
                    e["auroc"][0] += S.auc(y, s) / 5; e["auroc"][1] += S.auc_weighted(y, s, W) / 5
                    e["auprc"][0] += N.ap(y, s) / 5; e["auprc"][1] += N.ap_weighted(y, s, Wd) / 5
            for k, ms in acc.items():
                for met, (pt, dist) in ms.items():
                    lo, hi = S.ci(dist)
                    ci_rows.append(dict(task=task, cohort=coh, model=k, metric=met, value=pt, ci_lo=lo, ci_hi=hi))
            pairs = []
            for lbl in core_labels:
                img, stk = f"image: {lbl}", f"{lbl}+age+sex"
                pairs += [("image vs age+sex", img, "age+sex"),
                          ("image+age+sex vs age+sex (does the image add?)", stk, "age+sex"),
                          ("image+age+sex vs image (does age/sex add?)", stk, img)]
            if task == "hypertension":
                pairs += [("age+sex+dataset vs age+sex", "age+sex+dataset", "age+sex")]
            for fam, A, B in pairs:
                for met in ("auroc", "auprc"):
                    d = acc[A][met][1] - acc[B][met][1]
                    lo, hi = S.ci(d)
                    cmp_rows.append(dict(family=fam, task=task, cohort=coh, metric=met, a=A, b=B,
                                         value_a=acc[A][met][0], value_b=acc[B][met][0],
                                         diff=acc[A][met][0] - acc[B][met][0], ci_lo=lo, ci_hi=hi, p=S.pval(d)))
            print(f"  bootstrap done: {task}/{coh}", flush=True)
    ci = pd.DataFrame(ci_rows); ci.to_csv(a.out / "baseline_ci.csv", index=False)
    cmp_ = pd.DataFrame(cmp_rows)
    cmp_["p_holm"] = np.nan
    for _, idx in cmp_.groupby(["family", "task", "cohort", "metric"]).groups.items():
        cmp_.loc[idx, "p_holm"] = S.holm(cmp_.loc[idx, "p"].tolist())
    cmp_.to_csv(a.out / "baseline_comparisons.csv", index=False)

    # ---- is the image score an age estimate? -------------------------------
    from scipy.stats import spearmanr
    corr_rows, band_rows = [], []
    for task in a.tasks:
        lab = pred_store[(task, "_lab")]
        tests = []
        for lbl in sorted(runs[(runs.arm == "core") & (runs.task == task)].label.unique()):
            for f, rr in S.fold_runs(runs, "core", lbl, task).items():
                d = S.preds(rr.val_csv).assign(model=lbl, fold=f)
                d["age"] = lab.loc[d.sample_id, "_age"].to_numpy()
                tests.append(d)
        t = pd.concat(tests)
        edges = np.unique(np.nanquantile(lab["_age"], [0, 1 / 3, 2 / 3, 1]))
        t["age_band"] = pd.cut(t.age, edges, include_lowest=True, duplicates="drop").astype(str)
        for coh in COHORTS[task]:
            tc = t if coh == "pooled" else t[t.source_dataset == coh]
            for (lbl, f), g in tc.groupby(["model", "fold"]):
                ok = g.age.notna()
                rho_all = spearmanr(g.y_prob[ok], g.age[ok]).statistic
                neg = ok & (g.y_true == 0)
                rho_neg = spearmanr(g.y_prob[neg], g.age[neg]).statistic
                corr_rows.append(dict(task=task, cohort=coh, model=lbl, fold=f,
                                      spearman_score_age=rho_all, spearman_score_age_negatives_only=rho_neg))
            for (lbl, band), g in tc[tc.age.notna()].groupby(["model", "age_band"]):
                per_f = [S.auc(x.y_true, x.y_prob) for _, x in g.groupby("fold") if x.y_true.nunique() == 2]
                band_rows.append(dict(task=task, cohort=coh, model=lbl, age_band=band,
                                      images=len(g), positives=int(g.y_true.sum()),
                                      prevalence=float(g.y_true.mean()),
                                      auroc_mean_over_folds=float(np.mean(per_f)) if per_f else np.nan,
                                      auroc_sd_over_folds=float(np.std(per_f, ddof=1)) if len(per_f) > 1 else np.nan))
    corr = pd.DataFrame(corr_rows)
    corr.groupby(["task", "cohort", "model"]).mean(numeric_only=True).drop(columns="fold") \
        .reset_index().to_csv(a.out / "age_correlation.csv", index=False)
    corr.to_csv(a.out / "age_correlation_per_fold.csv", index=False)
    pd.DataFrame(band_rows).to_csv(a.out / "age_band_auroc.csv", index=False)

    # ---- summary -------------------------------------------------------------
    md = ["# Age / sex baselines\n", f"{a.boot} patient-bootstrap replicates, seed {a.seed}, "
          "same draws as isbi_stats.py.\n", "## Columns used\n", "```", json.dumps(info, indent=1, default=str), "```\n"]
    for task in a.tasks:
        for coh in COHORTS[task]:
            t = ci[(ci.task == task) & (ci.cohort == coh) & (ci.metric == "auroc")].sort_values("value", ascending=False)
            md += [f"## {task}, {coh}: mean 5-fold AUROC [95% CI]\n",
                   S.md_table(pd.DataFrame({"model": t.model, "AUROC": [f"{v:.4f} [{l:.4f}, {h:.4f}]" for v, l, h in
                                                                         zip(t.value, t.ci_lo, t.ci_hi)]})), ""]
            c = cmp_[(cmp_.task == task) & (cmp_.cohort == coh) & (cmp_.metric == "auroc")]
            md += [f"### {task}, {coh}: paired AUROC differences\n",
                   S.md_table(pd.DataFrame({"comparison": c.family, "A": c.a, "B": c.b,
                                            "A−B": c["diff"].map("{:+.4f}".format),
                                            "95% CI": [f"[{l:+.4f}, {h:+.4f}]" for l, h in zip(c.ci_lo, c.ci_hi)],
                                            "p (Holm)": c.p_holm.map("{:.4f}".format)})), ""]
    cs = pd.read_csv(a.out / "age_correlation.csv")
    md += ["## Spearman correlation of image-model score with age (mean over folds)\n",
           S.md_table(cs.round(3)), ""]
    (a.out / "summary.md").write_text("\n".join(md))
    print("\n".join(md[:6]))
    print(f"\nwritten to {a.out}/ : summary.md, baseline_ci.csv, baseline_comparisons.csv, baseline_folds.csv, "
          "age_correlation*.csv, age_band_auroc.csv, columns_used.json, predictions/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
