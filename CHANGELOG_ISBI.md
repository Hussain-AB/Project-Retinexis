# CHANGELOG — ISBI 2027 retrain preparation

Branch `isbi/retrain-prep`, cut from `main` @ `4cb3c73`.
All changes are pre-retrain: nothing here alters an already-trained model, and
the previous runs' splits are preserved so old results stay reproducible.

**Reference for every change:** `code_change_list.md` (delivered separately).
**Decisions this implements:** `retrain_decision_record.md`.

---

## Commits

| # | Commit | What |
|---|---|---|
| 1 | `5ff74f9` | merge: `xai/`, dinov3 checkpoint-load fix, restored 0-byte tests |
| 2 | `b9e3deb` | splits: nested inner-validation split, cohort stratification, duplicate guard |
| 3 | `c951f63` | data: de-duplicate hypertension metadata, regenerate both split sets |
| 4 | `8feb387` | port: vessel-mask data path forward from `Project-Retinexis-main` |
| 5 | `fbceb6d` | data: per-image identity in batches, three-way loaders, seeded workers |
| 6 | `798ec95` | config: seed-aware `run_dir`, prediction/manifest paths, scheduler horizon |
| 7 | `6d776b7` | trainer: per-image predictions CSV, unbiased selection, phase-2 fix, manifest |

Each commit message carries the full rationale and the verification performed.

---

## The headline changes

### 1. Per-image predictions CSV (requested)

Every run now writes **one row per image**, for both splits:

```
predictions/val_predictions.csv         held-out outer fold — the reported one
predictions/inner_val_predictions.csv   for temperature + threshold fitting
```

Columns: `sample_id, patient_id, source_dataset, split, y_true, y_logit, y_prob, y_pred, image_path`

- `val_predictions.csv` keeps its filename, so `aggregate_folds.py`, `calibrate.py`,
  `summarize_all_models.py` and `xai/check_frame.py` all keep working.
- `y_logit` is stored **raw**. `calibrate.py` used to recover logits by inverting a
  *clamped* sigmoid, which destroys the tails — every temperature fitted that way is
  contaminated. Fit from `y_logit` instead.
- `source_dataset` means within-cohort metrics need no positional join. Per-cohort
  AUROC is now printed at the end of every run.

### 2. Selection no longer happens on the reported fold

`generate_splits()` emits three patient-disjoint lists per fold. `val_sample_ids`
keeps its original meaning; `train_sample_ids` is now the outer training set minus
the new `inner_val_sample_ids`. Early stopping and `best.pt` selection read
inner-val; the outer fold is scored exactly once, at the end.

Expect reported AUROCs to fall ~0.005–0.015. **That drop is the bias that was
previously being reported.**

Back-compat: `inner_val_frac=0.0`, or a v1 splits file, restores the old behaviour
exactly (verified: train returns to 4,644 rows on CIMT fold 0) with a warning.

### 3. Three bugs that corrupted existing results

**`phase2_best_auroc` seeded from Phase 1.** Fine-tuning had to beat the *linear
probe* before writing any checkpoint, and `_run_phase1` writes `best.pt` itself when
`phase2_epochs == 0`. This is how SE-ResNeXt acquired a `best.pt`, a predictions file
and a Phase-2 summary row at 0.8418 while all five of its training logs contain
phase-1 epochs only. Now starts at 0.0.

**DataLoader workers were never seeded** (previously undiagnosed). PyTorch seeds only
its own RNG per worker; `transforms.py` augments with Python's `random`
(`RandomScale`, `RandomBlur`). On Linux all `num_workers` forks share one `random`
state, so every worker produced the *same* augmentation sequence and augmentation
randomness sat outside `config.seed`. That directly undermines the seed-variance
study. Fixed with `worker_init_fn` + a seeded generator, in both `create_dataloaders`
and `rebuild_loaders`.

**The cosine never annealed.** Built for `phase2_epochs=50` while early stopping fires
at 10–33, the LR multiplier was still 0.970 at epoch 10 — every model fine-tuned at an
effectively constant rate. `scheduler_total_epochs=30` fixes the horizon; the 50-epoch
cap and `patience=7` are untouched.

### 4. Seed is now part of `run_dir`

```
seed 42   -> {output_dir}/{dataset}/{arch}/fold_{k}          (unchanged)
otherwise -> {output_dir}/{dataset}/{arch}/fold_{k}_seed{s}
```

Without this a second seed overwrites the first's `best.pt`, silently destroying the
seed study the entire variance analysis rests on. `xai/modelio.py::checkpoint_path`
mirrors it — they must stay in step.

### 5. Data integrity

`scripts/fix_combined_metadata.py` removes the 6,029 fully-duplicated rows (22.4%):

| | before | after |
|---|---|---|
| rows | 26,957 | **20,928** |
| prevalence | 0.1256 | **0.1523** |
| BRSET : ODIR | 53 : 47 | **68 : 32** |

It never modifies its input and refuses to collapse any ID appearing twice with
*different* content. ODIR is kept in full — no QC filtering, per decision.

Loader guards that would have caught this on day one: `_subset_df` now raises when
split IDs are missing from the CSV, and always raises when rows outnumber distinct
IDs. `generate_splits()` rejects duplicate sample IDs outright.

### 6. Vessel path restored

The branch point predates the vessel work, yet this tree contains five trained
`efficientnet_vessel` folds — `_forward` called `self.model(images)` with one argument
while `EfficientNetVessel.forward` needs two, so **those runs cannot have come from
this commit**. Ported the four files forward (`base_dataset`, `loader`, `cimt`,
`trainer`) and added the vessel branch to `hypertension.py`, which had it in neither
tree despite being the task whose 22,255 RRWNet masks actually exist.

---

## ⚠️ Open issues

**1. `gender` still mixes encodings, and the metadata arm will read it as site.**
By explicit decision the column is untouched: BRSET `'1'`/`'2'` alongside ODIR
`'Male'`/`'Female'`, because `merge_metadata.py` reads raw `patient_sex` in preference
to the processed `gender`. Those four values partition **perfectly** by cohort, so
`MetadataEncoder.fit()` one-hot encodes them into features that identify the source
dataset. A model given "age + sex" is really given "age + sex + site". This affects
the selected metadata-fusion arm and any demographic baseline. Evidence for the
mapping, if you revisit: `metadata_processing.py:90` maps `{1:1, 2:0}`,
`Specs/data_prep.md:166` says `1 = Male, 0 = Female`, and BRSET is 62% code-2,
matching its published female majority.

**2. Provenance of the existing vessel and ODIR outputs is unexplained.** See §6
above, and the ODIR dataprep logs (`STATUS: FAIL`, 3,364 processed vs 8,002 on disk).

**3. `input_size` spans 224–518 across the roster.** Native resolution was chosen, so
the architecture comparison stays confounded with pixel count. A ~4 GPU-h probe
(DINOv2-L at 224 on fold 0) would bound the effect.

---

## Still to do before launch

| Item | Where |
|---|---|
| `logging.py`: rename `val_*` → `inner_val_*`, add `run_id`/`restart_idx` | §F |
| 24 config YAMLs: `use_weighted_bce: false`, `source_col`, new paths | §D4 |
| `scripts/ibex/submit_grid.sh`; fill the 0-byte `env_setup.sh` | §I |
| New tests: nested splits, predictions schema, worker seeding, phase-of-best | §K |
| Metrics: Brier, equal-mass ECE, cluster bootstrap, threshold selection | §G |
| `aggregate_folds.py`: drop the epoch-averaged harmonic mean | §H |

---

## Regenerating the data artifacts

`data/` is gitignored, so these are not committed:

```bash
python scripts/fix_combined_metadata.py \
  --csv data/hypertension_combined/metadata_processed/combined_metadata.csv \
  --out data/hypertension_combined/metadata_processed/combined_metadata_dedup.csv

python tasks/hypertension/splits/generate_combined_splits.py

python scripts/generate_splits.py \
  --csv data/cimt/Fundus_CIMT_2903_Dataset/metadata_processed/processed_metadata.csv \
  --out tasks/cimt/splits/cimt_5fold_splits.json \
  --task-name cimt --group-col patient_id --label-col label \
  --sample-id-col image_id --n-splits 5 --seed 42 --inner-val-frac 0.15
```

The splits the **existing** checkpoints were trained on are preserved as
`*_splits.v1.json`.

---

## Verification performed

- `generate_splits` / `validate_splits` against real CIMT and hypertension metadata:
  three-way patient disjointness, full per-fold coverage, val folds partition the
  dataset exactly once, zero validation errors.
- Duplicate guard rejects the live hypertension CSV (6,029 dups).
- `_load_fold` / `_subset_df` exercised from the shipped source: exact row-per-ID
  counts, duplicate guard, strict and non-strict missing-ID behaviour, v1 back-compat.
- `run_dir` differs by seed; all three new path properties resolve; `validate()`
  accepts defaults and rejects each new out-of-range value.
- Every touched file byte-compiles; trainer AST verified complete.

Not yet verified: an end-to-end training step. `torch` is not installed in this
environment, so the one-epoch smoke test (`code_change_list.md` §9) must run on IBEX
before the grid is submitted.
