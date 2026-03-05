# Hypertension Data Into Pipeline

Last updated: 2026-03-05

- BRSET preprocessing currently derives a `hypertension` field from the `comorbidities` column in `labels_brset.csv`.
- BRSET preprocessing also preserves `hypertensive_retinopathy` as a separate field.
- ODIR preprocessing maps ODIR column `H` directly into a generic `label` field.
- The shared pipeline expects one binary training target column and one patient grouping column.


## Current Shared Pipeline Contract

The shared pipeline is driven by:

- a dataset class,
- a labels CSV,
- a splits JSON,
- a config YAML.

For a training run to work, these must agree exactly on sample ID, patient ID, label column, and image path logic.

Relevant shared files:

- `src/data/base_dataset.py`
- `src/data/loader.py`
- `src/config/schema.py`
- `scripts/train.py`
- `scripts/generate_splits.py`

## What A Future Hypertension Dataset Class Must Do

For a real combined run, it needs one dataset class that supports one canonical combined schema and one reliable image path strategy.

Minimum expected behavior:

1. Read one row from the combined metadata CSV.
2. Resolve the actual image file path for that row.
3. Cast the chosen binary target to float for BCE loss.
4. Preserve globally unique patient IDs for grouping and reporting.
5. Return a `SampleDict` compatible with the trainer.
6. Optionally encode metadata columns if the config enables metadata fusion.

Recommended path logic:

- prefer a `relative_image_path` column in the combined CSV,
- fall back to `image_filename` only if all images live under one unified root and filenames are globally unique.

Why this matters:

- BRSET and ODIR are separate sources,
- filenames alone may not be sufficient or collision-safe,
- the dataset implementation should not rely on filename-only path guessing unless the combined dataset layout is standardized first.

## Recommended Combined Metadata Schema

Ideally, one final combined CSV with these columns.

Required:

- `image_id`
- `patient_id`
- `label`
- `source_dataset`
- `image_filename`

Strongly recommended:

- `relative_image_path`
- `eye`
- `age`
- `age_normalized`
- `gender`

Expected meaning:

- `image_id`: globally unique sample identifier
- `patient_id`: globally unique patient identifier
- `label`: final binary target 
- `source_dataset`: `brset` or `odir`
- `image_filename`: raw filename
- `relative_image_path`: path relative to a common `data_dir`

Recommended ID convention:

- `image_id = brset_<original>` or `odir_<original>`
- `patient_id = brset_<original>` or `odir_<original>`

This will avoid collisions across datasets and makes per-source auditing much easier.

## Where BRSET and ODIR Currently Mismatch

### Label construction

BRSET current preprocessing:

- derives `hypertension` from text search in `comorbidities`,
- also keeps `hypertensive_retinopathy`.

ODIR current preprocessing:

- maps `H` straight to `label`.

This is the core semantic issue. The repo does not currently prove these are the same target.

### Schema naming

ODIR processed metadata currently outputs:

- `patient_id`
- `image_id`
- `image_filename`
- `label`

But the existing committed ODIR split file metadata still expects:

- `ID`
- `filename`
- `H`

That mismatch is fundamental, but it is not a pipeline bug. It means the processed metadata schema and the saved split schema were produced at different stages of the work and are no longer aligned.

For combined runs, the existing BRSET-only and ODIR-only split files should not be treated as final inputs. Would be best togenerate a new combined split file from the final combined CSV.

## BRSET Label Note

The hypertension team pointed to the BRSET paper to justify the BRSET labeling approach.

## Practical Files For Unblocking 
These are the artifacts that need to exist for a combined run.

Code-facing files:

- one new hypertension dataset implementation under `src/data/datasets/` (probably hypertension.py)
- `src/data/datasets/__init__.py`
- `tasks/hypertension/configs/phase1/<combined_config>.yaml`

Data-facing artifacts:

- final BRSET cleaned metadata in canonical columns
- final ODIR cleaned metadata in canonical columns
- one merged combined metadata CSV
- one combined 5-fold grouped-stratified split JSON
- one standardized image directory layout that matches the chosen path strategy

Practical output locations could be:

- `data/hypertension_combined/metadata_processed/combined_metadata.csv`
- `data/hypertension_combined/images/...`
- `tasks/hypertension/splits/hypertension_combined_5fold_splits.json`
- `tasks/hypertension/configs/phase1/hypertension_combined_<model>.yaml`

## What Must Match Exactly

These four things must agree exactly:

1. Combined metadata CSV column names
2. Combined split JSON metadata:
   - `sample_id_col`
   - `group_col`
   - `label_col`
3. Training config YAML:
   - `sample_id_col`
   - `patient_id_col`
   - `label_col`
4. the future hypertension dataset class path-resolution logic

If any one of those drifts, the run will fail or maybe silently train on the wrong interpretation of the data.

## Recommended Combined Image Layout

The cleanest option is to standardize one root such as:

```text
data/hypertension_combined/
├── images/
│   ├── brset/
│   │   └── ...
│   └── odir/
│       └── ...
└── metadata_processed/
    └── combined_metadata.csv
```

Then store:

- `relative_image_path = brset/<file>` or `odir/<file>`

This gives the future dataset class one stable rule:

- `image_path = data_dir / relative_image_path`

That is probably much more robust than guessing from source or filename.

## What The Combined Split File Should Be Built From

The combined split file should be regenerated from the final combined CSV only, using:

- globally unique `image_id`
- globally unique `patient_id`
- final pooled `label`

Probably best to not reuse the current dataset-specific split files for the combined experiment as they were generated before the final combined schema exists, and the ODIR split metadata is already schema-drifted from current ODIR processed metadata + combined runs need one patient-grouped split space across both sources.
