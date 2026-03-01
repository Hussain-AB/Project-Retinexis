# Proposed Monorepo Restructure

```
Project-Retinexis/
│
├── src/                                    # All importable shared code
│   ├── __init__.py
│   │
│   ├── models/                             # Model definitions (already built)
│   │   ├── __init__.py
│   │   ├── base.py                         #   BaseClassifier, PreprocessingConfig
│   │   ├── registry.py                     #   MODEL_REGISTRY dict
│   │   ├── factory.py                      #   create_model()
│   │   ├── efficientnet.py                 #   done
│   │   ├── dinov2.py                       
│   │   ├── retfound_mae.py
│   │   ├── retfound_dinov2.py
│   │   ├── dinov3.py
│   │   ├── seresnext.py
│   │   ├── convnextv2.py
│   │   ├── retfound_green.py
│   │   └── resnet.py
│   │
│   ├── data/                               # Data loading infrastructure
│   │   ├── __init__.py
│   │   ├── base_dataset.py                 #   Abstract base — defines return contract
│   │   ├── transforms.py                   #   Build transforms from PreprocessingConfig
│   │   ├── splits.py                       #   StratifiedGroupKFold, save/load splits
│   │   ├── loader.py                       #   DataLoader factory
│   │   └── datasets/                       #   Team-specific dataset implementations
│   │       ├── __init__.py
│   │       ├── cimt.py                     #   CIMTDataset — CIMT team owns
│   │       └── brset.py                    #   BRSETDataset — BRSET team owns
│   │
│   ├── training/                           # Training loop and optimisation
│   │   ├── __init__.py
│   │   ├── trainer.py                      #   Trainer class (Phase 1 + Phase 2)
│   │   ├── optimiser.py                    #   AdamW, layer-wise LR decay
│   │   ├── scheduler.py                    #   Cosine with warmup
│   │   ├── callbacks.py                    #   Early stopping
│   │   └── loss.py                         #   BCE with logits, weighted BCE
│   │
│   ├── evaluation/                         # Metrics, predictions, external eval
│   │   ├── __init__.py
│   │   ├── metrics.py                      #   AUROC, AUPRC, ECE, sens, spec, F1
│   │   ├── predictions.py                  #   Save/load prediction CSVs
│   │   └── external.py                     #   Zero-shot + adapted external eval
│   │
│   ├── config/                             # Config loading and validation
│   │   ├── __init__.py
│   │   ├── schema.py                       #   TrainConfig dataclass
│   │   └── loader.py                       #   YAML loading + CLI override merging
│   │
│   └── utils/                              # Shared utilities
│       ├── __init__.py
│       ├── logging.py                      #   CSV logger, optional W&B wrapper
│       ├── reproducibility.py              #   Seed everything, deterministic flags
│       └── checkpoint.py                   #   Save/load checkpoint dicts
│
├── tasks/                                  # Team-specific configs, splits, EDA, analysis
│   ├── cimt/
│   │   ├── configs/                        #   One YAML per model
│   │   │   ├── efficientnet_b2.yaml
│   │   │   ├── retfound_mae.yaml
│   │   │   ├── retfound_dinov2.yaml
│   │   │   ├── dinov2_large.yaml
│   │   │   └── ...
│   │   ├── splits/                         #   Saved fold splits (committed to repo)
│   │   │   └── cimt_5fold_splits.json
│   │   ├── eda/                            #   CIMT EDA lives here
│   │   └── analysis/                       #   Post-training analysis
│   │       ├── phase1_results.ipynb        #     Linear probe triage
│   │       ├── model_comparison.ipynb      #     Phase 2 results, significance tests
│   │       └── cohort_shift.ipynb          #     Internal vs external performance
│   │
│   └── brset/
│       ├── configs/
│       │   ├── efficientnet_b2.yaml
│       │   ├── retfound_mae.yaml
│       │   └── ...
│       ├── splits/
│       │   └── brset_5fold_splits.json
│       ├── eda/                            #   BRSET EDA lives here
│       └── analysis/
│           ├── phase1_results.ipynb
│           ├── model_comparison.ipynb
│           └── cohort_shift.ipynb
│
├── scripts/                                # Entry points and Ibex job management
│   ├── train.py                            #   Main training entry point
│   ├── evaluate.py                         #   Standalone evaluation
│   ├── generate_splits.py                  #   One-time fold split generation
│   └── ibex/                               #   SLURM job infrastructure
│       ├── env_setup.sh                    #     Conda/module setup (sourced by all jobs)
│       ├── job_template.sh                 #     SLURM jobscript with auto-resume
│       ├── submit_phase1.sh                #     Submit all Phase 1 (linear probe) jobs
│       └── submit_phase2.sh                #     Submit Phase 2 for selected models
│
├── tests/                                  # All tests
│   ├── conftest.py
│   ├── models/
│   │   ├── test_models.py                  #   done — universal contract tests
│   │   └── test_efficientnet.py            #   done — model-specific tests; can be adpated per model
│   ├── data/
│   │   ├── test_transforms.py
│   │   ├── test_splits.py
│   │   ├── test_cimt_dataset.py            #   CIMT team owns
│   │   └── test_brset_dataset.py           #   BRSET team owns
│   ├── training/
│   │   ├── test_trainer.py
│   │   └── test_optimiser.py
│   ├── evaluation/
│   │   └── test_metrics.py
│   └── config/
│       └── test_config.py
│
├── outputs/                                # .gitignored — all run outputs
│   ├── cimt/
│   │   └── {model_name}/
│   │       └── fold_{k}/
│   │           ├── checkpoints/            #     best.pt, last.pt
│   │           ├── predictions/            #     val_predictions.csv
│   │           ├── logs/                   #     training_log.csv
│   │           └── config_resolved.yaml    #     exact config used
│   └── brset/
│       └── {model_name}/
│           └── fold_{k}/
│               └── ...
│
├── data/                                   # .gitignored — raw data (or symlink to /ibex/user/)
│   ├── cimt/
│   │   ├── images/
│   │   └── labels.csv
│   └── brset/
│       ├── images/
│       └── labels.csv
│
├── specs/                                   # Project documentation
│   ├── models.md                           
│   └── training_proposal.md                
│
├── pyproject.toml                          # Package metadata, pytest config
├── environment.yml                         # Pinned conda environment
├── requirements.txt                        # Pip fallback
├── README.md
└── .gitignore
```

## Import Examples

```python
# Models (unchanged from current)
from src.models import create_model, MODEL_REGISTRY, BaseClassifier

# Data
from src.data import build_transforms, create_dataloaders
from src.data.splits import load_splits, generate_splits
from src.data.datasets.cimt import CIMTDataset
from src.data.datasets.brset import BRSETDataset

# Training
from src.training import Trainer
from src.training.optimiser import build_optimiser, get_parameter_groups
from src.training.callbacks import EarlyStopping

# Evaluation
from src.evaluation.metrics import compute_metrics
from src.evaluation.predictions import save_predictions

# Config
from src.config import load_config, TrainConfig
```

## Ownership Boundaries

| Path | Owned by | Notes |
|------|----------|-------|
| `src/models/` | Both teams (model authors) | Anyone writes model files, all models shared |
| `src/data/base_dataset.py` | MLOps leads jointly | Defines the return contract |
| `src/data/transforms.py` | MLOps leads jointly | Driven by PreprocessingConfig |
| `src/data/splits.py` | MLOps leads jointly | Generic StratifiedGroupKFold |
| `src/data/loader.py` | MLOps leads jointly | Generic DataLoader factory |
| `src/data/datasets/cimt.py` | CIMT team | CIMT-specific CSV parsing and metadata |
| `src/data/datasets/brset.py` | BRSET team | BRSET-specific CSV parsing and metadata |
| `src/training/` | MLOps leads jointly | Shared training pipeline |
| `src/evaluation/` | MLOps leads jointly | Shared metrics and eval |
| `src/config/` | MLOps leads jointly | Shared config schema and loading |
| `tasks/cimt/` | CIMT team | Configs, splits, EDA, post-training analysis |
| `tasks/brset/` | BRSET team | Configs, splits, EDA, post-training analysis |
| `scripts/` | MLOps leads jointly | Entry points and Ibex scripts |
| `tests/data/test_cimt_dataset.py` | CIMT team | Tests for their dataset class |
| `tests/data/test_brset_dataset.py` | BRSET team | Tests for their dataset class |
| `tests/` (everything else) | MLOps leads jointly | Shared infrastructure tests |
