# Grad-CAM + Integrated Gradients over every held-out image

Added October 2026. It sits next to the RISE pipeline in `README.md` and does not
change it: same canonical 512 × 512 frame, same regions (`anatomy.py`), same
metrics and smoothness-matched null (`metrics.py`, `attribution.py`).

| file | what it does |
|---|---|
| `gradcam.py` | Grad-CAM with a fixed, documented target layer per architecture |
| `run_maps.py` | one trained run → Grad-CAM + IG for **all** its held-out test images, region numbers computed in the same pass, 128 × 128 copies of the maps kept |
| `summarize_maps.py` | all runs → tables, verdicts and figures |
| `../scripts/ibex/submit_xai.sh` | the whole grid as SLURM arrays (resumable) |

## Why both methods, and the two nulls

Grad-CAM on a ViT needs a layer choice and a token reshape; on a CNN it does not.
IG uses only input gradients, so it runs identical code for every architecture.
**Report a pattern only if both methods show it.**

Every region number is compared against two nulls:

1. **A smoothness-matched random map.** This is the same null as in `analyze.py`. A compact region scores above 1.0 on smooth noise alone.
2. **The same architecture with random weights** (`--random-init`). Gradient-times-input methods highlight edges (vessels, disc, the field-of-view rim) even when the model has learned nothing (Adebayo et al., 2018). A region counts as *learned* only if the trained model differs from its random-weight twin. The summary has a table for exactly this.

## Target layers

The layer for each run is printed and stored in `run_info.json`.

- **CNN:** the last spatial feature map before global pooling.
- **ViT:** `norm1` of the last transformer block. It is not the block output: when the head reads only the class token, gradients at the last block's patch-token outputs are zero. The leading class and register tokens are removed and the rest are reshaped to the patch grid.

## Order of work

```bash
# 0. CIMT vessel masks, made from the SAME folder the models train on
python scripts/generate_vessel_masks.py --help
python scripts/generate_vessel_masks.py \
    --images-dir data/cimt/Fundus_CIMT_2903_Dataset/images_preprocessed \
    --save-dir   data/cimt/vessel_masks

# 1. smoke test: 16 images, one CNN and one ViT per task (check the printed layer and fidelity)
python -m xai.run_maps --task cimt --model efficientnet_b2 --fold 0 \
    --vessel-dir data/cimt/vessel_masks --limit 16 --out-root outputs/xai_smoke

# 2. everything
bash scripts/ibex/submit_xai.sh --dry-run
bash scripts/ibex/submit_xai.sh

# 3. tables and figures (CPU only; ~64 GB RAM for the agreement step)
python -m xai.summarize_maps --out-root outputs/xai_full
```

## Checks built in

- **Run checks.** `run_maps` refuses a run whose manifest is missing, a run trained from a dirty tree, or a run whose fold or seed don't match. It also refuses when more than 1% of the test images have no vessel mask.
- **Fidelity.** Every image's canonical-frame logit is stored next to the logit the run recorded for it. `fidelity.csv` gives the AUROC change. A large change means the maps describe a slightly different input from the one the model was scored on.
- **IG completeness.** Sum(IG) / (logit(image) − logit(baseline)) is stored per image. A median far from 1 means `--ig-steps` should be raised.
- **The ceiling.** `agreement.csv` compares cross-model agreement with the *same* model retrained with another seed. If the two are about equal, model "differences" are retraining noise.
