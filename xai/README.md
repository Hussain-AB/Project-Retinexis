# `xai/` — anatomically anchored attribution study

Experiment plan and code for the ISBI submission. Claim under test:

> Six models spanning 9M–304M parameters fall within 0.007 AUROC on CIMT
> (0.818–0.825) and 0.010 on hypertension, yet rely on measurably different
> retinal evidence. AUROC parity conceals reasoning divergence.

**That claim is not yet supported.** Step 3 below decides whether it survives.
Run the steps in order and do not skip Step 0 or Step 3.

---

## Why not Grad-CAM

The model set splits into four ViTs (RETFound-DINOv2 @518px, RETFound-MAE
@224px, DINOv2-L @504px, DINOv3-L) and two CNNs (ConvNeXtV2-B @224px,
EfficientNet-B2 @260px). Grad-CAM on the ViTs needs a reshape transform and an
arbitrary block choice; on the CNNs it needs neither. Every difference it
measures is confounded by the method × architecture interaction. Reviewers
will cite the ViT Grad-CAM audit literature at you.

Instead:

| method | role | why it is comparable |
|---|---|---|
| **RISE** | primary | pure black box — forward passes only. No layer choice, no reshape. **We** set the output resolution via the mask grid, not the network's feature stride. |
| **Integrated Gradients (+SmoothGrad)** | secondary | needs only ∂output/∂input, which every model has. ~50× cheaper than RISE. |
| **Occlusion** | control | deterministic, no sampling noise. Confirms RISE isn't a mask artifact. |

Any claim in the paper must hold under at least two of the three.

Two fundus-specific choices, both load-bearing:

- **Never occlude to black.** A black region looks like the FOV surround —
  in-distribution for all six models — so the attribution collapses. Everything
  here blends toward a heavily blurred copy of the same image, which preserves
  colour and illumination while destroying vessel calibre, disc margin and
  texture.
- **The canonical frame.** The six models use different `crop_strategy`
  (`resize` vs `pad_to_square`) and different input sizes, so a given original
  pixel lands in *different* places in different models' input tensors.
  Everything is therefore evaluated on one shared square 512×512 frame, where
  both crop strategies are no-ops. See `frames.py`. Step 0 verifies this
  doesn't change what the models do.

---

## Step 0 — preconditions (do this first, it takes 10 minutes)

```bash
# Regenerate vessel masks from the SAME directory the models train on.
# generate_vessel_masks.py defaults to the RAW images dir; the configs use
# images_preprocessed/. If those differ, every enrichment number is wrong.
python scripts/generate_vessel_masks.py \
  --images-dir data/cimt/Fundus_CIMT_2903_Dataset/images_preprocessed \
  --save-dir   data/cimt/vessel_masks

python -m xai.check_frame --task cimt --fold 0 --vessel-dir data/cimt/vessel_masks
```

Checks, each of which silently invalidates the study if it fails:

1. canonical-frame AUROC reproduces the recorded `val_predictions.csv` AUROC
   (tolerance 0.01). If not, report it or fall back to per-model frames.
2. vessel masks register to the training images.
3. disc estimates land inside the FOV.

Then eyeball the disc localiser and **record the hit rate for the paper**:

```bash
python -m xai.figures disc-qc --task cimt --fold 0 --vessel-dir data/cimt/vessel_masks
```

You need fold checkpoints (`best.pt`) under
`outputs/{dataset}/{architecture}/fold_{k}/checkpoints/`. Pull them from the
team HF repo.

---

## Step 1 — attribution maps

Fold-matched and strictly out-of-fold: for fold *k*, every model's fold-*k*
checkpoint is scored on the images held out in fold *k*. All six models see
identical images that none of them trained on.

```bash
python -m xai.run_attribution --task cimt --method rise \
  --vessel-dir data/cimt/vessel_masks --folds 0 --n-images 200 --n-masks 2000
```

Resumable — one `.npz` per (model, fold), skipped if present.

**Budget, one GPU with AMP.** The large ViTs dominate: DINOv2-L at 504px is
1296 tokens, ~5–8× the cost of ConvNeXtV2 at 224px.

| model | ≈ per image @ N=2000 |
|---|---|
| EfficientNet-B2, ConvNeXtV2, RETFound-MAE, DINOv3-L | 2–4 s |
| DINOv2-L, RETFound-DINOv2 | 15–25 s |

≈ 4–6 h per task for one fold at `--n-images 200`. Do fold 0 end-to-end first,
then widen. `--method ig` is ~50× cheaper for a same-day answer.

**Convergence.** Every map carries `split_half_r` (correlation of two
independent halves of the mask set). Median below 0.90 → raise `--n-masks`. An
unconverged map reads as "model disagreement" and will fool you.

---

## Step 2 — regional metrics

```bash
python -m xai.analyze --task cimt --method rise --vessel-dir data/cimt/vessel_masks
```

Regions (`anatomy.py`), all confined to the FOV:

| ROI | definition | why |
|---|---|---|
| disc | heuristic disc centre, r = 0.065 × FOV diameter | highest-contrast structure; models often lock on |
| **zone_b** | annulus 0.5–1.0 DD from disc margin | where CRAE/CRVE/AVR are clinically measured — the strongest hypertension prior available |
| zone_c | 0.5–2.0 DD | wider vascular zone |
| artery / vein | RRWNet A/V channels | hypertensive narrowing is arteriolar-specific |
| vessel / skeleton / adjacent | RRWNet vessel channel | separates "on the vessel" from "near it" |
| macula | 1 DD disc, 2.5 DD temporal | |
| background | FOV minus vessels and disc | residual |
| **periphery** | outer 10% of FOV | failure detector — attribution here = border/vignetting artifacts, not biology |

Metrics: **enrichment ratio** (mass in ROI ÷ area fraction; 1.0 = uniform —
this is the headline number), pointing game, soft-IoU@top-10%, and the
**radial profile** (density vs distance from the disc, the best single figure).

Every one is reported against a **smoothness-matched random null**. A compact
ROI scores above 1.0 on smoothed noise alone, so enrichment without the null
number is meaningless. Holm correction is applied across the ROI × model grid
(~130 comparisons — uncorrected, several will clear 0.05 by chance).

---

## Step 3 — the ceiling. Do this before writing anything.

If two independently-trained runs of the *same* architecture disagree as much
as two different architectures do, there is no divergence result — only noise.

Train 2–3 extra seeds for at least one ViT and one CNN on fold 0, save as
`best_seed1.pt` / `best_seed2.pt` beside the originals, then:

```bash
python -m xai.run_attribution --task cimt --method rise --folds 0 \
  --models dinov2_large efficientnet_b2 --checkpoint-suffix seed1 \
  --vessel-dir data/cimt/vessel_masks
python -m xai.analyze --task cimt --method rise --vessel-dir data/cimt/vessel_masks
```

`analyze.py` prints `ceiling − cross_model`. A small or negative gap means the
divergence claim is dead — keep the descriptive title
("Beyond AUROC: Where Retinal Foundation Models Look…") and report the null
result, which is still worth publishing and still honest.

Cross-*fold* agreement is **not** a valid ceiling: 4 of the 5 fold checkpoints
saw any given image during training, and memorised images produce visibly
sharper saliency.

---

## Step 4 — the causal test. This is what makes it a paper.

```bash
python -m xai.run_ablation --task cimt --vessel-dir data/cimt/vessel_masks \
  --folds 0 1 2 3 4 --age-col age --sex-col sex
```

Masks each region (blending to the blurred baseline) and measures the AUROC
drop. Enrichment says a model's saliency *points* at vessels; this says whether
it *uses* them.

> "RETFound shows 2.1× vessel enrichment and loses 0.06 AUROC when vessels are
> masked; EfficientNet-B2 shows 1.2× and loses 0.01"

is publishable. Enrichment alone is not. High enrichment with a near-zero AUROC
drop is itself a finding: the explanation does not describe the computation.

`--age-col/--sex-col` also runs the demographic confound check. Fundus networks
read age and sex well; if these models predict CIMT largely through demographic
proxies then "where they look" is partly "where they read age from", and a
reviewer *will* ask. Have the number.

---

## Step 5 — figures

```bash
python -m xai.figures all --task cimt --method rise
```

`radial` is the one to keep. `ablation` is the second. In 4 pages, `enrichment`
is better as a table. Include `agreement` only if you have Step 3.

---

## Minimum viable paper

If time is short, this carries the whole claim:

1. Step 0 preconditions
2. RISE, CIMT + hypertension, fold 0, 200 images
3. enrichment vs null (`analyze.py`)
4. **seed ceiling** (Step 3)
5. **vessel-masking ablation** (Step 4)

Five experiments. Everything else — IG/occlusion cross-checks, all five folds,
A/V split, the Saudi cohort as external evaluation — is strengthening.

## Files

```
frames.py       canonical frame, FOV, blurred baseline, map normalisation
modelio.py      config+checkpoint -> model; CanonicalScorer (the scoring API)
folddata.py     held-out fold rows -> image paths (fold-matched, out-of-fold)
attribution.py  RISE, occlusion, integrated gradients, random nulls
anatomy.py      RRWNet vessels/AV, disc estimate, zones, ROI masks
metrics.py      enrichment, pointing, IoU, radial, agreement, bootstrap, Holm
check_frame.py  STEP 0  preconditions
run_attribution.py STEP 1  compute maps
analyze.py      STEP 2/3 regional metrics, agreement, ceiling verdict
run_ablation.py STEP 4  causal masking + demographic confound
figures.py      STEP 5  paper figures + disc QC
```

Outputs land in `outputs/xai/{maps,tables,figures}/`.

## Caveats to state in the paper

- The optic disc is **estimated** (brightness + vessel convergence), not
  segmented. Report the QC hit rate. Swapping in a real segmenter means
  replacing `anatomy.estimate_disc` only.
- The canonical frame removes each model's own crop strategy. Report the Step 0
  AUROC deltas.
- Disc diameter is a fixed fraction of FOV diameter (`DISC_DIAM_FRAC = 0.13`),
  not measured per image.
- RISE is stochastic. Report median `split_half_r`.
