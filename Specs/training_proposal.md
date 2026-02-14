# Training Strategy (Proposal)

From: MLOps Lead for CIMT team
To: MLOps Lead for BRSET team (and for the understanding/discussion of both teams)

## Overview of What We’re Optimising For

We have two concerns pulling in the same direction but with different priorities.

**Coursework deliverable:** Find the best-performing models for CIMT and BRSET respectively, with well-justified methodology and clear documentation of our decision-making throughout. Given we don't have an actual product to show, professor's made it clear that we'll be graded on the models themselves and our approach, not on producing a research paper.

**SDM deliverable / future research paper(s):** Build a scientific narrative into our methodology from the start — specifically the generalist-vs-specialist comparison with cross-population validation — so that if we are to submit to a conference afterward, the work is already structured for it. We’re not going to do this work twice, so the experiment design should serve both purposes.

I don't see these as mutually exclusive concerns. Choosing models systematically rather than arbitrarily is better engineering and better science. Running controlled comparisons (same architecture, different pretraining) produces both the best-justified model selection for coursework and the cleanest publishable results. The key is that when the two concerns conflict, performance and justifiability win, as these priorities satisfy tue needs of the coursework and that of the client.

## The Problem

We have 9 models planned across the shared pipeline, 2 datasets, and roughly 200 GPU-hours (realistically; could technically do more time-wise but it wouldn't look great to train everything fully without pruning). So we need a time-efficient way to gather signal before committing compute to the expensive full training runs.

So to mitigate this problem, I would propose a two-phased approach: a low-cost exploratory phase that screens all models via linear probing, followed by full training on a prioritised subset. The teams can then make the final call on which models advance after seeing the exploratory results.

## The Model Pool

Once all model files are written and passing `test_models.py` [I HAVEN'T WRITTEN THIS TEST YET (but will derive largely from  test_efficientnet.py when it's written)], we’ll hopefully have 9 registered models to train.

**Tier 1 — Core comparisons:** RETFound-MAE (retinal specialist, 304M), RETFound-DINOv2 (specialist + DINOv2 objective, 304M), DINOv2-Large (generalist, 304M). These are the capacity-matched ViT-Large models that let us isolate the effect of pretraining data and objective. They’re the strongest candidates for top performance on our specific tasks based on the literature, and running all three gives us a controlled comparison that will document why we chose our final model.

**Tier 2 — Strong candidates:** EfficientNet-B2 (CNN, 9M), DINOv3-Large (latest generalist, 304M), SE-ResNeXt-50 (CIMT literature baseline, 27M), ConvNeXt V2-Base (BRSET literature baseline, 89M). Each serves a purpose: the CNN tests whether ViTs are even necessary, DINOv3 checks whether we’re missing a better generalist ViT model, and the literature baselines let us benchmark against published work on our datasets.

**Tier 3 — Exploratory:** RETFound-Green (efficient specialist, 22M), ResNet-50 (universal anchor, 25M). Worth screening but lowest priority for full runs.

## Phase 1: Exploratory Phase

### What it is

Run Phase 1 only (linear probe, frozen backbone, head-only training) on all 9 models across all 5 CV folds on both CIMT and BRSET. ~15 minutes per model per dataset, roughly 2 GPU-hours total.

### What it tells us

A frozen-feature AUROC well above 0.50 means the pretrained representation already encodes task-relevant signal. This is standard evaluation in the foundation model literature (Zhou et al. 2025, Hou et al. 2025) and gives us an early picture of which models are likely to perform well after full fine-tuning.

### What it doesn’t tell us

Linear probing is less reliable for comparing across pretraining objectives. MAE-pretrained models (including RETFound-MAE) are documented to produce features that look mediocre under linear probing but recover after fine-tuning — the original MAE paper showed this, and Zhou et al.’s data confirms the gap is largest for MAE models. A strict Phase 1 cutoff could falsely penalise one of our strongest candidates.

### How we can use Phase 1 results

**Phase 1 is informed prioritisation, not a hard filter.** With 9 models we’re deciding whether models 6–9 justify the extra compute, we're not running a model arena. Tier 1 models advance regardless because they’re our best-performing candidates and the comparisons between them justify whichever one we ultimately select.

Phase 1 is most useful for catching broken implementations (AUROC ~0.50 before we waste 17 hours on Phase 2), ordering the queue when time is tight, and producing early results that demonstrate our methodology.

### Deliverables

For each model we fill in a row of the Phase 1 results table (AUROC mean ± std, 5 folds, both datasets) plus a one-paragraph interpretation. We review the table, and decide together what advances.

## Phase 2: Model Selection

### Criteria

**Performance signal.** Models with strong Phase 1 are safer bets for top performance. Weak Phase 1 models may recover after fine-tuning (especially MAE), but that’s a deliberate gamble.

**Justifiability.** For the coursework deliverable, we need to show why we chose our final model. Running controlled comparisons (same architecture, different pretraining; same pretraining, different architecture) gives us that documentation. A CNN baseline against the foundation models demonstrates whether the extra complexity (and far greater compute burden) was warranted. The literature baselines show how we compare to published work.

**Compute budget.** Each ViT-Large costs ~17.5 GPU-hours for Phase 2 across 5 folds. CNNs cost ~4–8 hours. Data efficiency experiments multiply by 4. *By compute budget I only mean time not cost, as we effectively have unlimted access to an A100 with 80Gb VRAM, so our main constraint is going to be time, and for these ViTs that's going to have to be a serious model selection decision point as multiple ~17hr training runs will stall our progress if we don't make smart model choices going into phase 2. Compute calculations are approximate based on the dataset size and the original RETFound paper's documented training times for different configurations and dataset sizes.

### Selection scenarios

|N|Models                                           |Phase 2 cost|
|-|-------------------------------------------------|------------|
|3|RETFound-DINOv2, DINOv2-Large, + 1 CNN           |~39 hrs     |
|4|+ DINOv3-Large or RETFound-MAE                   |~56 hrs     |
|5|+ literature baseline (SE-ResNeXt or ConvNeXt V2)|~65 hrs     |
|6|+ additional model from Tier 2/3                 |~82 hrs     |

The CNN slot is open — each team fills it based on Phase 1 results. EfficientNet-B2 is the reference implementation, not automatically the best CNN for either dataset.

### Independent decisions per team

CIMT and BRSET don’t have to select the same subset. A model might earn its spot on one dataset but not the other. We each commit to Tier 1, then independently pick 1–2 additions based on Phase 1 numbers and what makes sense for our dataset.

### Cross-Validation Strategy

We use 5-fold cross-validation. The full dataset splits into 5 non-overlapping groups. Each round, 4 groups train the model and the 5th is held out for evaluation. This repeats 5 times so every image gets tested exactly once, and we report results as mean ± std across the 5 folds.

Three properties matter for our splits. They're stratified, meaning each fold preserves the overall class ratio (important for CIMT where ~70% of samples are positive). As discussed previously, they're patient-grouped, so both eyes from the same patient always land in the same fold. And they're fixed to file, so every model trains on identical folds so that performance comparisons are paired and fair.

The external evaluation cohort (Saudi or Bangladeshi, whatever we get given) is completely separate from cross-validation. After a model finishes training on a given fold, we run inference on the external set with the saved checkpoint. This gives us 5 external AUROC values (one per fold's model), whose mean ± std captures how consistently the model generalises to an unseen population.


## What Full Training Would Look Like

**Phase 1**: Linear probe, 5 epochs, frozen backbone.

**Phase 2:** Full fine-tune, up to 50 epochs, early stopping (patience 7, val AUROC). Layer-wise LR decay (0.65/layer), cosine schedule with 5-epoch warmup. Weighted BCE for class imbalance?

**Internal evaluation:** Held-out fold. AUROC, AUPRC, ECE, sensitivity/specificity at operating points, F1.

**External evaluation:** Inference on an out-of-distribution cohort (Saudi or Bangladeshi patient data, TBD). Same metrics, no retraining. The delta between internal and external performance per model measures robustness under cohort shift.

**Data efficiency (optional per model):** Repeat Phase 2 at 10%/30%/50%/100% of training data. Suggest limiting to top 3 models.

## Open Questions

1. **Data efficiency: all finalists or top 3?**
2. **External cohort timing.** If it arrives during exploration, do we factor it into selection?
3. **Cross-dataset completeness.** Run a CIMT-strong model on BRSET for the contrast, or save the hours?
4. **DINOv2 scale variants.** Worth the runs for a scaling curve, or cite Zhou et al.?
5. **Statistical testing.** Bootstrap CIs from prediction-level data alongside DeLong on fold-level AUROCs?
