# Models

## How the Pipeline Works

Read `efficientnet.py` first — it's the reference implementation.

Each model file contains a class that inherits from `BaseClassifier` and implements 6 things: `forward(x)`, `get_features(x)`, `backbone` property, `head` property, `feature_dim` property, and `preprocessing_config` property. Check EfficientNet for the exact pattern.

The training loop, evaluation, data loading, and logging are model-agnostic. They call `create_model("model_a", variant="...", pretrained=True)` and everything should work.

**What you get for free from BaseClassifier:**

`freeze_backbone()` and `unfreeze_backbone()` operate on whatever your `backbone` property returns. `set_linear_probe_mode()` freezes the backbone AND locks it so that `model.train()` can't silently revert BatchNorm/dropout to training mode. `set_finetune_mode()` unlocks everything. `_validate_output_shape()` catches shape errors immediately.

**PreprocessingConfig** means each model declares its own input size, normalisation stats, and crop strategy. The training pipeline reads this from the model. EfficientNet-B2 gets 260px, ViTs get 224px, and nobody needs to hardcode anything.

**Registration is 3 steps** after writing your file: add to `registry.py`, add your valid kwargs to `_EXPECTED_KWARGS` in `factory.py`, add to `__init__.py` exports.


## The One-File Rule

**If variants only differ in config values (model string, feature dim, input size, scale), they go in one file** with a `VARIANT_CONFIG` dict — like EfficientNet's b0/b2/b4.

**If variants need different construction logic** — different backbone architecture, different weight format, different loading code — **they get separate files.** So no `if` statements in `__init__` to decide how to build the backbone.

This is why RETFound-MAE and RETFound-DINOv2 should be separate files despite both being "RETFound." MAE is ViT-Large/16, DINOv2 is ViT-Large/14 — different patch size, different positional embeddings, different checkpoint format, different loading. Same reasoning for DINOv3 vs DINOv2 (different repo, different loading path) and RETFound-Green vs RETFound-MAE (entirely different architecture).

##  Models

These suggestions are the obvious models for our use case, but if anyone finds additional models to add then it's a simple process as described above. Once we have model files for some of or all of these, both datasets can use them — only the config YAML changes between CIMT and BRSET runs. In reality not all of these will make it to full (50 epochs) training runs, but once we have the training pipeline setup, each dataset can start running short exploratory runs of these to see where we get signal.

### `efficientnet.py` — EfficientNet b0/b2/b4 [THIS IS ALREADY DONE]

CNN baseline. ~9M params (B2). Feature dim 1408. Pretrained on ImageNet. Loaded via `timm`. Variants are config-only (scale differences), so b0/b2/b4 coexist in one file via `VARIANT_CONFIG`. B2 is the primary variant — native 260px input. This is the reference implementation everyone can read before writing files.

### `retfound_mae.py` — RETFound-MAE

Specialist foundation model. ViT-Large/16, ~304M params. Feature dim 1024. Pretrained on 904K retinal images with MAE objective. This is the original model from the Nature 2023 paper. Weights on HuggingFace via `rmaphoh/RETFound`. Key detail: needs `interpolate_pos_embed()` if input size differs from 224. Head is `nn.Linear(1024, 1)`. `get_features()` returns the CLS token or global pool.

### `retfound_dinov2.py` — RETFound-DINOv2

Specialist with modern pretraining. ViT-Large/14, ~304M params. Feature dim 1024. Same retinal data as RETFound-MAE but pretrained with DINOv2 objective. Separate file because it's a different patch size (14 vs 16), different weight format, different loading logic. This is the top performer in Zhou et al. 2025. Different HuggingFace checkpoint from the same repo.

### `dinov2.py` — DINOv2 base/large/giant

Generalist foundation model. No retinal data in pretraining. Loaded via `torch.hub.load('facebookresearch/dinov2', ...)`. Comes headless — you add `nn.Linear(feature_dim, 1)`. Variants are config-only (scale), so base (86M, dim 768), large (304M, dim 1024), and giant (1.1B, dim 1536) coexist via `VARIANT_CONFIG`. Large is the primary variant. Giant needs 3.5x GPU memory — may need reduced batch size. Zhou et al. found giant actually underperformed large on retinal tasks.

### `dinov3.py` — DINOv3-Large

Latest generalist from Meta. ViT-Large, ~304M params. Feature dim 1024. Pretrained on ~1.7B images. Separate file from DINOv2 because it comes from a different repo with different loading. Already integrated into the RETFound codebase as of Sept 2025. Zhou et al. showed it outperforms DINOv2-Giant at the same param count as DINOv2-Large.

### `seresnext.py` — SE-ResNeXt-50

CIMT literature baseline. CNN with squeeze-and-excitation, ~27M params. Feature dim 2048. Loaded via `timm.create_model('seresnext50_32x4d', ...)`. Follows the EfficientNet pattern almost exactly. This is the backbone from the original China-FundusCIMT paper (Dai et al.), though they used a dual-eye bilateral input pipeline that we don't replicate — same backbone, different pipeline, so our results are a lower bound versus their published numbers.

### `convnextv2.py` — ConvNeXt V2-Base

BRSET literature baseline. Modern CNN, ~89M params. Feature dim 1024. Loaded via `timm.create_model('convnextv2_base', ...)`. Same pattern as EfficientNet. This is what the published BRSET paper used (Nakayama et al. 2024), beating their earlier ResNet-50. Also serves as the "modern CNN vs ViT" test.

### `retfound_green.py` — RETFound-Green

Efficient specialist. ViT-Small, ~22M params. Feature dim 384. Pretrained on half the retinal data as RETFound-MAE, DINOv2-initialised, novel token reconstruction objective. Published Nature Comms July 2025 (Engelmann et al.). Separate file because it's an entirely different architecture from the ViT-Large RETFound models. Only 22M params — lighter than EfficientNet-B2. If it holds up on CIMT/BRSET, that's significant for deployment.

### `resnet.py` — ResNet-50

Universal baseline. ~25M params. Feature dim 2048. Loaded via `timm.create_model('resnet50', ...)`. The simplest model file to write — same timm pattern as EfficientNet. Earlier BRSET work used this before switching to ConvNeXt V2. Anchors results to the broader literature.


## Dataset Priorities

- Core for both teams: EfficientNet-B2, RETFound-MAE, DINOv2-Large.
- Strong for both: RETFound-DINOv2, DINOv3-Large.
- CIMT-specific priority: SE-ResNeXt-50 (their literature baseline).
- BRSET-specific priority: ConvNeXt V2-Base (their literature baseline).
- Exploratory: DINOv2-Base, DINOv2-Giant, RETFound-Green, ResNet-50.

**NOTE:** The timm-based models (SE-ResNeXt, ConvNeXt V2, ResNet-50) are essentially the EfficientNet file with different model strings. The ViT-based models (RETFound variants, DINOv2, DINOv3) need more thought around weight loading and headless architectures.


## Checklist

- Class inherits from `BaseClassifier`
- `forward()` returns shape `(B, 1)` and calls `_validate_output_shape`
- `get_features()` returns shape `(B, feature_dim)`
- `backbone` property returns the feature extractor, not the head
- `head` property returns the classification layer
- `feature_dim` matches actual output of `get_features()`
- `preprocessing_config` declares correct input size and normalisation
- Added to `registry.py`
- Added kwargs to `_EXPECTED_KWARGS` in `factory.py`
- Added to `__init__.py` exports
- No hardcoded hyperparameters