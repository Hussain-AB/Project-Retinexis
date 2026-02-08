# CVD-Risk Models — Architecture Rundown

This covers everything that will live in the CVD Risk/src/models/ directory.

**Note:** As a general guiding narrative for the models we're training for CVD risk, there will be an interesting comparison and potentially genuinely useful finding if a general purpose ViT outperforms a domain specific ViT, which is why the Meta model is included here.

**Caveat:** I'm in the process of writing these files so everything here is an overview/roadmap at this point and nothing is set in stone. Parts of this might/likely-will have to be adjusted post-testing.


## File-by-File Explainers

### base.py

*The contract that every model signs.*

`BaseClassifier` is the abstract base class. It defines six things a model must provide: `forward()`, `backbone`, `head`, `get_features()`, `feature_dim`, and `preprocessing_config`. If a new model doesn't implement all six, Python raises an error at instantiation — not at some random point during training. This is an evolution from our previous pipeline whereby we had seperate training scripts for transfer learning and our from scratch CNN.

### efficientnet.py

*The CNN baseline.*

EfficientNet-B2 is loaded from timm with `num_classes=0`, which strips the original classifier head and returns pooled features. We attach our own single linear layer. The model runs at its **native 260px resolution** because a baseline should represent the standard approach — downscaling it to 224px to match the ViTs would disadvantage unfairly.

### dinov2.py

*The general-purpose Vision Transformer.*

DINOv2 is loaded via **timm, not torch.hub**. This is a deliberate choice. The `torch.hub.load('facebookresearch/dinov2', ...)` path hits GitHub at runtime, breaks when the repo changes, and returns different output formats (dicts, 2D tensors, 3D tensors) depending on version. The timm path is version-pinned, works offline, and with `global_pool='token'` always returns a clean `(B, dim)` CLS token.

KEY PRINCIPLE: Two variants are used in the comparison: **ViT-B** (86M parameters, tests architecture effect against EfficientNet) and **ViT-L** (307M parameters, capacity-matched to RETFound for the headline domain comparison).

### retfound.py

*Our domain-specific ViT foundation model.*

RETFound is a ViT-Large pretrained on 1.6 million retinal fundus images using Masked Autoencoding. Unlike the other models, it cannot be loaded with a single `timm.create_model()` call — we build the ViT-Large architecture ourselves via timm's `VisionTransformer` class and then load the pretrained weights manually.

The **checkpoint loading pipeline** will be the most complex part of this file. MAE checkpoints are messy: they contain decoder weights we don't need, may have DDP prefixes from distributed training, and store position embeddings for a fixed input size. The pipeline handles each of these steps in order, and at every step validates that the result makes sense. I don't know how messy this will be until I start writing code and step testing. 

For validation will use an **allowlist approach**, not a count threshold. We know exactly which keys should be missing (our classification head replaces the original) and which should be present (all encoder layers). Anything outside this policy raises an error immediately.

### registry.py

*The single source of truth for which models exist.*

`MODEL_REGISTRY` is a dictionary mapping architecture names to classes. The factory reads from it, sweeps enumerate it, tests parametrise over it. Adding a new architecture means adding one entry here and one entry in the factory's `_EXPECTED_KWARGS`. Again this is part of my effort to make our pipeline more streamlined than last time. 

### factory.py

*The single entry point for training scripts.*

`create_model(architecture, **kwargs)` will look up the class in the registry, validate that the keyword arguments make sense for that architecture, and return an instance.
