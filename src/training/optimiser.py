"""
training.optimiser -
Assigns decaying LR per backbone layer for Phase 2 fine-tuning.
 Splits params into decay/no-decay groups.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW



# helper function: remove weight decay from tenspr
def split_decay_or_no_decay(params: list[nn.Parameter]) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        decay = []
        no_decay = []
        for p in params:
            if not p.requires_grad:
                continue
            if p.ndim <= 1:
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay

# review: must match trainer import
def build_layer_wise_param_groups(
    model: nn.Module,
    base_lr: float,
    lr_decay_factor: float,
    weight_decay: float,
    metadata_head: nn.Module | None = None,
) -> list[dict]:
    # review: true layer-wise decay — lr_i = base_lr * (factor ^ (N - i))
    # Head gets full base_lr, each backbone layer gets progressively less.

    param_groups = []

    # Head params (or metadata_head) — full LR
    if metadata_head is not None:
        head_params = list(metadata_head.parameters())
    else:
        head_params = list(model.head.parameters())

    head_decay, head_no_decay = split_decay_or_no_decay(head_params)
    param_groups.append({"params": head_decay, "lr": base_lr, "weight_decay": weight_decay})
    param_groups.append({"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0})

    # Backbone — group by layer depth
    # Most timm/torch ViTs have backbone.blocks.0, backbone.blocks.1, etc.
    # CNNs have backbone.layer1, backbone.layer2, etc.
    # Fall back to single group if layer structure isn't detectable.
    backbone = model.backbone
    layer_params = {}

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        # Extract layer index from parameter name
        # Handles: blocks.0.xxx, blocks.11.xxx, layer1.xxx, layer2.xxx
        parts = name.split(".")
        layer_id = None
        for part in parts:
            if part.isdigit():
                layer_id = int(part)
                break
            if part.startswith("layer") and part[-1].isdigit():
                layer_id = int(part.replace("layer", ""))
                break
        if layer_id is None:
            layer_id = 0  # embeddings, cls_token, etc. — earliest layer

        if layer_id not in layer_params:
            layer_params[layer_id] = []
        layer_params[layer_id].append(param)

    num_layers = max(layer_params.keys()) + 1 if layer_params else 1

    for layer_id, params in sorted(layer_params.items()):
        layer_lr = base_lr * (lr_decay_factor ** (num_layers - layer_id))
        decay, no_decay = split_decay_or_no_decay(params)
        if decay:
            param_groups.append({"params": decay, "lr": layer_lr, "weight_decay": weight_decay})
        if no_decay:
            param_groups.append({"params": no_decay, "lr": layer_lr, "weight_decay": 0.0})

    return param_groups
