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

def optimiser(
    model: nn.Module,
    base_lr: float, 
    lr_decay_factor: float,
    weight_decay: float,
    metadata_head: nn.Module | None = None
) -> list[dict]:
    head_params = []
    if metadata_head is not None:
        head_params.extend(list(metadata_head.parameters()))
    else:
        head_params.extend(list(model.head.paramaters()))
    
    backbone_params = list(model.backbone.parameters())

    # backbone will learn slower than the head
    backbone_lr = base_lr * lr_decay_factor

    # apply split from helper function
    head_decay, head_no_decay = split_decay_or_no_decay(head_params)
    backbone_decay, backbone_no_decay = split_decay_or_no_decay(backbone_params)

    return [
        {"params": head_decay, "lr":base_lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0},
        {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
    ]

    
    

