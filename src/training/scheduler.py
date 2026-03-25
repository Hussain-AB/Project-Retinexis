"""
scheduler -
Linear warmup then cosine decay to min_lr.
Returns a LambdaLR that preserves layer-wise ratios.
"""

from __future__ import annotations

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# review: must match trainer import
def build_cosine_warmup_scheduler(
    optimiser: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float,
    base_lr: float
) -> LambdaLR:

    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))

        # cosine decay
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs)) # review: decay over remaining epochs, not total

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimiser, lr_lambda)
