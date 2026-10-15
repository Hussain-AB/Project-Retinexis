"""
xai.gradcam — Grad-CAM for the six-model roster, in canonical-frame coordinates.

Added for the ISBI paper as a SECOND gradient-based method next to Integrated
Gradients. attribution.py explains why Grad-CAM was originally left out: on a
ViT it needs a layer choice and a token->grid reshape that a CNN does not. That
caveat still holds, so every Grad-CAM finding is reported only if Integrated
Gradients (identical code for every architecture) shows the same pattern.

Layer choice, fixed in advance and printed in every output file:

  CNN (ConvNeXtV2-B, EfficientNet-B2)
      the LAST module, in forward-execution order, that outputs a spatial
      feature map (B, C, h, w) with h, w >= 2. That is the final stage just
      before global pooling -- the standard Grad-CAM target.

  ViT (RETFound-DINOv2, RETFound-MAE, DINOv2-L, DINOv3-L)
      the input-side LayerNorm (`norm1`) of the LAST transformer block. Not the
      block's output: when the head reads only the class token, gradients with
      respect to the last block's patch-token OUTPUTS are exactly zero, because
      nothing mixes tokens after that block. norm1 of the last block sits
      before its attention, so patch tokens still reach the class token.
      Tokens (B, N, C) are split into N - g*g leading special tokens (class,
      registers) and a g x g patch grid.

Either choice can be overridden with --gradcam-layer, but the default is what
the paper should report.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CamLayer:
    name: str
    kind: str            # "cnn" or "vit"
    grid: int            # spatial size of the map before upsampling
    n_special: int       # ViT only: leading non-patch tokens
    channels: int


def _first_tensor(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return None


def _trace_shapes(model, x) -> list[tuple[str, tuple]]:
    """(module name, output shape) for every module, in execution order."""
    seen, hooks = [], []
    for name, mod in model.named_modules():
        if not name:
            continue
        def hook(m, i, o, name=name):
            t = _first_tensor(o)
            if t is not None:
                seen.append((name, tuple(t.shape)))
        hooks.append(mod.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
    return seen


def discover_layer(model, x, family: str, override: str | None = None) -> CamLayer:
    """Pick the Grad-CAM target. family is 'cnn-*' or 'vit-*' (xai.paths.MODEL_FAMILY)."""
    trace = _trace_shapes(model, x)
    if not trace:
        raise RuntimeError("no module produced a tensor during the trace forward")
    kind = "cnn" if family.startswith("cnn") else "vit"

    def describe(name, shape):
        if len(shape) == 4:
            return CamLayer(name, "cnn", shape[-1], 0, shape[1]) if shape[-1] == shape[-2] else None
        if len(shape) == 3:
            n = shape[1]
            g = math.isqrt(n)
            k = n - g * g
            if g >= 4 and k <= 16:
                return CamLayer(name, "vit", g, k, shape[2])
        return None

    if override:
        match = [(n, s) for n, s in trace if n == override]
        if not match:
            raise ValueError(f"--gradcam-layer {override!r} not found; "
                             f"last modules: {[n for n, _ in trace[-15:]]}")
        layer = describe(*match[-1])
        if layer is None:
            raise ValueError(f"{override} outputs {match[-1][1]}, not a usable feature map")
        return layer

    if kind == "cnn":
        cands = [(n, s) for n, s in trace if len(s) == 4 and s[-1] >= 2 and s[-2] >= 2 and s[1] >= 64]
        if not cands:
            raise RuntimeError("CNN: no 4-D spatial feature map found")
        return describe(*cands[-1])

    cands = [(n, s) for n, s in trace if re.search(r"(^|\.)norm1$", n) and len(s) == 3]
    cands = [c for c in cands if describe(*c) is not None]
    if not cands:
        raise RuntimeError("ViT: no '<block>.norm1' with a (B, N, C) token output found; "
                           f"last modules: {[n for n, _ in trace[-15:]]}. Pass --gradcam-layer.")
    return describe(*cands[-1])


class GradCAM:
    """Batched Grad-CAM on a CanonicalScorer.

    Inputs are canonical-frame uint8 images (B, H, W, 3); output maps are
    returned at the same H x W (bilinear upsampling of the h x w CAM), plus the
    fp32 logits of the same forward pass.
    """

    def __init__(self, scorer, family: str, probe: np.ndarray, override: str | None = None):
        self.scorer = scorer
        x = self._to_input(probe[None])
        self.layer = discover_layer(scorer.model, scorer._prepare(x), family, override)
        self.module = dict(scorer.model.named_modules())[self.layer.name]
        self._act = None
        self.module.register_forward_hook(self._keep)

    def _keep(self, m, i, o):
        self._act = _first_tensor(o)

    def _to_input(self, images: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.scorer.device)
        return x.permute(0, 3, 1, 2).float().div_(255.0)

    def __call__(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H, W = images.shape[1:3]
        x = self._to_input(images).requires_grad_(True)   # builds the graph; weights stay frozen
        logits = self.scorer.logits_with_grad(x)            # fp32, (B,)
        act = self._act
        grad, = torch.autograd.grad(logits.sum(), act)      # samples are independent in eval mode
        L = self.layer
        if L.kind == "cnn":
            w = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * act).sum(1))                  # (B, h, w)
        else:
            a = act[:, L.n_special:, :]
            g = grad[:, L.n_special:, :]
            w = g.mean(dim=1, keepdim=True)                 # (B, 1, C)
            cam = F.relu((w * a).sum(-1)).view(-1, L.grid, L.grid)
        cam = F.interpolate(cam[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        return cam.detach().float().cpu().numpy(), logits.detach().float().cpu().numpy()
