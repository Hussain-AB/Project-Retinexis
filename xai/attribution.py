"""
xai.attribution — architecture-agnostic attribution.

Grad-CAM is NOT used here, deliberately. On this model set it would compare
four ViTs (which need a reshape transform and an arbitrary block choice) with
two CNNs (which do not), so any measured difference would be confounded by the
method x architecture interaction rather than by what the models use. All three
methods below treat the network as a black box or use only input gradients, so
every model goes through byte-identical code.

    rise()      primary   -- interventional, black box, resolution set by us
    occlusion() control   -- deterministic, coarse, no sampling noise
    integrated_gradients() secondary -- ~50x cheaper than RISE, gradient-based

All three return saliency in CANONICAL FRAME coordinates (see frames.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .frames import blurred_baseline


# --------------------------------------------------------------------------
# RISE
# --------------------------------------------------------------------------
@dataclass
class RiseResult:
    saliency: np.ndarray          # (H, W) signed, canonical frame
    split_half_r: float           # convergence diagnostic -- report this
    n_masks: int
    base_logit: float
    mean_masked_logit: float
    params: dict = field(default_factory=dict)


def _rise_mask_chunk(
    n: int, H: int, W: int, s: int, p: float, rng: np.random.Generator
) -> np.ndarray:
    """Smooth, non-grid-aligned binary masks (Petsiuk et al., BMVC 2018).

    Low-res Bernoulli grid -> bilinear upsample -> random shift-crop. The shift
    is what stops the masks from being blocky and grid-locked, which otherwise
    imprints the grid onto every saliency map.
    """
    cell_h = int(np.ceil(H / s))
    cell_w = int(np.ceil(W / s))
    grid = (rng.random((n, 1, s, s)) < p).astype(np.float32)
    up = F.interpolate(
        torch.from_numpy(grid),
        size=((s + 1) * cell_h, (s + 1) * cell_w),
        mode="bilinear",
        align_corners=False,
    ).numpy()[:, 0]

    out = np.empty((n, H, W), dtype=np.float32)
    ys = rng.integers(0, cell_h, size=n)
    xs = rng.integers(0, cell_w, size=n)
    for i in range(n):
        out[i] = up[i, ys[i]:ys[i] + H, xs[i]:xs[i] + W]
    return out


def rise(
    scorer,
    image: np.ndarray,
    n_masks: int = 2000,
    s: int = 7,
    p: float = 0.5,
    batch_size: int = 32,
    seed: int = 0,
    baseline: np.ndarray | None = None,
) -> RiseResult:
    """Randomized Input Sampling for Explanation.

    Estimates cov(score, mask_visible) per pixel, i.e. how much the model's
    logit rises when a region is visible rather than blurred away. The
    covariance form (rather than the raw score-weighted mean of the original
    paper) removes the large additive offset mean_score/p and makes maps from
    models with different logit scales directly comparable.

    Returns a signed map: positive = evidence for the positive class.
    """
    H, W = image.shape[:2]
    img = image.astype(np.float32)
    base = (blurred_baseline(image) if baseline is None else baseline).astype(np.float32)
    rng = np.random.default_rng(seed)

    acc = [
        dict(sM=np.zeros((H, W), np.float64), M=np.zeros((H, W), np.float64), s=0.0, n=0)
        for _ in range(2)
    ]

    done = 0
    half = 0
    while done < n_masks:
        take = min(batch_size, n_masks - done)
        masks = _rise_mask_chunk(take, H, W, s, p, rng)
        batch = img[None] * masks[..., None] + base[None] * (1.0 - masks[..., None])
        scores = scorer.score_batch(batch).astype(np.float64)

        a = acc[half]
        a["sM"] += np.einsum("n,nhw->hw", scores, masks.astype(np.float64))
        a["M"] += masks.astype(np.float64).sum(0)
        a["s"] += scores.sum()
        a["n"] += take

        done += take
        half ^= 1

    def finish(a):
        n = a["n"]
        e_sM = a["sM"] / n
        e_M = a["M"] / n
        e_s = a["s"] / n
        var_M = np.clip(e_M * (1.0 - e_M), 1e-3, None)
        return (e_sM - e_s * e_M) / var_M, e_s

    s0, m0 = finish(acc[0])
    s1, m1 = finish(acc[1])
    saliency = 0.5 * (s0 + s1)

    r = float(np.corrcoef(s0.ravel(), s1.ravel())[0, 1])
    base_logit = float(scorer.score_batch(img[None])[0])

    return RiseResult(
        saliency=saliency.astype(np.float32),
        split_half_r=r,
        n_masks=n_masks,
        base_logit=base_logit,
        mean_masked_logit=float(0.5 * (m0 + m1)),
        params=dict(method="rise", s=s, p=p, n_masks=n_masks, seed=seed),
    )


# --------------------------------------------------------------------------
# Occlusion
# --------------------------------------------------------------------------
def occlusion(
    scorer,
    image: np.ndarray,
    patch: int = 48,
    stride: int = 24,
    batch_size: int = 32,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic sliding-window occlusion (Zeiler & Fergus, 2014).

    No sampling noise, so it is the control that tells you whether a RISE
    result is real or a mask-sampling artifact. Coarse by construction.
    Returns (H, W): logit drop when the covering patch is blurred away.
    """
    H, W = image.shape[:2]
    img = image.astype(np.float32)
    base = (blurred_baseline(image) if baseline is None else baseline).astype(np.float32)

    ref = float(scorer.score_batch(img[None])[0])
    heat = np.zeros((H, W), np.float64)
    count = np.zeros((H, W), np.float64)

    coords = [(y, x)
              for y in range(0, max(H - patch, 0) + 1, stride)
              for x in range(0, max(W - patch, 0) + 1, stride)]

    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = np.repeat(img[None], len(chunk), axis=0)
        for j, (y, x) in enumerate(chunk):
            batch[j, y:y + patch, x:x + patch] = base[y:y + patch, x:x + patch]
        scores = scorer.score_batch(batch)
        for j, (y, x) in enumerate(chunk):
            heat[y:y + patch, x:x + patch] += ref - float(scores[j])
            count[y:y + patch, x:x + patch] += 1

    return (heat / np.clip(count, 1, None)).astype(np.float32)


# --------------------------------------------------------------------------
# Integrated Gradients (+ optional SmoothGrad)
# --------------------------------------------------------------------------
def integrated_gradients(
    scorer,
    image: np.ndarray,
    steps: int = 64,
    smooth_samples: int = 0,
    noise_frac: float = 0.10,
    batch_size: int = 8,
    seed: int = 0,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """IG with a blurred-fundus baseline, computed in canonical coordinates.

    The resize into model resolution is inside the autograd graph, so gradients
    come back at canonical resolution directly -- no upsampling of a
    low-resolution map, and no per-architecture feature-map stride to reconcile.

    A BLACK baseline is wrong for fundus images (it is in-distribution -- it
    looks like the FOV surround), so the blurred image is used, matching RISE
    and occlusion.

    smooth_samples > 0 enables SmoothGrad-IG: average IG over noisy copies.
    """
    device = scorer.device
    H, W = image.shape[:2]
    rng = np.random.default_rng(seed)

    x_img = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
    x_base = torch.from_numpy(
        (blurred_baseline(image) if baseline is None else baseline).astype(np.float32) / 255.0
    ).permute(2, 0, 1).to(device)

    n_rounds = max(1, smooth_samples)
    total = torch.zeros_like(x_img)

    for r in range(n_rounds):
        target = x_img
        if smooth_samples > 0:
            noise = torch.from_numpy(
                rng.normal(0.0, noise_frac, size=tuple(x_img.shape)).astype(np.float32)
            ).to(device)
            target = (x_img + noise).clamp(0, 1)

        delta = target - x_base
        grads = torch.zeros_like(x_img)
        alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps

        for i in range(0, steps, batch_size):
            a = alphas[i:i + batch_size].view(-1, 1, 1, 1)
            pts = (x_base.unsqueeze(0) + a * delta.unsqueeze(0)).requires_grad_(True)
            out = scorer.logits_with_grad(pts).sum()
            g, = torch.autograd.grad(out, pts)
            grads += g.sum(0)

        total += delta * grads / steps

    ig = (total / n_rounds).sum(0)  # sum over colour channels
    return ig.detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------
# nulls
# --------------------------------------------------------------------------
def random_smooth_map(H: int, W: int, sigma: float, seed: int = 0) -> np.ndarray:
    """Smoothness-matched random map. The null every metric is scored against.

    An enrichment number means nothing without this: heavily smoothed noise
    already scores above 1.0 on a compact ROI purely from spatial correlation.
    """
    import cv2
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(H, W)).astype(np.float32)
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(noise, (k, k), sigmaX=sigma, sigmaY=sigma)


def map_smoothness_sigma(sal: np.ndarray) -> float:
    """Estimate an equivalent Gaussian sigma from a map's spatial autocorrelation.

    Used to match the random null to each method's own smoothness rather than
    picking a sigma by hand.
    """
    s = sal - sal.mean()
    denom = float((s * s).mean())
    if denom <= 0:
        return 1.0
    shifted = np.roll(s, 1, axis=0)
    rho = float(np.clip((s * shifted).mean() / denom, 1e-6, 0.999999))
    return float(np.sqrt(-1.0 / (2.0 * np.log(rho))))
