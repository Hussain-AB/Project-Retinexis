"""
xai.metrics — regional attribution statistics, cross-model agreement, and stats.

Every regional metric is reported against a smoothness-matched random null,
because a compact ROI scores above chance on smoothed noise alone. An
enrichment of 1.4 is only interesting relative to what the null gets.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

from .frames import normalise_map


# --------------------------------------------------------------------------
# regional metrics
# --------------------------------------------------------------------------
def enrichment(sal_norm: np.ndarray, roi: np.ndarray, fov: np.ndarray) -> float:
    """Attribution mass in the ROI divided by the ROI's area fraction.

    1.0 = uniform. >1 = the model concentrates evidence there. This is the
    headline number: it is comparable across models, across ROIs of different
    size, and across tasks.
    """
    area = float((roi & fov).sum())
    total_area = float(fov.sum())
    if area <= 0 or total_area <= 0:
        return np.nan
    mass = float(sal_norm[roi & fov].sum())
    return mass / (area / total_area)


def mass_fraction(sal_norm: np.ndarray, roi: np.ndarray, fov: np.ndarray) -> float:
    return float(sal_norm[roi & fov].sum())


def pointing_game(sal: np.ndarray, roi: np.ndarray, fov: np.ndarray) -> float:
    """Does the saliency peak land inside the ROI? Binary, per image."""
    masked = np.where(fov, sal, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return float(bool(roi[idx]))


def soft_iou_topk(sal: np.ndarray, roi: np.ndarray, fov: np.ndarray, k: float = 0.10) -> float:
    """IoU between the top-k% most salient FOV pixels and the ROI."""
    vals = sal[fov]
    if vals.size == 0:
        return np.nan
    thresh = np.quantile(vals, 1.0 - k)
    top = (sal >= thresh) & fov
    r = roi & fov
    union = float((top | r).sum())
    if union <= 0:
        return np.nan
    return float((top & r).sum()) / union


def radial_profile(
    sal_norm: np.ndarray,
    fov: np.ndarray,
    disc,
    n_bins: int = 16,
    max_dd: float = 4.0,
) -> np.ndarray:
    """Attribution density vs distance from the disc centre, in disc diameters.

    Density (mass per pixel, normalised by the mean) rather than raw mass, so
    the curve is not dominated by the fact that outer annuli contain more
    pixels. This is the figure that shows the SHAPE of a model's attention
    rather than one scalar.
    """
    H, W = sal_norm.shape
    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((xx - disc.cx) ** 2 + (yy - disc.cy) ** 2) / disc.dd

    edges = np.linspace(0.0, max_dd, n_bins + 1)
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = fov & (dist >= edges[i]) & (dist < edges[i + 1])
        n = int(sel.sum())
        if n > 0:
            out[i] = float(sal_norm[sel].sum()) / n
    mean_density = 1.0 / float(fov.sum())
    return out / mean_density


# --------------------------------------------------------------------------
# cross-model agreement
# --------------------------------------------------------------------------
def spearman_agreement(a: np.ndarray, b: np.ndarray, fov: np.ndarray, stride: int = 2) -> float:
    """Rank correlation of two saliency maps over FOV pixels.

    Sensitive to the whole distribution, not just the peak -- the complement of
    soft_iou_topk, which only sees the top decile. Report both: they disagree
    when two models rank the same peak region but differ everywhere else.
    """
    sub = np.zeros_like(fov)
    sub[::stride, ::stride] = True
    sel = fov & sub
    if sel.sum() < 50:
        sel = fov
    r, _ = stats.spearmanr(a[sel], b[sel])
    return float(r)


def iou_agreement(a: np.ndarray, b: np.ndarray, fov: np.ndarray, k: float = 0.10) -> float:
    va, vb = a[fov], b[fov]
    ta = np.quantile(va, 1.0 - k)
    tb = np.quantile(vb, 1.0 - k)
    ma, mb = (a >= ta) & fov, (b >= tb) & fov
    union = float((ma | mb).sum())
    return float((ma & mb).sum()) / union if union > 0 else np.nan


def centroid_shift(a: np.ndarray, b: np.ndarray, fov: np.ndarray, dd: float) -> float:
    """Distance between the two maps' centres of positive mass, in disc diameters.

    A physically interpretable companion to the correlation metrics: 'the two
    models' attention centroids sit 1.8 disc diameters apart' reads better in a
    paper than 'rho = 0.31'.
    """
    na, nb = normalise_map(a, fov), normalise_map(b, fov)
    H, W = a.shape
    yy, xx = np.mgrid[:H, :W]
    ca = (float((na * xx).sum()), float((na * yy).sum()))
    cb = (float((nb * xx).sum()), float((nb * yy).sum()))
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]) / dd)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap over images. Returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float, float]:
    """Paired difference a - b over the SAME images, with a two-sided p.

    Model comparisons here are paired by construction (every model is scored on
    the same held-out images), so an unpaired test would throw away most of the
    power and overstate the uncertainty.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), float(min(p, 1.0))


def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values.

    The ROI x model x task grid is ~130 comparisons. Uncorrected, a handful
    will clear 0.05 by chance and a reviewer will say so.
    """
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    idx = np.argsort(p[ok])
    vals = p[ok][idx]
    m = vals.size
    adj = np.maximum.accumulate((m - np.arange(m)) * vals)
    res = np.empty(m)
    res[idx] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def delong_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def bootstrap_auc_delta(
    y_true: np.ndarray, base: np.ndarray, ablated: np.ndarray,
    n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap of AUROC(base) - AUROC(ablated), resampling images.

    This is the causal counterpart to enrichment: how much predictive signal
    actually lived in the region that was masked.
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    n = y_true.size
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        deltas.append(roc_auc_score(yt, base[idx]) - roc_auc_score(yt, ablated[idx]))
    deltas = np.asarray(deltas)
    point = roc_auc_score(y_true, base) - roc_auc_score(y_true, ablated)
    if deltas.size == 0:
        return float(point), np.nan, np.nan
    return float(point), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
