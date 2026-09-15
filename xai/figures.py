"""
STEP 5 -- the paper figures, plus the disc-estimation QC panel.

    python -m xai.figures all --task cimt --method rise
    python -m xai.figures disc-qc --task cimt --fold 0 --vessel-dir data/cimt/vessel_masks

Figure roles (ISBI is 4 pages -- you have room for two, maybe three):

  radial    THE figure. Attribution density vs distance from the disc, one
            curve per model. Shows the SHAPE of each model's attention rather
            than collapsing it to a scalar, and makes convergence/divergence
            legible at a glance.
  ablation  the causal panel: AUROC drop per masked region. Pair with radial.
  enrichment  model x ROI matrix. Dense; better as a table in a 4-page paper.
  agreement   cross-model similarity with the same-architecture ceiling drawn
              as a reference line. Only include if you HAVE the ceiling runs.

Palette is fixed-order and CVD-validated (six checks, light mode). Series are
also distinguished by marker and dash, so identity never rests on color alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from .paths import PROJECT_ROOT

# Fixed categorical order -- never cycled, never reassigned when a model is
# dropped from a plot. Validated: lightness band, chroma floor, CVD separation,
# normal-vision floor, contrast vs surface.
MODEL_ORDER = ["retfound_dinov2", "retfound_mae", "dinov2_large",
               "dinov3_large", "convnextv2", "efficientnet_b2"]
PALETTE = dict(zip(MODEL_ORDER,
                   ["#3468C0", "#D94F5C", "#00897B", "#B8860B", "#9C51B6", "#C2185B"]))
MARKERS = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v", "P"]))
DASHES = dict(zip(MODEL_ORDER,
                  [(None, None), (5, 1.5), (1.5, 1.5), (7, 2, 1.5, 2), (4, 1, 1, 1), (2, 2)]))
LABEL = {
    "retfound_dinov2": "RETFound-DINOv2", "retfound_mae": "RETFound-MAE",
    "dinov2_large": "DINOv2-L", "dinov3_large": "DINOv3-L",
    "convnextv2": "ConvNeXtV2-B", "efficientnet_b2": "EfficientNet-B2",
}
INK, MUTED, SURFACE = "#1a1a1a", "#6b7280", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e5e7eb", "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

TABLES = PROJECT_ROOT / "outputs" / "xai" / "tables"
FIGS = PROJECT_ROOT / "outputs" / "xai" / "figures"


def _order(models) -> list[str]:
    return [m for m in MODEL_ORDER if m in set(models)]


def fig_radial(task: str, method: str) -> Path:
    df = pd.read_csv(TABLES / f"radial_{task}_{method}.csv")
    df = df[df["run"] == "primary"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.grid(axis="y", zorder=0)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":", zorder=1)

    for m in _order(df["model"].unique()):
        g = df[df["model"] == m].groupby("bin").agg(
            x=("dd_lo", "mean"), y=("density", "mean")).reset_index()
        ax.plot(g["x"], g["y"], color=PALETTE[m], lw=2.0, marker=MARKERS[m],
                markersize=3.2, dashes=DASHES[m], label=LABEL[m], zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=0.5)

    ax.set_xlabel("distance from optic disc centre (disc diameters)")
    ax.set_ylabel("attribution density\n(x uniform)")
    ax.set_title(f"{task.upper()} — where each model looks", loc="left", color=INK)
    # legend below the axes: the curves peak near the disc and fill the upper
    # left, so any in-axes placement collides with the data
    ax.legend(frameon=False, ncol=3, labelcolor=INK, handlelength=2.4,
              loc="upper center", bbox_to_anchor=(0.5, -0.30), columnspacing=1.1,
              handletextpad=0.5)
    out = FIGS / f"radial_{task}_{method}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def fig_enrichment(task: str, method: str) -> Path:
    df = pd.read_csv(TABLES / f"summary_{task}_{method}.csv")
    rois = ["disc", "zone_b", "artery", "vein", "vessel", "macula", "background", "periphery"]
    models = _order(df["model"].unique())
    mat = (df.pivot(index="model", columns="roi", values="enrichment")
             .reindex(index=models, columns=rois))
    sig = (df.pivot(index="model", columns="roi", values="p_holm")
             .reindex(index=models, columns=rois))

    # Diverging: enrichment has a true neutral midpoint at 1.0 (uniform
    # attribution). Two hues, neutral gray at the middle -- never a rainbow.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "enr", ["#3468C0", "#f3f4f6", "#C2185B"])
    vmax = float(np.nanmax(np.abs(mat.values - 1.0))) + 1.0

    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    im = ax.imshow(mat.values, cmap=cmap,
                   norm=TwoSlopeNorm(vcenter=1.0, vmin=2.0 - vmax, vmax=vmax))
    ax.set_xticks(range(len(rois)), [r.replace("_", " ") for r in rois],
                  rotation=35, ha="right")
    ax.set_yticks(range(len(models)), [LABEL[m] for m in models])
    for i in range(len(models)):
        for j in range(len(rois)):
            v = mat.values[i, j]
            if not np.isfinite(v):
                continue
            star = "*" if np.isfinite(sig.values[i, j]) and sig.values[i, j] < 0.05 else ""
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=6,
                    color=INK if abs(v - 1.0) < 0.5 * (vmax - 1.0) else SURFACE)
    ax.set_title(f"{task.upper()} — attribution enrichment (1.0 = uniform; * Holm p<0.05)",
                 loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).outline.set_visible(False)
    out = FIGS / f"enrichment_{task}_{method}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def fig_ablation(task: str) -> Path:
    df = pd.read_csv(TABLES / f"ablation_{task}.csv")
    models = _order(df["model"].unique())
    rois = [r for r in ["vessel", "disc", "zone_b", "macula", "background"]
            if r in set(df["roi"])]
    agg = df.groupby(["model", "roi"]).agg(
        d=("delta_auc", "mean"), lo=("ci_lo", "mean"), hi=("ci_hi", "mean")).reset_index()

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.grid(axis="y", zorder=0)
    ax.axhline(0.0, color=MUTED, lw=0.7, zorder=1)
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        g = agg[agg["model"] == m].set_index("roi").reindex(rois)
        x = np.arange(len(rois)) + (k - (len(models) - 1) / 2) * w
        # 2px surface gap between adjacent bars
        ax.bar(x, g["d"], width=w * 0.86, color=PALETTE[m], label=LABEL[m],
               zorder=3, edgecolor=SURFACE, linewidth=0.8)
        # clip: a percentile bootstrap CI can occasionally fall on one side of
        # the point estimate, which matplotlib rejects as a negative error bar
        lo_err = np.clip(g["d"].to_numpy() - g["lo"].to_numpy(), 0, None)
        hi_err = np.clip(g["hi"].to_numpy() - g["d"].to_numpy(), 0, None)
        ax.errorbar(x, g["d"], yerr=[lo_err, hi_err],
                    fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=4)

    ax.set_xticks(range(len(rois)), [r.replace("_", " ") for r in rois])
    ax.set_ylabel("AUROC drop when masked")
    ax.set_title(f"{task.upper()} — does the highlighted region matter?",
                 loc="left", color=INK)
    ax.legend(frameon=False, ncol=2, labelcolor=INK)
    out = FIGS / f"ablation_{task}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def fig_agreement(task: str, method: str) -> Path:
    df = pd.read_csv(TABLES / f"agreement_{task}_{method}.csv")
    cross = df[df["kind"] == "cross_model"]
    models = _order(set(cross["arch_a"]) | set(cross["arch_b"]))
    mat = np.full((len(models), len(models)), np.nan)
    idx = {m: i for i, m in enumerate(models)}
    for (a, b), g in cross.groupby(["arch_a", "arch_b"]):
        if a in idx and b in idx:
            mat[idx[a], idx[b]] = mat[idx[b], idx[a]] = g["spearman"].mean()
    np.fill_diagonal(mat, np.nan)

    # Sequential: one hue, light -> dark (magnitude, no meaningful midpoint).
    cmap = mpl.colors.LinearSegmentedColormap.from_list("agr", ["#eef2fa", "#3468C0"])
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(models)), [LABEL[m] for m in models], rotation=35, ha="right")
    ax.set_yticks(range(len(models)), [LABEL[m] for m in models])
    for i in range(len(models)):
        for j in range(len(models)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color=INK if mat[i, j] < 0.55 else SURFACE)

    ceiling = df[df["kind"] == "ceiling"]
    title = f"{task.upper()} — cross-model saliency agreement (Spearman)"
    if len(ceiling):
        title += f"\nsame-architecture ceiling = {ceiling['spearman'].mean():.2f}"
    else:
        title += "\nNO CEILING RUNS — uninterpretable as stated"
    ax.set_title(title, loc="left", color=INK, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).outline.set_visible(False)
    out = FIGS / f"agreement_{task}_{method}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def fig_disc_qc(task: str, fold: int, vessel_dir: str, n: int = 30) -> Path:
    """Eyeball the heuristic disc localisation. Report the hit rate in the paper."""
    from .anatomy import estimate_disc, load_vessel_channels
    from .folddata import prepare_fold
    from .frames import CANON_SIZE, fov_mask, load_canonical, load_canonical_mask
    from .modelio import config_path, load_config

    cfg = load_config(config_path(task, "efficientnet_b2"), overrides={"fold": fold})
    df = prepare_fold(cfg, fold, vessel_dir).head(n)

    cols = 6
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))
    for ax, (_, r) in zip(np.ravel(axes), df.iterrows()):
        img = load_canonical(r["_image_path"], CANON_SIZE)
        ax.imshow(img)
        vp = Path(r["_vessel_path"])
        if vp.exists():
            ves = load_vessel_channels(load_canonical_mask(str(vp), CANON_SIZE))
            d = estimate_disc(img, fov_mask(img), ves["vessel"])
            ax.add_patch(plt.Circle((d.cx, d.cy), d.radius, fill=False,
                                    color="#B8860B", lw=1.4))
            ax.add_patch(plt.Circle((d.cx, d.cy), d.radius + 1.0 * d.dd, fill=False,
                                    color="#00897B", lw=0.8, ls="--"))
        ax.set_title(str(r["_sample_id"])[:14], fontsize=5, color=MUTED)
        ax.axis("off")
    for ax in np.ravel(axes)[len(df):]:
        ax.axis("off")
    fig.suptitle("disc estimate (solid) and Zone B outer edge (dashed) — count the misses",
                 fontsize=8, color=INK)
    out = FIGS / f"disc_qc_{task}_fold{fold}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["all", "radial", "enrichment", "ablation",
                                      "agreement", "disc-qc"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", default="rise")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--vessel-dir", default=None)
    args = ap.parse_args()

    made = []
    def attempt(fn, *a):
        try:
            made.append(fn(*a))
        except FileNotFoundError as e:
            print(f"  skip {fn.__name__}: {e}")

    if args.which in ("all", "radial"):
        attempt(fig_radial, args.task, args.method)
    if args.which in ("all", "enrichment"):
        attempt(fig_enrichment, args.task, args.method)
    if args.which in ("all", "ablation"):
        attempt(fig_ablation, args.task)
    if args.which in ("all", "agreement"):
        attempt(fig_agreement, args.task, args.method)
    if args.which == "disc-qc":
        attempt(fig_disc_qc, args.task, args.fold, args.vessel_dir)

    for p in made:
        print(f"  {p}")
    print("\nRendered. Now OPEN them and check for label collisions and overflow "
          "before dropping them into the paper.")


if __name__ == "__main__":
    main()
