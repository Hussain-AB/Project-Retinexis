"""
xai.modelio — load a trained (task, model, fold) and score canonical-frame images.

Reuses the project's own config loader and model factory so nothing about
architecture construction is duplicated here. Checkpoints are read from the
run layout TrainConfig already defines:

    {output_dir}/{dataset}/{architecture}/fold_{fold}/checkpoints/best.pt

`CanonicalScorer` is the single entry point every attribution method uses. It
takes uint8 canonical-frame images and returns raw logits, applying exactly the
resize+normalise the model's PreprocessingConfig declares. Because the input is
already square, this is equivalent to the val-time torchvision pipeline with
the crop step removed (see frames.py for why that matters).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .paths import (MODEL_FAMILY, MODEL_KEYS, PROJECT_ROOT, TASK_DIR,  # noqa: F401
                    config_path)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.loader import load_config  # noqa: E402
from src.models import create_model  # noqa: E402


def checkpoint_path(cfg, project_root: Path = PROJECT_ROOT) -> Path:
    """Mirror of TrainConfig.run_dir, which now encodes the seed.

    Runs at the reference seed (42) keep the historical layout; any other seed
    lives in fold_{k}_seed{s}. This MUST stay in step with
    src/config/schema.py::run_dir or --checkpoint-suffix resolves to the wrong
    file and the seed ceiling is computed from the wrong models.
    """
    run_dir = Path(cfg.output_dir)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    tag = f"fold_{cfg.fold}"
    seed = int(getattr(cfg, "seed", 42))
    if seed != 42:
        tag = f"{tag}_seed{seed}"
    return run_dir / cfg.dataset / cfg.architecture / tag / "checkpoints" / "best.pt"


@dataclass
class LoadedModel:
    key: str
    task: str
    fold: int
    model: torch.nn.Module
    cfg: object
    input_size: int
    ckpt_path: Path


def load_model(
    task: str,
    model_key: str,
    fold: int,
    device: torch.device,
    project_root: Path = PROJECT_ROOT,
    checkpoint_override: Path | None = None,
) -> LoadedModel:
    cfg = load_config(config_path(task, model_key, project_root), overrides={"fold": fold})

    # pretrained=False: weights come from the checkpoint, so skip the download.
    kwargs = dict(cfg.model_kwargs)
    if "pretrained" in kwargs:
        kwargs["pretrained"] = False
    try:
        model = create_model(cfg.architecture, **kwargs)
    except Exception:
        # Some backbones need their hub/HF scaffolding to build at all.
        model = create_model(cfg.architecture, **cfg.model_kwargs)

    ckpt = Path(checkpoint_override) if checkpoint_override else checkpoint_path(cfg, project_root)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint for {task}/{model_key}/fold{fold} at {ckpt}.\n"
            f"Download the fold checkpoints from the team HF repo into {ckpt.parent}."
        )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"{model_key} fold{fold}: missing keys {missing[:6]} ...")
    if unexpected:
        print(f"  [warn] {model_key} fold{fold}: unexpected keys ignored: {unexpected[:4]}")

    if getattr(cfg, "use_metadata", False):
        raise ValueError(
            f"{model_key}/{task} config has use_metadata=True. The attribution study "
            "must run on the image-only models -- a metadata head makes the saliency "
            "un-interpretable as image evidence. Point this at the image-only configs."
        )

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    return LoadedModel(
        key=model_key,
        task=task,
        fold=fold,
        model=model,
        cfg=cfg,
        input_size=model.preprocessing_config.input_size,
        ckpt_path=ckpt,
    )


class CanonicalScorer:
    """Score canonical-frame images with one model. Returns raw logits."""

    def __init__(self, loaded: LoadedModel, device: torch.device, amp: bool = True):
        pcfg = loaded.model.preprocessing_config
        self.model = loaded.model
        self.device = device
        self.size = pcfg.input_size
        self.amp = amp and device.type == "cuda"
        self.mean = torch.tensor(pcfg.mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(pcfg.std, device=device).view(1, 3, 1, 1)

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float in [0, 1] on device."""
        if x.shape[-1] != self.size:
            x = F.interpolate(
                x, size=(self.size, self.size),
                mode="bicubic", align_corners=False, antialias=True,
            ).clamp(0, 1)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def score_batch(self, images: np.ndarray) -> np.ndarray:
        """images: (B, H, W, 3) float32 or uint8 in [0, 255]. -> (B,) logits."""
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            logits = self.model(self._prepare(x))
        return logits.float().squeeze(1).cpu().numpy()

    def logits_with_grad(self, x01: torch.Tensor) -> torch.Tensor:
        """Differentiable path for Integrated Gradients.

        x01: (B, 3, H, W) float in [0, 1] on device, requires_grad on the leaf.
        Runs in fp32 -- autocast is off here because IG accumulates small
        gradients over many steps and fp16 underflows them.
        """
        return self.model(self._prepare(x01)).squeeze(1)


def stratified_subsample(
    df, label_col: str, n: int, seed: int = 42
):
    """Label-stratified subsample of a fold's held-out rows.

    Attribution is expensive; a stratified subsample keeps the positive rate of
    the full fold so enrichment statistics are not silently reweighted.
    """
    if n <= 0 or n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in df.groupby(df[label_col].astype(int)):
        take = max(1, int(round(n * len(grp) / len(df))))
        take = min(take, len(grp))
        parts.append(grp.iloc[rng.permutation(len(grp))[:take]])
    import pandas as pd
    return pd.concat(parts).sort_index()
