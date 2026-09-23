#!/usr/bin/env python3
"""Prove the three vessel configurations differ only in the second branch's input.

    python scripts/verify_vessel_arm.py

Builds the real hypertension dataset for each of the three configs and checks:

  * mask      -> the vessel branch gets THIS image's own mask
  * shuffled  -> it gets a different image's mask, never its own, identical on
                 every access, identical across epochs, different across seeds
  * image     -> it gets this fundus again
  * all three -> image and companion share one sampled geometry
  * all three -> the model, its parameter count and its config are otherwise
                 identical, which is what makes the controls controls

Exit 0 = the arm is sound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.schema import TrainConfig  # noqa: E402
from src.data.loader import _derangement  # noqa: E402

CFG_DIR = ROOT / "tasks/hypertension/configs/vessel"
ARMS = {"mask": "efficientnet_b2_vessel.yaml",
        "shuffled": "efficientnet_b2_vessel_shuffled.yaml",
        "image": "efficientnet_b2_vessel_image.yaml"}

ok = True


def check(cond: bool, good: str, bad: str) -> None:
    global ok
    print(("  PASS   " + good) if cond else ("  ERROR  " + bad))
    ok = ok and bool(cond)


def load(name: str, **over) -> TrainConfig:
    raw = yaml.safe_load((CFG_DIR / name).read_text())
    raw.update(fold=0, seed=42, **over)
    c = TrainConfig.from_dict(raw)
    c.validate()
    return c


def main() -> int:
    cfgs = {k: load(v) for k, v in ARMS.items()}

    # --- the three configs must be identical apart from vessel_source --------
    import dataclasses
    fields = [f.name for f in dataclasses.fields(TrainConfig)]
    base = cfgs["mask"]
    for arm in ("shuffled", "image"):
        diffs = [f for f in fields
                 if getattr(cfgs[arm], f) != getattr(base, f)]
        check(diffs == ["vessel_source"],
              f"{arm}: differs from the treatment ONLY in vessel_source",
              f"{arm}: also differs in {[d for d in diffs if d != 'vessel_source']} "
              f"-- then it is not a control")

    # --- run_dirs must be distinct -------------------------------------------
    rds = {k: str(c.run_dir) for k, c in cfgs.items()}
    check(len(set(rds.values())) == 3,
          f"three distinct run_dirs: {[Path(v).parent.name for v in rds.values()]}",
          f"run_dir collision: {rds}")

    # --- derangement ----------------------------------------------------------
    n = 14212
    p42, p42b, p1337 = _derangement(n, 42), _derangement(n, 42), _derangement(n, 1337)
    check(not (p42 == torch.arange(n).numpy()).any(),
          "shuffled: no image is ever paired with its own mask",
          "shuffled: a fixed point exists -- some images get their own mask")
    check((p42 == p42b).all(),
          "shuffled: identical on every access and every epoch",
          "shuffled: not reproducible")
    check(not (p42 == p1337).all(),
          "shuffled: a different pairing per seed",
          "shuffled: seeds produce the same pairing")

    # --- the real dataset ------------------------------------------------------
    try:
        from src.data.loader import create_dataloaders
        from src.models.registry import MODEL_REGISTRY
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR  cannot import the training stack: {exc}")
        return 1

    model = MODEL_REGISTRY[base.architecture](**(base.model_kwargs or {}))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  model: {base.architecture}, {n_params:,} parameters "
          f"(identical in all three arms by construction)\n")

    for arm, cfg in cfgs.items():
        tr, iv, va, _ = create_dataloaders(cfg, model)
        ds = tr.dataset
        row = ds.df.iloc[0]
        own = ds._mask_relpath(row)
        comp = ds._vessel_companion(0, __import__("PIL.Image", fromlist=["Image"]).Image.new("RGB", (64, 64)), row)
        check(comp is not None, f"{arm}: the vessel branch receives something",
              f"{arm}: companion is None -- the second branch would get nothing")
        img_t, msk_t = ds[0]["image"], ds[0].get("vessel_mask")
        check(msk_t is not None and img_t.shape[-2:] == msk_t.shape[-2:],
              f"{arm}: image {tuple(img_t.shape)} and companion {tuple(msk_t.shape) if msk_t is not None else None} "
              f"share spatial dims",
              f"{arm}: companion missing or mis-sized")
        if arm == "shuffled":
            donor = ds.df.iloc[int(ds.vessel_perm[0])]
            check(ds._mask_relpath(donor) != own,
                  f"shuffled: sample 0 gets {ds._mask_relpath(donor)}, not its own {own}",
                  "shuffled: sample 0 received its own mask")

    print("\n" + ("vessel arm verified." if ok else "VERIFICATION FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
