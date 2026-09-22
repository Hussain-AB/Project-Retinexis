#!/usr/bin/env python3
"""Prove the vessel mask receives the same geometry as its image.

    python scripts/verify_paired_aug.py

Test design: photometric augmentation is switched off, so the ONLY thing that
can make the image and mask tensors differ is the geometric draw. With shared
geometry they must be bitwise identical on every trial. The same comparison is
run against the old independent-pipeline path, which must FAIL -- otherwise the
test is not sensitive to the bug it exists to catch.

Exit 0 = paired geometry verified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import PairedTransform, build_transforms, build_vessel_transforms  # noqa: E402
from src.models.base import PreprocessingConfig  # noqa: E402

N_TRIALS = 200


class Cfg:
    """Geometric augmentation on, photometric off."""
    aug_hflip = True
    aug_rotation = 10.0
    aug_scale_min = 0.98
    aug_scale_max = 1.02
    aug_brightness = 0.0
    aug_contrast = 0.0
    aug_saturation = 0.0
    aug_hue = 0.0
    aug_blur_prob = 0.0
    aug_blur_limit = 3
    aug_random_erasing = 0.0


def asymmetric() -> Image.Image:
    """Deliberately not left-right symmetric, so an hflip is detectable."""
    img = Image.new("RGB", (512, 512), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([60, 60, 180, 180], fill=(255, 255, 255))     # "disc", upper left
    d.line([100, 300, 400, 340], fill=(200, 40, 40), width=9)
    d.polygon([(380, 80), (470, 130), (400, 200)], fill=(40, 200, 40))
    return img


def main() -> int:
    cfg = PreprocessingConfig(
        input_size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        crop_strategy="pad_to_square",
    )
    src = asymmetric()

    print(f"trials: {N_TRIALS}, photometric augmentation disabled\n")

    # --- paired path -------------------------------------------------------
    paired = PairedTransform(cfg, cfg, Cfg(), is_train=True)
    mismatches, moved = 0, 0
    baseline = None
    for _ in range(N_TRIALS):
        img_t, msk_t = paired(src.copy(), src.copy())
        if not torch.equal(img_t, msk_t):
            mismatches += 1
        if baseline is None:
            baseline = img_t
        elif not torch.equal(img_t, baseline):
            moved += 1

    ok = True
    if mismatches == 0:
        print(f"  PASS   shared geometry: image == mask on all {N_TRIALS} trials")
    else:
        print(f"  ERROR  image != mask on {mismatches}/{N_TRIALS} trials")
        ok = False

    if moved > N_TRIALS * 0.5:
        print(f"  PASS   augmentation is live: {moved}/{N_TRIALS - 1} trials differ from the first")
    else:
        print(f"  ERROR  only {moved} trials differed -- augmentation may be inert, "
              f"so the test above proves nothing")
        ok = False

    # --- eval mode ---------------------------------------------------------
    ev = PairedTransform(cfg, cfg, Cfg(), is_train=False)
    a_i, a_m = ev(src.copy(), src.copy())
    b_i, b_m = ev(src.copy(), src.copy())
    if torch.equal(a_i, b_i) and torch.equal(a_i, a_m) and torch.equal(b_i, b_m):
        print("  PASS   eval mode is deterministic and still aligned")
    else:
        print("  ERROR  eval mode is not deterministic or not aligned")
        ok = False

    # --- the OLD path, which must fail -------------------------------------
    img_tf = build_transforms(cfg, is_train=True, config=Cfg())
    msk_tf = build_vessel_transforms(cfg)
    old_mismatch = sum(
        0 if torch.equal(img_tf(src.copy()), msk_tf(src.copy())) else 1
        for _ in range(N_TRIALS)
    )
    rate = old_mismatch / N_TRIALS
    if old_mismatch > 0:
        print(f"  PASS   old independent path misaligns {old_mismatch}/{N_TRIALS} "
              f"({rate:.0%}) -- the test is sensitive to the bug")
    else:
        print("  ERROR  old path showed no misalignment; this test cannot detect the bug")
        ok = False

    print(f"\n{'paired geometry verified.' if ok else 'VERIFICATION FAILED.'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
