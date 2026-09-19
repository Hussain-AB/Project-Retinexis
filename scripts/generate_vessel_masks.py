"""
generate_vessel_masks.py — Generate RRWNet vessel masks for CIMT fundus images.

Loads RRWNet directly from HuggingFace (no manual weight download needed),
generates ROI masks from the fundus background, applies the RITE preprocessing,
and saves artery/vein/vessel segmentation masks as RGB PNGs.

Usage:
    python scripts/generate_vessel_masks.py \
        --images-dir data/cimt/Fundus_CIMT_2903_Dataset/images \
        --save-dir data/cimt/vessel_masks

Output:
    One PNG per input image (same filename) in save-dir.
    Each PNG is RGB [0, 255] where:
        Red   = arteries
        Green = veins
        Blue  = vessels (artery + vein union)

Requirements:
    pip install huggingface_hub scikit-image scipy
    (torch, torchvision already in your venv)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin
from scipy import ndimage
from torchvision import utils as vutils
from PIL import Image

# Import RRWNet model and helpers from the rrwnet subfolder.
# preprocessing.py already has enhance_image which handles:
#   - mask erosion (disk(5) kernel)
#   - zoom + high-pass filter + std normalisation
# utils.py has pad_images_unet and to_torch_tensors.
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "rrwnet"))
from model import RRWNet as _RRWNetBase
from preprocessing import enhance_image, remove_corner_artifact
from utils import pad_images_unet, to_torch_tensors


# Wrap with HuggingFace mixin for from_pretrained support
class RRWNet(_RRWNetBase, PyTorchModelHubMixin):
    def __init__(self, input_ch=3, output_ch=3, base_ch=64, iterations=5):
        super().__init__(input_ch, output_ch, base_ch, iterations)


def generate_roi_mask(img: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Generate a binary ROI mask from a fundus image numpy array.

    Detects the circular retinal field by thresholding the black background.
    Returns uint8 array (H, W) with values 0 or 255.

    Note: no pre-erosion here — enhance_image already erodes with disk(5).
    """
    gray = img.max(axis=2)
    binary = (gray > threshold).astype(np.uint8)
    filled = ndimage.binary_fill_holes(binary).astype(np.uint8)

    labeled, n = ndimage.label(filled)
    if n == 0:
        return np.ones(gray.shape, dtype=np.uint8) * 255

    sizes = ndimage.sum(filled, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    mask = (labeled == largest).astype(np.uint8)

    return mask * 255


def process_image(image_path: Path, model: RRWNet, device: torch.device) -> torch.Tensor:
    """Run RRWNet on a single fundus image. Returns (1, 3, H, W) float tensor [0, 1]."""
    # Load raw image as numpy array
    img = np.array(Image.open(image_path).convert("RGB"))

    # Remove corner watermark before any processing
    img = remove_corner_artifact(img)

    # Generate ROI mask from the cleaned image (enhance_image handles erosion internally)
    roi_mask = generate_roi_mask(img)

    # Preprocess: enhance_image accepts numpy arrays directly
    img_enhanced, mask = enhance_image(img, roi_mask)

    # Pad to UNet-compatible dimensions (multiples of 32)
    imgs, paddings = pad_images_unet([img_enhanced, mask])
    img_padded, padding = imgs[0], paddings[0]
    mask_padded = np.stack([imgs[1]] * 3, axis=2)

    tensors = to_torch_tensors([img_padded, mask_padded])
    image_tensor = tensors[0].unsqueeze(0).to(device)
    mask_tensor = tensors[1].unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)
        last_pred = torch.sigmoid(predictions[-1])
        last_pred[mask_tensor < 0.5] = 0

        # Remove padding (padding may be 2- or 3-tuple; only H/W dims needed)
        r, c = padding[0], padding[1]
        if r[1] > 0 and c[1] > 0:
            last_pred = last_pred[:, :, r[0]:-r[1], c[0]:-c[1]]
        elif r[1] > 0:
            last_pred = last_pred[:, :, r[0]:-r[1], :]
        elif c[1] > 0:
            last_pred = last_pred[:, :, :, c[0]:-c[1]]

    return last_pred


def main():
    parser = argparse.ArgumentParser(description="Generate RRWNet vessel masks for CIMT images")
    parser.add_argument("--images-dir", required=True, help="Directory of raw fundus images (PNG)")
    parser.add_argument("--save-dir", required=True, help="Directory to save vessel masks")
    parser.add_argument("--hf-repo", default="j-morano/rrwnet-rite",
                        help="HuggingFace repo (default: j-morano/rrwnet-rite)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="RRWNet refinement iterations (default: 5)")
    parser.add_argument("--ext", default="png", help="Image file extension (default: png)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading RRWNet from HuggingFace: {args.hf_repo}")
    model = RRWNet.from_pretrained(args.hf_repo, iterations=args.iterations)
    model.eval().to(device)
    print("Model loaded.")

    image_files = sorted(images_dir.glob(f"*.{args.ext}"))
    if not image_files:
        print(f"No .{args.ext} files found in {images_dir}")
        return

    print(f"Processing {len(image_files)} images → {save_dir}")
    failed = []

    for i, img_path in enumerate(image_files):
        try:
            pred = process_image(img_path, model, device)
            vutils.save_image(pred, save_dir / img_path.name)
        except Exception as e:
            print(f"  FAILED {img_path.name}: {e}")
            failed.append(img_path.name)

        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  {i + 1}/{len(image_files)}")

    print(f"\nDone. {len(image_files) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()