import os
import json
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from glob import glob
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── Configuration ────────────────────────────────────────────────────────────
# Set this to the root directory containing Fundus_CIMT_2903_Dataset/
DATA_ROOT = ""

DATASET_DIR      = DATA_ROOT + "Fundus_CIMT_2903_Dataset/"
IMAGE_DIR        = DATASET_DIR + "images/"
OUTPUT_IMAGE_DIR = DATASET_DIR + "images_preprocessed/"
METADATA_PATH    = DATASET_DIR + "data_info.json"
METADATA_OUT_DIR = DATASET_DIR + "metadata_processed/"

# ── Image preprocessing ───────────────────────────────────────────────────────

def remove_top_left_overlay(img, width=90, height=50, white_threshold=200):
    """Black out the top-left region if it contains white overlay pixels."""
    img_copy = img.copy()
    top_left = img_copy[:height, :width]
    gray = cv2.cvtColor(top_left, cv2.COLOR_RGB2GRAY) if len(top_left.shape) == 3 else top_left
    if np.sum(gray > white_threshold) > 0:
        img_copy[:height, :width] = np.array([6, 6, 6], dtype=img_copy.dtype)
        return img_copy, True
    return img_copy, False


def apply_clahe(img):
    """Apply CLAHE to the green channel only (highest signal in fundus images)."""
    r, g, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.merge((r, clahe.apply(g), b))


class ImagePreprocessor:
    """Sequential image preprocessing pipeline."""

    def __init__(self):
        self.techniques = []

    def add_technique(self, name, func):
        self.techniques.append((name, func))
        return self

    def process(self, img):
        for _, func in self.techniques:
            img = func(img)
        return img

    def process_batch(self, image_paths, output_dir, progress_interval=100):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing {len(image_paths)} images with {len(self.techniques)} technique(s)...\n")
        processed_count = 0

        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed = self.process(img_rgb)
            cv2.imwrite(
                os.path.join(output_dir, Path(img_path).name),
                cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
            )
            processed_count += 1

            if (i + 1) % progress_interval == 0:
                print(f"  {i + 1}/{len(image_paths)} images processed...")

        print(f"\nDone. {processed_count}/{len(image_paths)} images saved to: {output_dir}")


# ── Metadata preprocessing ────────────────────────────────────────────────────

class MetadataProcessor:
    def __init__(self, metadata_json_path: str, image_dir: str = None):
        with open(metadata_json_path, "r") as f:
            self.metadata = json.load(f)
        self.image_dir = image_dir
        self.df = None
        self.validation_errors = []

    def flatten_to_dataframe(self):
        """Convert nested JSON to flat DataFrame with one row per image."""
        rows = []
        for patient_id, data in self.metadata.items():
            for eye, side in [("right_eye", "R"), ("left_eye", "L")]:
                if data.get(eye):
                    rows.append({
                        "patient_id":       patient_id,
                        "image_id":         f"{patient_id}_{side}",
                        "image_filename":   data[eye],
                        "eye":              side,
                        "label":            data["label"],
                        "gender":           data["gender"],
                        "age":              data["True_age"],
                        "age_normalized":   data["age"],
                        "thickness":        data["thickness"],
                        "group":            data["group"],
                    })
        self.df = pd.DataFrame(rows)
        print(f"Flattened {len(self.df)} image entries from {len(self.metadata)} patients.")

    def validate_data(self):
        missing = self.df.isnull().sum()
        if missing.any():
            self.validation_errors.append(
                f"Missing values: {missing[missing > 0].to_dict()}"
            )

        if self.image_dir:
            missing_files = [
                f for f in self.df["image_filename"]
                if not os.path.exists(os.path.join(self.image_dir, f))
            ]
            if missing_files:
                self.validation_errors.append(
                    f"Missing {len(missing_files)} image files on disk."
                )

        if (self.df["thickness"] <= 0).any():
            self.validation_errors.append("Invalid thickness values (<=0) found.")

        imbalance = self.df["label"].value_counts(normalize=True).max()
        if imbalance > 0.8:
            self.validation_errors.append(
                f"Severe class imbalance detected ({imbalance:.1%})."
            )

    def get_statistics(self):
        return {
            "total_images":        len(self.df),
            "label_distribution":  self.df["label"].value_counts().to_dict(),
            "gender_distribution": self.df["gender"].value_counts().to_dict(),
            "thickness_stats":     self.df["thickness"].describe().to_dict(),
            "age_distribution":    self.df["age"].describe().to_dict(),
        }

    def save_processed_metadata(self, output_dir: str):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(out_path / "processed_metadata.csv", index=False)

        with open(out_path / "dataset_statistics.json", "w") as f:
            json.dump(self.get_statistics(), f, indent=2)

        with open(out_path / "validation_report.txt", "w") as f:
            f.write("METADATA VALIDATION REPORT\n" + "=" * 25 + "\n")
            if not self.validation_errors:
                f.write("Status: All checks passed.\n")
            else:
                for err in self.validation_errors:
                    f.write(f"Error: {err}\n")

        print(f"Metadata saved to: {output_dir}")


# ── Augmentation (for training use, not baked into preprocessed images) ───────

def get_train_augmentation():
    """Returns an albumentations transform for use in a training DataLoader."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-10, 10), scale=(0.98, 1.02), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.4),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retina_images = glob(IMAGE_DIR + "*.png")
    print(f"Found {len(retina_images)} images.\n")

    # Image preprocessing
    preprocessor = ImagePreprocessor()
    preprocessor.add_technique("remove_overlay", lambda img: remove_top_left_overlay(img)[0])
    preprocessor.add_technique("CLAHE", apply_clahe)
    aug = get_train_augmentation()
    preprocessor.add_technique("randaug", lambda img: aug(image=img)["image"])
    preprocessor.process_batch(retina_images, OUTPUT_IMAGE_DIR)

    # Metadata preprocessing
    print()
    metadata_processor = MetadataProcessor(
        metadata_json_path=METADATA_PATH,
        image_dir=IMAGE_DIR,
    )
    metadata_processor.flatten_to_dataframe()
    metadata_processor.validate_data()
    metadata_processor.save_processed_metadata(METADATA_OUT_DIR)
