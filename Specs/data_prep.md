# Data Preparation Specification - Project Retinexis

## Overview

This document (formatted by Claude but with our input for content) defines a blueprint for the standardized data preparation pipeline we want to use for Project Retinexis, focusing on image preprocessing and metadata handling for retinal fundus images.

---

## 1. Image Preprocessing Pipeline

### 1.1 Quality Control & Filtering

**Step 1: Image Quality Assessment**
- Check for minimum resolution (recommend ≥ 512x512 after preprocessing)
- Identify and flag images with:
  - Excessive blur (calculate Laplacian variance)
  - Poor illumination (mean pixel intensity checks)
  - Artifacts or occlusions
  - Incorrect color space
- Document percentage of images filtered and reasons

**Step 2: Duplicate Detection**
- Check for duplicate images using perceptual hashing
- Remove exact duplicates
- Flag near-duplicates for manual review

### 1.2 Standardization Steps

Apply the following preprocessing steps in order:

#### 1.2.1 Image Resizing
- **Target Resolution:** 512x512 or 224x224 (decide based on computational constraints)
- **Method:** Maintain aspect ratio with center cropping or padding
- **Interpolation:** Bicubic for downsampling, bilinear for upsampling

#### 1.2.2 Color Normalization
- **Method:** Standardize color distribution across dataset
- Options:
  - Histogram equalization (CLAHE!!!)
  - Mean/std normalization per channel
  - Normalize to ImageNet statistics (if using transfer learning)

#### 1.2.3 Retinal-Specific Preprocessing
- **ROI Extraction:** Detect and crop to optic disc region if needed
- **Vessel Enhancement:** Optional - apply filters to enhance vascular structures
- **Green Channel Emphasis:** Consider using green channel (highest contrast in fundus images)

#### 1.2.4 Data Augmentation Strategy (for training only)
Document your augmentation pipeline:
- Horizontal flips (common for fundus images)
- Random rotations (±15-20 degrees)
- Random brightness/contrast adjustments (±10-20%)
- Gaussian noise addition (σ = 0.01-0.05)
- Random cropping and resizing
- **Avoid:** Vertical flips (anatomically incorrect)

### 1.3 Preprocessing Code Template (This is pretty much all Claude code, but it has the methods we had in mind. Implement them the way you like.)

```python
# preprocessing/preprocess_images.py

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict

class FundusImagePreprocessor:
    """
    Standardized preprocessor for retinal fundus images.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.preprocessing_log = []
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image in RGB format."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def check_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess image quality metrics."""
        
        # Blur detection using Laplacian variance
        # Higher = sharper image. Typical range: 0-2000+
        # < 100 = very blurry, 100-500 = acceptable, > 500 = sharp
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness check
        # Range: 0-255. Good fundus images typically 80-180
        brightness = np.mean(image)
        
        # Contrast check
        # Range: 0-100+. Higher = more contrast. Good if > 30
        contrast = np.std(image)
        
        return {
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast
        }
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize with aspect ratio preservation."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance contrast."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def preprocess_pipeline(self, image_path: str, apply_clahe: bool = True) -> np.ndarray:
        """Complete preprocessing pipeline."""
        # Load
        img = self.load_image(image_path)
        
        # Quality check
        quality = self.check_quality(img)
        self.preprocessing_log.append({
            'image': image_path,
            'quality_metrics': quality
        })
        
        # Resize
        img = self.resize_image(img)
        
        # CLAHE (optional)
        if apply_clahe:
            img = self.apply_clahe(img)
        
        # Normalize
        img = self.normalize(img)
        
        return img
    
    def save_log(self, output_path: str):
        """Save preprocessing log."""
        with open(output_path, 'w') as f:
            json.dump(self.preprocessing_log, f, indent=2)
```

---

## 2. Metadata Handling

### 2.1 Understanding Your Dataset's Metadata Structure

Based on the High-Resolution Fundus dataset, the metadata is structured as a JSON/dictionary format with patient IDs as keys. Here's the structure:

```python
{
    "patient_id": {
        "gender": int,           # 1 = Male, 0 = Female (verify this)
        "thickness": float,      # Retinal thickness measurement
        "label": int,           # 0 = No CVD, 1 = CVD present
        "group": int,           # Patient group identifier
        "True_age": int,        # Actual patient age
        "age": float,           # Normalized/processed age
        "right_eye": string,    # Filename for right eye image (e.g., "248330005_R.png")
        "left_eye": string      # Filename for left eye image (e.g., "248330005_L.png")
    }
}
```

**Example from your dataset:**
```python
"248330005": {
    "gender": 1,
    "thickness": 1.1,
    "label": 1,
    "group": 1,
    "True_age": 70,
    "age": 0.7808219178082191,
    "right_eye": "248330005_R.png",
    "left_eye": "248330005_L.png"
}
```

### 2.2 Converting to Standardized Format

You'll need to flatten this into a CSV format for easier processing. Target structure:

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `patient_id` | string | Unique patient identifier | `248330005` |
| `image_id` | string | Unique image identifier | `248330005_R` |
| `image_filename` | string | Image filename | `248330005_R.png` |
| `eye` | string | R (right) or L (left) | `R` |
| `label` | int | Binary CVD label (0=negative, 1=positive) | `1` |
| `gender` | int | Gender (1=M, 0=F - verify encoding) | `1` |
| `age` | int | Patient age in years | `70` |
| `age_normalized` | float | Normalized age value | `0.7808` |
| `thickness` | float | Retinal thickness measurement | `1.1` |
| `group` | int | Patient group identifier | `1` |
| `split` | string | train/val/test (to be assigned) | `train` |

### 2.3 Metadata Processing Script for Your Dataset (This is pretty much all Claude code, but it has the methods we had in mind. Implement them the way you like.)

```python
# preprocessing/process_metadata.py

import pandas as pd
import json
from pathlib import Path

class CVDMetadataProcessor:
    """Process metadata for High-Resolution Fundus CVD dataset."""
    
    def __init__(self, metadata_json_path: str):
        """Load metadata from JSON file."""
        with open(metadata_json_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.df = None
        self.validation_errors = []
    
    def flatten_to_dataframe(self):
        """
        Convert nested JSON structure to flat DataFrame.
        Each patient has 2 rows (one per eye).
        """
        rows = []
        
        for patient_id, patient_data in self.metadata.items():
            # Right eye entry
            if patient_data.get('right_eye'):
                rows.append({
                    'patient_id': patient_id,
                    'image_id': f"{patient_id}_R",
                    'image_filename': patient_data['right_eye'],
                    'eye': 'R',
                    'label': patient_data['label'],
                    'gender': patient_data['gender'],
                    'age': patient_data['True_age'],
                    'age_normalized': patient_data['age'],
                    'thickness': patient_data['thickness'],
                    'group': patient_data['group']
                })
            
            # Left eye entry
            if patient_data.get('left_eye'):
                rows.append({
                    'patient_id': patient_id,
                    'image_id': f"{patient_id}_L",
                    'image_filename': patient_data['left_eye'],
                    'eye': 'L',
                    'label': patient_data['label'],
                    'gender': patient_data['gender'],
                    'age': patient_data['True_age'],
                    'age_normalized': patient_data['age'],
                    'thickness': patient_data['thickness'],
                    'group': patient_data['group']
                })
        
        self.df = pd.DataFrame(rows)
        print(f"Created DataFrame with {len(self.df)} images from {len(self.metadata)} patients")
    
    def validate_data(self):
        """Check for data quality issues."""
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            self.validation_errors.append(f"Missing values:\n{missing[missing > 0]}")
        
        # Validate labels are binary
        unique_labels = self.df['label'].unique()
        if not set(unique_labels).issubset({0, 1}):
            self.validation_errors.append(f"Invalid labels: {unique_labels}")
        
        # Check for duplicate image_ids
        duplicates = self.df['image_id'].duplicated().sum()
        if duplicates > 0:
            self.validation_errors.append(f"Found {duplicates} duplicate image IDs")
        
        # Validate age ranges
        if (self.df['age'] < 0).any() or (self.df['age'] > 120).any():
            self.validation_errors.append("Invalid age values found")
    
    def get_statistics(self):
        """Generate dataset statistics."""
        stats = {
            'total_images': len(self.df),
            'total_patients': self.df['patient_id'].nunique(),
            'class_distribution': self.df['label'].value_counts().to_dict(),
            'gender_distribution': self.df['gender'].value_counts().to_dict(),
            'age_stats': {
                'mean': float(self.df['age'].mean()),
                'std': float(self.df['age'].std()),
                'min': int(self.df['age'].min()),
                'max': int(self.df['age'].max())
            },
            'thickness_stats': {
                'mean': float(self.df['thickness'].mean()),
                'std': float(self.df['thickness'].std()),
                'min': float(self.df['thickness'].min()),
                'max': float(self.df['thickness'].max())
            }
        }
        
        return stats
    
    def save_processed_metadata(self, output_dir: str):
        """Save processed metadata and statistics."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_path / 'processed_metadata.csv'
        self.df.to_csv(csv_path, index=False)
        print(f"\nProcessed metadata saved to {csv_path}")
        
        # Save statistics
        stats = self.get_statistics()
        stats_path = output_path / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_path}")
        
        # Save validation report
        report_path = output_path / 'validation_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("METADATA VALIDATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            if self.validation_errors:
                f.write("⚠️  VALIDATION ERRORS FOUND:\n")
                f.write("-"*60 + "\n")
                for error in self.validation_errors:
                    f.write(f"{error}\n\n")
            else:
                f.write("✓ No validation errors found\n\n")
            
            f.write("="*60 + "\n")
            f.write("DATASET STATISTICS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Images: {stats['total_images']}\n")
            f.write(f"Total Patients: {stats['total_patients']}\n\n")
            
            f.write("Class Distribution:\n")
            for label, count in stats['class_distribution'].items():
                f.write(f"  Label {label}: {count} ({count/stats['total_images']*100:.1f}%)\n")
        
        print(f"Validation report saved to {report_path}")


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = CVDMetadataProcessor('path/to/your/metadata.json')
    
    # Convert to DataFrame
    processor.flatten_to_dataframe()
    
    # Validate
    processor.validate_data()
    
    # Get statistics
    stats = processor.get_statistics()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Save everything
    processor.save_processed_metadata('data/processed/cardiovascular/metadata/')
```

**CRITICAL NOTE on Train/Val/Test Splits:**
Creating data splits is part of the training phase, not data prep. However, when you do create splits later, remember to keep both eyes from the same patient in the same split to avoid data leakage.

---

## Reference Notes

### Recommended Libraries
- **Image Processing:** OpenCV, PIL, scikit-image
- **Data Handling:** pandas, numpy
- **Augmentation:** albumentations, imgaug
- **Deep Learning:** PyTorch (torchvision.transforms)

### Fundus-Specific Tips
- Use CLAHE for contrast enhancement
- Green channel has highest contrast in fundus images
- Avoid vertical flips (anatomically incorrect)
- Check for blur using Laplacian variance

---