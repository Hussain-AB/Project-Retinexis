# Data Exploration & Analysis Specification

## Overview

This document defines the protocols for the EDA phase. Our goal is not just to view the images, but to rigorously **audit** them. We must prove that our model will learn medical features rather than cheating on artifacts (e.g., camera noise, black borders, or patient age).

---

## 1. Data Forensics & Quality (The "Sanity Check")

Before trusting the data, we must verify its physical integrity.

### 1.1 Physical Audit

* **Resolution Clustering:** Plot the `Width` vs `Height` of all images.
* *Goal:* Detect if multiple cameras were used (e.g., a cluster at 2000 x 2000 vs one at 512 x 512). Different cameras introduce "domain shift" that can confuse the model.


* **Black Border Analysis (Fill Factor):** Calculate the ratio of the eye area to the total image area.
* *Goal:* Identify images with inconsistent cropping. If the model sees more "black background" on sick patients, it might learn to predict disease based on the border size.


* **Green Channel Variance:** Calculate the standard deviation of the **Green Channel** specifically.
* *Goal:* The green channel contains the most vascular detail. Low variance here means the image is washed out or blurry.


* **Blur Detection:** Measure the Laplacian variance (or other blur metrics) for all images to identify severely blurred images.
* *Goal:* Quantify how many images are too blurry to be useful for training. Flag images below a threshold (e.g., bottom 5%) and determine if they should be excluded from the training set.
* *Impact:* Blurry images don't contain reliable medical features and can reduce model performance.


---

## 2. Leakage Prevention (The "Safety Check")

**CRITICAL:** This is the most common reason for project failure. We must ensure the model cannot "cheat."

### 2.1 Patient-Level Isolation

* **Rule:** Never split data by `Image_ID`. You must split by `Patient_ID`.
* **Verification:** Run a code cell that asserts: `set(train_patients).intersection(set(test_patients))` is empty.
* **Laterality Check:** Verify `Left` vs `Right` labels.
* *Method:* The Optic Disc is usually the brightest spot. If the label is "Left Eye," the bright spot should be on the left half of the image. Flag mismatches.



### 2.2 Artifact Scanning (The "Average Image")

* **Method:** Compute the pixel-wise average of all images in the "Healthy" class and all images in the "Disease" class.
* **Visualization:** Display these two "Average Images" side-by-side.
* **What to look for:**
* **Ghost Text:** Can you read "R" or "L" or date stamps in the average image?
* **Laser Scars:** Does the "Disease" average look darker or have spotty patterns (laser treatment)?
* *Why?* If artifacts appear in the average, the model will use them as a shortcut.



---

## 3. Clinical & Demographic Analysis (The "Bias Check")

We must verify that our dataset doesn't contain hidden biases that could lead to unfair medical predictions.

### 3.1 Confounding Variables

* **Feature vs. Disease:**
* *Plot:* Boxplot of a certain feature for "Healthy" vs "Disease".
* *Test:* Run a **T-test** or **Kruskal-Wallis test**.
* *Risk:* If sick patients are strongly correlated with a certain feature, the model might just learn to predict that feature instead of the disease. Focus on features such as Age and Sex


---

## 4. Advanced Imaging Analytics (The "Deep Dive")

Move beyond histograms. We need to visualize the "manifold" (the shape of the data).

### 4.1 UMAP Visualization

* **Feature Extraction:** Pass a sample of images through a pre-trained network (like ResNet18) and grab the feature vector (before the final classification layer).
* **Projection:** Use **UMAP** to project these high-dimensional vectors into 2D dots.
* **Coloring:**
* Color by **Disease Grade**: Do they separate well?
* Color by **Hospital/Source**: Do they cluster by hospital? (This indicates a "Batch Effect" error).



---

## 5. Code Templates

### 5.1 Average Image & Artifact Scan (Code Written by Gemini)

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

def compute_average_image(image_paths):
    """Computes the average image from a list of paths to detect artifacts."""
    avg_img = None
    count = 0
    
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (512, 512)) # Standardize size for averaging
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)
        
        if avg_img is None:
            avg_img = img
        else:
            avg_img += img
        count += 1
            
    avg_img /= count
    return avg_img.astype(np.uint8)

# Usage:
# healthy_avg = compute_average_image(healthy_paths)
# disease_avg = compute_average_image(disease_paths)
# Plot them side-by-side to check for "ghost" text or laser scars.

```

### 5.2 UMAP Manifold Visualization (Code Written by Gemini)

```python
import umap
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. Load Pre-trained ResNet (Feature Extractor)
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1]) # Remove classification layer
model.eval()

# 2. Extract Features
def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose()
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze().numpy()

# Run this on a subset (e.g., 500 images)
embeddings = [get_embedding(p) for p in subset_paths] 

# 3. UMAP Projection
reducer = umap.UMAP()
embedding_2d = reducer.fit_transform(embeddings)

# 4. Plot
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=subset_labels, cmap='Spectral')
plt.title("Data Manifold: Do disease classes cluster naturally?")
plt.colorbar()
plt.show()

```

### 5.3 Statistical Hypothesis Testing (Code Written by Gemini)

```python
from scipy import stats
import pandas as pd

def check_bias(df, target_col='Label', sensitive_col='Age'):
    """
    Checks if a continuous variable (Age) differs significantly across classes.
    """
    groups = [df[df[target_col] == cls][sensitive_col] for cls in df[target_col].unique()]
    
    # Kruskal-Wallis Test (Non-parametric ANOVA)
    stat, p_value = stats.kruskal(*groups)
    
    print(f"Bias Check ({sensitive_col}): P-value = {p_value:.4f}")
    if p_value < 0.05:
        print("⚠️ SIGNIFICANT BIAS DETECTED: The distributions are different!")
    else:
        print("✅ No significant bias detected.")

```