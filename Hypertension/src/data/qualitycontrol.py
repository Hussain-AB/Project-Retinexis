"""
Quality Control and Filtering for Retinal Fundus Images
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import imagehash
from PIL import Image
from collections import defaultdict
import json
from datetime import datetime

class ImageQualityController:
    def __init__(self, images_dir, output_dir, min_resolution=512, max_images=None):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_resolution = min_resolution
        self.max_images = max_images  # None = process all, or set a number for testing

        self.passed_images_dir = self.output_dir / "passed_images"
        self.passed_images_dir.mkdir(parents=True, exist_ok=True)

        
        # Quality metrics storage
        self.quality_report = []
        self.filtered_images = []
        self.duplicates = defaultdict(list)
        self.near_duplicates = defaultdict(list)
        
    def calculate_blur(self, image):
        """Calculate Laplacian variance to detect blur"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def check_illumination(self, image):
        """Check mean pixel intensity for poor illumination"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        return mean_intensity, std_intensity
    
    def has_text_overlay(self, image):
        """Detect text overlay in top-left corner (top 15% of image, left 30%)"""
        height, width = image.shape[:2]
        top_left_region = image[0:int(height*0.15), 0:int(width*0.3)]
        
        # Convert to grayscale
        gray = cv2.cvtColor(top_left_region, cv2.COLOR_BGR2GRAY)
        
        # Check for high contrast edges (typical of text)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Text typically has high edge density
        return edge_density > 0.05
    
    def assess_image_quality(self, image_path):
        """Comprehensive quality assessment for a single image"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    'filename': image_path.name,
                    'status': 'FAILED',
                    'reason': 'Cannot read image',
                    'pass_qc': False
                }
            
            height, width = image.shape[:2]
            
            # Initialize quality metrics
            issues = []
            
            # 1. Check resolution
            min_dimension = min(height, width)
            if min_dimension < self.min_resolution:
                issues.append(f'Low resolution: {width}x{height}')
            
            # 2. Check blur
            blur_score = self.calculate_blur(image)
            if blur_score < 10:  # Threshold for excessive blur
                issues.append(f'Excessive blur: {blur_score:.2f}')
            
            # 3. Check illumination
            mean_intensity, std_intensity = self.check_illumination(image)
            if mean_intensity < 30:
                issues.append(f'Too dark: mean={mean_intensity:.2f}')
            elif mean_intensity > 225:
                issues.append(f'Too bright: mean={mean_intensity:.2f}')
            
            if std_intensity < 20:
                issues.append(f'Low contrast: std={std_intensity:.2f}')
            
            # 4. Check color space
            if len(image.shape) != 3 or image.shape[2] != 3:
                issues.append('Incorrect color space (not BGR/RGB)')
            
            # 5. Check for text overlay
            if self.has_text_overlay(image):
                issues.append('Text overlay detected in top-left')
            
            # 6. Check for artifacts (excessive noise)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            noise_level = np.std(cv2.Laplacian(gray, cv2.CV_64F))
            if noise_level > 50:
                issues.append(f'High noise level: {noise_level:.2f}')
            
            return {
                'filename': image_path.name,
                'status': 'PASSED' if len(issues) == 0 else 'FAILED',
                'resolution': f'{width}x{height}',
                'blur_score': f'{blur_score:.2f}',
                'mean_intensity': f'{mean_intensity:.2f}',
                'std_intensity': f'{std_intensity:.2f}',
                'noise_level': f'{noise_level:.2f}',
                'issues': '; '.join(issues) if issues else 'None',
                'pass_qc': len(issues) == 0
            }
            
        except Exception as e:
            return {
                'filename': image_path.name,
                'status': 'ERROR',
                'reason': str(e),
                'pass_qc': False
            }
    
    def detect_duplicates(self, image_paths):
        """Detect exact and near duplicates using perceptual hashing"""
        print("Detecting duplicates...")
        hash_dict = {}
        
        for img_path in image_paths:
            try:
                # Calculate perceptual hash
                img = Image.open(img_path)
                img_hash = imagehash.phash(img)
                
                # Check for exact duplicates
                hash_str = str(img_hash)
                if hash_str in hash_dict:
                    self.duplicates[hash_str].append(img_path.name)
                else:
                    hash_dict[hash_str] = img_path.name
                
                # Check for near duplicates (Hamming distance <= 5)
                for existing_hash, existing_file in hash_dict.items():
                    if existing_hash != hash_str:
                        distance = imagehash.hex_to_hash(existing_hash) - img_hash
                        if distance <= 5 and distance > 0:
                            self.near_duplicates[existing_file].append({
                                'duplicate': img_path.name,
                                'distance': distance
                            })
                            
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        return hash_dict
    
    def run_quality_control(self):
        """Run complete quality control pipeline"""
        print(f"Starting quality control on images in: {self.images_dir}")
        
        # Get all image files
        image_extensions = {'.jpg'}
        image_paths = [p for p in self.images_dir.rglob('*') 
                      if p.suffix.lower() in image_extensions]
        
        if self.max_images:
            image_paths = image_paths[:self.max_images]
            print(f"** TESTING MODE: Limited to first {self.max_images} images **")
        
        total_images = len(image_paths)
        print(f"Found {total_images} images")
        
        if total_images == 0:
            print("No images found!")
            return
        
        # Step 1: Quality Assessment
        print("\nStep 1: Assessing image quality...")
        passed_image_paths = []

        for i, img_path in enumerate(image_paths, 1):
            if i % 500 == 0:
                print(f"  Processed {i}/{total_images} images...")
            
            quality_result = self.assess_image_quality(img_path)
            self.quality_report.append(quality_result)
            
            if not quality_result['pass_qc']:
                self.filtered_images.append(quality_result)
            else:
                passed_image_paths.append(img_path)

        print(f"\nCopying {len(passed_image_paths)} passed images to {self.passed_images_dir}...")
        import shutil
        for img_path in passed_image_paths:
            dest_path = self.passed_images_dir / img_path.name
            shutil.copy2(img_path, dest_path)
        
        # Step 2: Duplicate Detection
        print("\nStep 2: Detecting duplicates...")
        self.detect_duplicates(image_paths)
        
        # Generate report
        self.generate_report()
        
    def generate_report(self):
        """Generate comprehensive quality control report"""
        print("\n" + "="*60)
        print("QUALITY CONTROL REPORT")
        print("="*60)
        
        # Convert to DataFrame
        df_quality = pd.DataFrame(self.quality_report)
        
        # Summary statistics
        total_images = len(df_quality)
        passed_images = df_quality['pass_qc'].sum()
        failed_images = total_images - passed_images
        
        print(f"\nTotal Images Processed: {total_images}")
        print(f"Passed QC: {passed_images} ({passed_images/total_images*100:.2f}%)")
        print(f"Failed QC: {failed_images} ({failed_images/total_images*100:.2f}%)")
        
        # Failure reasons breakdown
        if failed_images > 0:
            print(f"\nFailure Reasons:")
            failed_df = df_quality[df_quality['pass_qc'] == False]
            
            # Count each type of issue
            issue_counts = defaultdict(int)
            for issues_str in failed_df['issues']:
                if pd.notna(issues_str):
                    for issue in issues_str.split('; '):
                        issue_type = issue.split(':')[0].strip()
                        issue_counts[issue_type] += 1
            
            for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                print(f"  - {issue}: {count} ({count/total_images*100:.2f}%)")
        
        # Duplicate detection results
        exact_dupes = sum(len(v) for v in self.duplicates.values())
        near_dupes = sum(len(v) for v in self.near_duplicates.values())
        
        print(f"\nDuplicate Detection:")
        print(f"  - Exact duplicates: {exact_dupes}")
        print(f"  - Near-duplicates (for review): {near_dupes}")

        print(f"\nPassed Images:")
        print(f"  - Copied to: {self.passed_images_dir}")
        print(f"  - Total passed: {passed_images}")
        
        # Save detailed reports
        output_path = self.output_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_quality.to_csv(output_path, index=False)
        print(f"\nDetailed quality report saved to: {output_path}")
        
        # Save filtered images list
        if self.filtered_images:
            filtered_path = self.output_dir / "filtered_images.csv"
            pd.DataFrame(self.filtered_images).to_csv(filtered_path, index=False)
            print(f"Filtered images list saved to: {filtered_path}")
        
        # Save duplicates report
        if self.duplicates or self.near_duplicates:
            duplicates_report = {
                'exact_duplicates': {k: v for k, v in self.duplicates.items()},
                'near_duplicates': {k: v for k, v in self.near_duplicates.items()}
            }
            dup_path = self.output_dir / "duplicates_report.json"
            with open(dup_path, 'w') as f:
                json.dump(duplicates_report, f, indent=2)
            print(f"Duplicates report saved to: {dup_path}")
        
        print("="*60)
        
        return df_quality


if __name__ == "__main__":
    # ====== CONFIGURATION ======
    # Path to the fundus_photos directory
    IMAGES_DIR = r"C:\Users\omara\Downloads\BRSet\physionet.org\files\brazilian-ophthalmological\1.0.1\fundus_photos"
    
    # Where to save quality control reports
    OUTPUT_DIR = "qc_output"
    
    # Minimum resolution requirement
    MIN_RESOLUTION = 512
    # ================================================

    # For full run: set to None
    MAX_IMAGES = 100  # Change to None for full dataset
    
    # Run quality control
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Minimum resolution: {MIN_RESOLUTION}x{MIN_RESOLUTION}\n")
    print(f"Max images to process: {MAX_IMAGES if MAX_IMAGES else 'All'}\n")

    
    qc = ImageQualityController(IMAGES_DIR, OUTPUT_DIR, MIN_RESOLUTION, MAX_IMAGES)
    qc.run_quality_control()