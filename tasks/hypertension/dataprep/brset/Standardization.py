"""
Image Standardization for Retinal Fundus Images
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import json
from datetime import datetime
from tqdm import tqdm
import re

class FundusImageStandardizer:
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 target_size: int = 512,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: int = 8,
                 enhance_vessels: bool = False,
                 use_green_channel: bool = False,
                 blur_threshold: float = 100.0):
        """
        Initialize the standardizer.
        
        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save standardized images
            target_size: Target image size (512 or 224)
            clahe_clip_limit: CLAHE clip limit (default 2.0)
            clahe_tile_size: CLAHE tile grid size (default 8x8)
            enhance_vessels: Whether to apply vessel enhancement
            use_green_channel: Whether to emphasize green channel
            blur_threshold: Minimum blur score to pass (default 100, higher = sharper required)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.enhance_vessels = enhance_vessels
        self.use_green_channel = use_green_channel
        
        # Processing log
        self.processing_log = []

        # Blur threshold for quality filtering
        self.blur_threshold = 100  # Default, can be configured

    def calculate_blur(self, image: np.ndarray) -> float:
        """Calculate Laplacian variance to detect blur. Higher = sharper."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
        
    def detect_and_remove_text(self, image: np.ndarray) -> np.ndarray:
        """
        Detect "31床" text or any text in top-left corner and black it out.
        Searches top 20% height, left 30% width of image.
        """
        height, width = image.shape[:2]
        
        # Define top-left region
        search_height = int(height * 0.20)
        search_width = int(width * 0.30)
        
        # Create mask for region with potential text
        mask = np.ones_like(image)
        
        # Convert region to grayscale for text detection
        top_left = image[0:search_height, 0:search_width]
        gray = cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding to detect text
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.THRESH_BINARY_INV,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2
        )
        
        # Find contours (potential text regions)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Text-like regions: reasonable size and aspect ratio
            if 50 < area < 5000 and 0.1 < h/w < 10:
                # Add some padding
                x = max(0, x - 5)
                y = max(0, y - 5)
                w = min(search_width - x, w + 10)
                h = min(search_height - y, h + 10)
                text_regions.append((x, y, w, h))
        
        # Black out detected text regions
        result = image.copy()
        for x, y, w, h in text_regions:
            result[y:y+h, x:x+w] = 0
        
        return result
    
    def resize_with_aspect_ratio(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize image to target_size while maintaining aspect ratio.
        Uses center cropping if image is larger, padding if smaller.
        """
        height, width = image.shape[:2]
        
        # Calculate scaling to fit the target size
        scale = target_size / min(height, width)
        
        # Resize maintaining aspect ratio
        if scale < 1:  # Downsampling
            interpolation = cv2.INTER_CUBIC
        else:  # Upsampling
            interpolation = cv2.INTER_LINEAR
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        # Center crop or pad to exactly target_size x target_size
        if new_height > target_size or new_width > target_size:
            # Center crop
            start_y = (new_height - target_size) // 2
            start_x = (new_width - target_size) // 2
            result = resized[start_y:start_y+target_size, start_x:start_x+target_size]
        else:
            # Center pad
            result = np.zeros((target_size, target_size, 3), dtype=image.dtype)
            start_y = (target_size - new_height) // 2
            start_x = (target_size - new_width) // 2
            result[start_y:start_y+new_height, start_x:start_x+new_width] = resized
        
        return result
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE.
        Applies to each channel separately.
        """
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )
        l_clahe = clahe.apply(l)
        
        # Merge and convert back
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return result
    
    def normalize_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using mean/std normalization per channel.
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Per-channel mean/std normalization
        mean = np.mean(img_float, axis=(0, 1))
        std = np.std(img_float, axis=(0, 1)) + 1e-7  # Avoid division by zero
        result = (img_float - mean) / std
        
        return result
    
    def apply_green_channel_emphasis(self, image: np.ndarray) -> np.ndarray:
        """
        Emphasize green channel as it has highest contrast in fundus images.
        """
        # Split channels (BGR)
        b, g, r = cv2.split(image)
        
        # Enhance green channel
        g_enhanced = cv2.equalizeHist(g)
        
        # Option 1: Replace green channel
        result = cv2.merge([b, g_enhanced, r])
        
        # Option 2: Use green channel only (convert to 3-channel for consistency)
        #result = cv2.merge([g_enhanced, g_enhanced, g_enhanced])
        
        return result
    
    def apply_vessel_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance vascular structures using morphological operations and filtering.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Top-hat transform to enhance vessels
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Black-hat transform
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine
        enhanced = cv2.add(gray, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
        
        # Convert back to 3-channel
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def process_image(self, image_path: Path) -> Tuple[Optional[np.ndarray], dict]:
        """
        Apply complete standardization pipeline to a single image.
        
        Returns:
            Tuple of (processed_image, metadata)
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None, {'status': 'error', 'reason': 'Cannot read image'}
            
            original_shape = image.shape
            
            # Step 1: Resize with aspect ratio preservation
            image = self.resize_with_aspect_ratio(image, self.target_size)
            
            # Step 2: Apply CLAHE for contrast enhancement
            image = self.apply_clahe(image)
            
            # Step 3: Optional - Green channel emphasis
            if self.use_green_channel:
                image = self.apply_green_channel_emphasis(image)
            
            # Step 4: Optional - Vessel enhancement
            if self.enhance_vessels:
                image = self.apply_vessel_enhancement(image)
            
            # Step 5: Normalize (this should be last for training)
            # Note: Normalization to float is usually done just before model input
            # For now, keep in uint8 for saving
            # image_normalized = self.normalize_channels(image)

            # Step 6: Quality check - Blur detection
            blur_score = self.calculate_blur(image)
            passes_blur_check = blur_score >= self.blur_threshold
            
            if not passes_blur_check:
                return None, {
                    'filename': image_path.name,
                    'status': 'filtered',
                    'reason': f'Blur score {blur_score:.2f} below threshold {self.blur_threshold}',
                    'blur_score': blur_score
                }
            
            metadata = {
                'filename': image_path.name,
                'status': 'success',
                'original_shape': f"{original_shape[1]}x{original_shape[0]}",
                'final_shape': f"{self.target_size}x{self.target_size}",
                'blur_score': f"{blur_score:.2f}",
                'clahe_applied': True,
                'green_channel': self.use_green_channel,
                'vessel_enhanced': self.enhance_vessels
            }
            
            return image, metadata
            
        except Exception as e:
            return None, {'status': 'error', 'reason': str(e)}
    
    def standardize_dataset(self, file_list: Optional[list] = None):
        """
        Process entire dataset.
        
        Args:
            file_list: Optional list of specific files to process. 
                      If None, processes all images in input_dir.
        """
        print(f"Starting standardization pipeline")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target size: {self.target_size}x{self.target_size}")
        print(f"CLAHE: clip_limit={self.clahe_clip_limit}, tile_size={self.clahe_tile_size}x{self.clahe_tile_size}")
        print(f"Green channel emphasis: {self.use_green_channel}")
        print(f"Vessel enhancement: {self.enhance_vessels}")
        print(f"Blur threshold: {self.blur_threshold} (images below this will be filtered)")
        print()
        
        # Get list of images to process
        if file_list is None:
            image_extensions = {'.jpg'}
            image_paths = [p for p in self.input_dir.rglob('*') 
                          if p.suffix.lower() in image_extensions]
        else:
            image_paths = [self.input_dir / f for f in file_list]
        
        total_images = len(image_paths)
        print(f"Found {total_images} images to process\n")
        
        if total_images == 0:
            print("No images found!")
            return
        
        # Process images
        successful = 0
        failed = 0
        filtered = 0
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            processed_img, metadata = self.process_image(img_path)
            
            if processed_img is not None:
                # Save processed image
                output_path = self.output_dir / img_path.name
                cv2.imwrite(str(output_path), processed_img)
                successful += 1
            elif metadata.get('status') == 'filtered':
                # Image was processed but filtered due to blur
                filtered += 1
            else:
                # Processing error
                failed += 1
            
            self.processing_log.append(metadata)
        
        # Generate report
        self.generate_report(successful, filtered, failed, total_images)
    
    def generate_report(self, successful: int, filtered: int, failed: int, total: int):
        """Generate standardization report."""
        print("\n" + "="*60)
        print("STANDARDIZATION REPORT")
        print("="*60)
        print(f"\nTotal Images: {total}")
        print(f"Successfully Processed: {successful} ({successful/total*100:.2f}%)")
        print(f"Filtered (blur too low): {filtered} ({filtered/total*100:.2f}%)")
        print(f"Failed: {failed} ({failed/total*100:.2f}%)")
        
        # Save detailed log
        log_path = self.output_dir / f"standardization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_log = pd.DataFrame(self.processing_log)
        df_log.to_csv(log_path, index=False)
        print(f"\nDetailed log saved to: {log_path}")
        
        # Save configuration
        config = {
            'target_size': self.target_size,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_size': self.clahe_tile_size,
            'enhance_vessels': self.enhance_vessels,
            'use_green_channel': self.use_green_channel,
            'blur_threshold': self.blur_threshold,
            'timestamp': datetime.now().isoformat()
        }
        config_path = self.output_dir / "standardization_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {config_path}")
        
        print("="*60)


if __name__ == "__main__":
    # ====== CONFIGURATION ======
    INPUT_DIR = r"C:\Users\omara\Project-Retinexis\qc_output\passed_images"
    OUTPUT_DIR = "standardized_images"  # UPDATE THIS
    
    # Image size: 512 or 224
    TARGET_SIZE = 512
    
    # CLAHE parameters
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = 8

    # Blur threshold for quality filtering (higher = stricter)
    # Typical values: 50-200 for fundus images after standardization
    BLUR_THRESHOLD = 100.0
    
    # Optional enhancements
    USE_GREEN_CHANNEL = True  # Set True to emphasize green channel
    ENHANCE_VESSELS = False    # Set True to enhance vascular structures
    
    # For testing: specify files or set to None for all
    TEST_FILES = None # e.g., ['img00001.jpg', 'img00002.jpg'] or None
    # ==========================================
    
    standardizer = FundusImageStandardizer(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE,
        clahe_clip_limit=CLAHE_CLIP_LIMIT,
        clahe_tile_size=CLAHE_TILE_SIZE,
        enhance_vessels=ENHANCE_VESSELS,
        use_green_channel=USE_GREEN_CHANNEL,
        blur_threshold=BLUR_THRESHOLD
    )
    
    standardizer.standardize_dataset(file_list=TEST_FILES)