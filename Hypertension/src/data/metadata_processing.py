"""
Metadata Processing Script
Flattens and standardizes labels.csv to required format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

class MetadataProcessor:
    def __init__(self, labels_csv_path: str, output_path: str, 
                 standardized_images_dir: str = None):
        """
        Initialize metadata processor.
        
        Args:
            labels_csv_path: Path to original labels.csv
            output_path: Path to save processed metadata CSV
            standardized_images_dir: Optional - if provided, only includes images that exist in this directory
        """
        self.labels_csv_path = Path(labels_csv_path)
        self.output_path = Path(output_path)
        self.standardized_images_dir = Path(standardized_images_dir) if standardized_images_dir else None
        
    def check_hypertension_in_comorbidities(self, comorbidities_str):
        """
        Check if 'hypertension' appears in the comorbidities string.
        Returns 1 if found, 0 otherwise.
        """
        if pd.isna(comorbidities_str):
            return 0
        
        # Convert to string and lowercase for case-insensitive matching
        comorbidities_str = str(comorbidities_str).lower()
        
        # Check if 'hypertension' is in the string
        if 'hypertension' in comorbidities_str:
            return 1
        else:
            return 0
    
    def normalize_age(self, age_series):
        """
        Normalize age to 0-1 range using min-max normalization.
        Formula: (age - min_age) / (max_age - min_age)
        """
        min_age = age_series.min()
        max_age = age_series.max()
        
        # Avoid division by zero
        if max_age == min_age:
            return pd.Series([0.5] * len(age_series))
        
        normalized = (age_series - min_age) / (max_age - min_age)
        return normalized
    
    def process_metadata(self):
        """Process labels.csv into standardized metadata format."""
        
        print(f"Reading labels from: {self.labels_csv_path}")
        df = pd.read_csv(self.labels_csv_path)
        
        print(f"Original dataset: {len(df)} rows")
        
        # Create new dataframe with required columns
        metadata = pd.DataFrame()
        
        # 1. patient_id (string, same as in CSV)
        metadata['patient_id'] = df['patient_id'].astype(str)
        
        # 2. image_id (string, same as in CSV)
        metadata['image_id'] = df['image_id'].astype(str)
        
        # 3. image_filename (string, image_id + .jpg)
        metadata['image_filename'] = metadata['image_id'] + '.jpg'
        
        # 4. eye (string, exam_eye: 1=L, 2=R)
        metadata['eye'] = df['exam_eye'].map({1: 'L', 2: 'R'})
        
        # 5. hypertension (int, 1 if 'hypertension' in comorbidities string, 0 otherwise)
        metadata['hypertension'] = df['comorbidities'].apply(
            self.check_hypertension_in_comorbidities
        )
        
        # 6. hypertensive_retinopathy (save as is)
        metadata['hypertensive_retinopathy'] = df['hypertensive_retinopathy']
        
        # 7. gender (named patient_sex, save as is)
        metadata['gender'] = df['patient_sex']
        
        # 8. age (named patient_age, save as is)
        metadata['age'] = df['patient_age']
        
        # 9. age_normalized (normalized age value 0-1)
        metadata['age_normalized'] = self.normalize_age(df['patient_age'])
        
        # 10. vascular_occlusion (save as is)
        metadata['vascular_occlusion'] = df['vascular_occlusion']
        
        # 11. split (empty column for now)
        metadata['split'] = ''
        
        # Filter to only include images that exist in standardized_images_dir
        if self.standardized_images_dir:
            print(f"\nFiltering to only include images in: {self.standardized_images_dir}")
            
            # Get list of standardized image filenames
            standardized_files = set()
            for ext in ['.jpg', '.jpeg', '.png']:
                standardized_files.update(
                    f.name for f in self.standardized_images_dir.glob(f'*{ext}')
                )
            
            print(f"Found {len(standardized_files)} images in standardized directory")
            
            # Filter metadata
            before_filter = len(metadata)
            metadata = metadata[metadata['image_filename'].isin(standardized_files)]
            after_filter = len(metadata)
            
            print(f"Filtered: {before_filter} → {after_filter} rows ({before_filter - after_filter} removed)")
        
        # Save processed metadata
        metadata.to_csv(self.output_path, index=False)
        print(f"\nProcessed metadata saved to: {self.output_path}")
        
        # Print summary statistics
        self.print_summary(metadata)
        
        return metadata
    
    def print_summary(self, metadata):
        """Print summary statistics of processed metadata."""
        print("\n" + "="*60)
        print("METADATA PROCESSING SUMMARY")
        print("="*60)
        
        print(f"\nTotal Records: {len(metadata)}")
        
        print(f"\nUnique Patients: {metadata['patient_id'].nunique()}")
        
        print(f"\nEye Distribution:")
        print(metadata['eye'].value_counts())
        
        print(f"\nHypertension Distribution:")
        print(f"  No hypertension (0): {(metadata['hypertension'] == 0).sum()}")
        print(f"  Has hypertension (1): {(metadata['hypertension'] == 1).sum()}")
        
        print(f"\nHypertensive Retinopathy Distribution:")
        print(f"  No (0): {(metadata['hypertensive_retinopathy'] == 0).sum()}")
        print(f"  Yes (1): {(metadata['hypertensive_retinopathy'] == 1).sum()}")
        
        print(f"\nGender Distribution:")
        print(metadata['gender'].value_counts())
        
        print(f"\nAge Statistics:")
        print(f"  Min: {metadata['age'].min()}")
        print(f"  Max: {metadata['age'].max()}")
        print(f"  Mean: {metadata['age'].mean():.2f}")
        print(f"  Median: {metadata['age'].median():.2f}")
        
        print(f"\nVascular Occlusion Distribution:")
        print(f"  No (0): {(metadata['vascular_occlusion'] == 0).sum()}")
        print(f"  Yes (1): {(metadata['vascular_occlusion'] == 1).sum()}")
        
        print("="*60)


if __name__ == "__main__":
    # ====== CONFIGURATION - UPDATE THESE ======
    LABELS_CSV = r"C:\Users\omara\Downloads\BRSet\physionet.org\files\brazilian-ophthalmological\1.0.1\labels_brset.csv"
    OUTPUT_CSV = "processed_metadata.csv"
    
    # Optional: Only include images that made it through standardization
    # Set to None to include all images from labels.csv
    STANDARDIZED_IMAGES_DIR = r"C:\Users\omara\Project-Retinexis\standardized_images" 
    # ==========================================
    
    processor = MetadataProcessor(
        labels_csv_path=LABELS_CSV,
        output_path=OUTPUT_CSV,
        standardized_images_dir=STANDARDIZED_IMAGES_DIR
    )
    
    metadata = processor.process_metadata()
    
    print(f"\n✓ Done! Processed metadata saved to: {OUTPUT_CSV}")
    print(f"✓ Ready to use for model training")