import pandas as pd
from pathlib import Path
import os
import shutil

def normalize_age(age):
    if pd.isna(age):
        return None
    # Assuming max age around 100
    try:
        return float(age) / 100.0
    except (ValueError, TypeError):
        return None

def merge_metadata():
    # Paths
    project_dir = Path(__file__).resolve().parents[3]
    github_dir = project_dir.parent
    
    brset_adeq_csv = github_dir / "adequate_brset_images" / "labels_brset_adequate.csv"
    odir_proc_csv = project_dir / "tasks" / "hypertension" / "dataprep" / "odir" / "processed" / "metadata_processed.csv"
    
    out_dir = project_dir / "data" / "hypertension_combined" / "metadata_processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "combined_metadata.csv"
    
    records = []
    
    # --- Process BRSET ---
    print(f"Loading BRSET from {brset_adeq_csv}")
    if brset_adeq_csv.exists():
        brset_df = pd.read_csv(brset_adeq_csv)
        print(f"BRSET columns: {list(brset_df.columns)}")
        
        # BRSET derives hypertension from 'comorbidities'
        # Or from 'hypertension' if already present (some scripts output 'hypertension')
        for _, row in brset_df.iterrows():
            # Original ID logic
            orig_image_id = str(row.get("image_id", ""))
            orig_patient_id = str(row.get("patient_id", orig_image_id))
            
            image_id = f"brset_{orig_image_id}"
            patient_id = f"brset_{orig_patient_id}"
            image_filename = f"{orig_image_id}.jpg" if not orig_image_id.endswith(".jpg") else orig_image_id
            
            # Label
            label = 0
            if "comorbidities" in row and pd.notna(row["comorbidities"]):
                if "hypertension" in str(row["comorbidities"]).lower():
                    label = 1
            elif "hypertension" in row and pd.notna(row["hypertension"]):
                label = int(row["hypertension"])
                
            age = row.get("patient_age", row.get("age", None))
            gender = row.get("patient_sex", row.get("sex", row.get("gender", None)))
            
            records.append({
                "image_id": image_id,
                "patient_id": patient_id,
                "label": float(label),
                "source_dataset": "brset",
                "image_filename": image_filename,
                "relative_image_path": f"brset/{image_filename}",
                "eye": None, # BRSET doesn't always specify eye reliably in this subset, or we can leave None
                "age": age,
                "age_normalized": normalize_age(age),
                "gender": gender
            })
    else:
        print(f"Warning: BRSET CSV not found at {brset_adeq_csv}")
        
    # --- Process ODIR ---
    print(f"Loading ODIR from {odir_proc_csv}")
    if odir_proc_csv.exists():
        odir_df = pd.read_csv(odir_proc_csv)
        print(f"ODIR columns: {list(odir_df.columns)}")
        for _, row in odir_df.iterrows():
            orig_image_id = str(row.get("image_id", ""))
            orig_patient_id = str(row.get("patient_id", ""))
            
            image_id = f"odir_{orig_image_id}"
            patient_id = f"odir_{orig_patient_id}"
            image_filename = str(row.get("image_filename", ""))
            
            label = float(row.get("label", 0))
            age = row.get("age", None)
            gender = row.get("gender", None)
            eye = row.get("eye", None)
            
            records.append({
                "image_id": image_id,
                "patient_id": patient_id,
                "label": label,
                "source_dataset": "odir",
                "image_filename": image_filename,
                "relative_image_path": f"odir/{image_filename}",
                "eye": eye,
                "age": age,
                "age_normalized": normalize_age(age),
                "gender": gender
            })
    else:
        print(f"Warning: ODIR CSV not found at {odir_proc_csv}")
        
    # --- Combine and Save ---
    combined_df = pd.DataFrame(records)
    print(f"\nCombined dataset size: {len(combined_df)}")
    if not combined_df.empty:
        print(f"Label distribution:\n{combined_df['label'].value_counts()}")
        print(f"Source distribution:\n{combined_df['source_dataset'].value_counts()}")
        combined_df.to_csv(out_csv, index=False)
        print(f"Saved combined metadata to {out_csv}")
    else:
        print("No records to save.")

if __name__ == "__main__":
    merge_metadata()
