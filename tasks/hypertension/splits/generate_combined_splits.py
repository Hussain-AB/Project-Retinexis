import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def generate_splits():
    project_dir = Path(__file__).resolve().parents[3]
    combined_csv = project_dir / "data" / "hypertension_combined" / "metadata_processed" / "combined_metadata.csv"
    
    out_dir = project_dir / "tasks" / "hypertension" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "hypertension_combined_5fold_splits.json"
    
    print(f"Loading {combined_csv}")
    df = pd.read_csv(combined_csv)
    
    # Needs a sample_id to track images uniquely
    df["sample_id"] = df["image_id"]
    
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, df["label"], groups=df["patient_id"])):
        train_samples = df.iloc[train_idx]["sample_id"].tolist()
        val_samples = df.iloc[val_idx]["sample_id"].tolist()
        
        folds.append({
            "fold": fold_idx,
            "train_sample_ids": train_samples,
            "val_sample_ids": val_samples
        })
        
    split_space = {
        "metadata": {
            "dataset_name": "hypertension_combined",
            "n_folds": n_splits,
            "sample_id_col": "image_id",
            "group_col": "patient_id",
            "label_col": "label"
        },
        "folds": folds
    }
    
    with open(out_json, "w") as f:
        json.dump(split_space, f, indent=2)
        
    print(f"Saved {n_splits}-fold splits to {out_json}")
    
    # Optional Validation Print
    for fold in folds:
        print(f"Fold {fold['fold']}: Train {len(fold['train_sample_ids'])}, Val {len(fold['val_sample_ids'])}")

if __name__ == "__main__":
    generate_splits()
