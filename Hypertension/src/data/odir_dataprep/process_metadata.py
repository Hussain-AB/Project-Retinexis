import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------
DATA_DIR = Path(__file__).resolve().parents[1]  # -> data/
ODIR_DIR = DATA_DIR / "ODIR-5K"
TRAIN_DIR = ODIR_DIR / "Training Images"
TEST_DIR = ODIR_DIR / "Testing Images"
METADATA_XLSX = ODIR_DIR / "data.xlsx"

OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "metadata_processed.csv"


# -------------------------
# Helpers
# -------------------------
def derive_eye(filename):
    if "_left" in filename:
        return "L"
    elif "_right" in filename:
        return "R"
    return None


def derive_subset(filename):
    if (TRAIN_DIR / filename).exists():
        return "train"
    elif (TEST_DIR / filename).exists():
        return "test"
    return ""


def normalize_age(age):
    if pd.isna(age):
        return None
    return age / 100.0


# -------------------------
# Main processing
# -------------------------
def process_metadata():
    print("Loading Kaggle ODIR-5K metadata...")

    if not METADATA_XLSX.exists():
        raise FileNotFoundError(f"data.xlsx not found at: {METADATA_XLSX}")

    df = pd.read_excel(METADATA_XLSX)

    records = []

    for _, row in df.iterrows():
        patient_id = int(row["ID"])
        label = int(row["H"])
        age = row["Patient Age"]
        gender = row["Patient Sex"]

        # Each patient row has left + right fundus images
        for fundus_col, eye_expected in [("Left-Fundus", "L"), ("Right-Fundus", "R")]:
            image_filename = str(row[fundus_col])
            eye = derive_eye(image_filename)
            if eye is None:
                eye = eye_expected

            image_id = Path(image_filename).stem
            subset = derive_subset(image_filename)

            records.append({
                "patient_id": patient_id,
                "image_id": image_id,
                "image_filename": image_filename,
                "eye": eye,
                "label": label,
                "gender": gender,
                "age": age,
                "age_normalized": normalize_age(age),
                "subset": subset,
            })

    processed_df = pd.DataFrame(records)

    assert processed_df["image_filename"].str.endswith(".jpg").all()

    # Patient split leakage check
    train_patients = set(processed_df.loc[processed_df["subset"] == "train", "patient_id"])
    test_patients = set(processed_df.loc[processed_df["subset"] == "test", "patient_id"])
    leaked = train_patients & test_patients
    if leaked:
        print(f"WARNING: {len(leaked)} patient(s) appear in BOTH train and test: {sorted(leaked)[:10]}")

    processed_df.to_csv(OUTPUT_CSV, index=False)

    print("Metadata processing complete.")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total images: {len(processed_df)}")
    print(f"Patients: {processed_df['patient_id'].nunique()}")
    print("Label distribution:")
    print(processed_df["label"].value_counts())
    print("Subset distribution:")
    print(processed_df["subset"].value_counts())


if __name__ == "__main__":
    process_metadata()
