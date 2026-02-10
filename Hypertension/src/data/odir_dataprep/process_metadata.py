import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "raw" / "ODIR-5K" / "brazilian-ophthalmological" / "1.0.1"
OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_CSV = RAW_DIR / "labels_brset.csv"
OUTPUT_CSV = OUTPUT_DIR / "metadata_processed.csv"


# -------------------------
# Helpers
# -------------------------
def map_eye(exam_eye):
    if exam_eye == 1:
        return "R"
    elif exam_eye == 2:
        return "L"
    else:
        return None


def normalize_age(age):
    if pd.isna(age):
        return None
    return age / 100.0


# -------------------------
# Main processing
# -------------------------
def process_metadata():
    print("Loading BRSET metadata...")
    df = pd.read_csv(LABELS_CSV)

    records = []

    for _, row in df.iterrows():
        eye = map_eye(row["exam_eye"])
        if eye is None:
            continue

        image_id = row["image_id"]

        records.append({
            "patient_id": row["patient_id"],
            "image_id": f"{image_id}_{eye}",
            "image_filename": f"{image_id}",
            "eye": eye,
            "label": int(row["hypertensive_retinopathy"]),
            "gender": row["patient_sex"],
            "age": row["patient_age"],
            "age_normalized": normalize_age(row["patient_age"]),
            "split": ""  # intentionally empty (training phase later)
        })

    processed_df = pd.DataFrame(records)

    processed_df.to_csv(OUTPUT_CSV, index=False)

    print("Metadata processing complete.")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total images: {len(processed_df)}")
    print("Label distribution:")
    print(processed_df["label"].value_counts())


if __name__ == "__main__":
    process_metadata()