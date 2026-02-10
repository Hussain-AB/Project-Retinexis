import pandas as pd
from pathlib import Path

# -------------------------
# Paths
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]  # Project-Retinexis
RAW_DIR = REPO_ROOT / "raw" / "ODIR-5K" / "brazilian-ophthalmological" / "1.0.1"
LABELS_CSV = RAW_DIR / "labels_brset.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels CSV not found at: {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)

    records = []

    for _, row in df.iterrows():
        eye = map_eye(row["exam_eye"])
        if eye is None:
            continue

        image_id = str(row["image_id"])

        # filenames are like img16266.jpg
        image_filename = (
            image_id if image_id.lower().endswith(".jpg")
            else f"{image_id}.jpg"
        )

        records.append({
            "patient_id": row["patient_id"],
            "image_id": image_id,              # matches filename stem
            "image_filename": image_filename,  # must exactly match disk
            "eye": eye,
            "label": int(row["hypertensive_retinopathy"]),
            "gender": row["patient_sex"],
            "age": row["patient_age"],
            "age_normalized": normalize_age(row["patient_age"]),
            "split": ""
        })

    processed_df = pd.DataFrame(records)

    assert processed_df["image_filename"].str.endswith(".jpg").all()

    processed_df.to_csv(OUTPUT_CSV, index=False)

    print("Metadata processing complete.")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total images: {len(processed_df)}")
    print("Label distribution:")
    print(processed_df["label"].value_counts())


if __name__ == "__main__":
    process_metadata()