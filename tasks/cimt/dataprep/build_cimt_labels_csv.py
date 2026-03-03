"""
Convert CIMT data_info.json to pipeline-ready labels CSV.

Input:  data_info.json — dict keyed by patient_id, each entry has
        left_eye/right_eye filenames and metadata.
Output: labels.csv — one row per eye image, columns match TrainConfig fields.

Run once, commit the CSV (or regenerate from the checked-in JSON).

Usage:
    python tasks/cimt/dataprep/build_cimt_labels_csv.py \
        --json data/cimt/data_info.json \
        --image-dir "data/cimt/Fundus_CIMT_2903 Dataset" \
        --out data/cimt/labels.csv \
        --strict-missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def build_labels_csv(
    json_path: Path,
    image_dir: Path,
    strict_missing: bool = False,
) -> pd.DataFrame:
    """Parse data_info.json into a flat DataFrame.

    The JSON is a dict keyed by patient_id:
        {
            "2491006": {
                "gender": 0, "thickness": 0.8, "label": 0,
                "group": 1, "True_age": 63, "age": 0.684...,
                "right_eye": "2491006_R.png", "left_eye": "2491006_L.png"
            },
            ...
        }

    Each patient has two eyes. Each eye becomes one row with:
        image_id, filename, patient_id, eye, label,
        thickness, gender, group, age

    The label field is binary: 1 = thickened CIMT, 0 = normal.
    """
    with open(json_path) as f:
        data = json.load(f)

    records = []
    missing = []

    for patient_id_key, patient in data.items():
        patient_id = str(patient_id_key)
        label = int(patient["label"])
        thickness = patient.get("thickness")
        gender = patient.get("gender")
        group = patient.get("group")
        age = patient.get("True_age") or patient.get("age")

        for eye_key, eye_label in [("left_eye", "L"), ("right_eye", "R")]:
            filename = patient.get(eye_key)
            if filename is None:
                continue

            full_path = image_dir / filename
            if not full_path.exists():
                missing.append(filename)
                if strict_missing:
                    continue

            image_id = Path(filename).stem

            records.append({
                "image_id": image_id,
                "filename": filename,
                "patient_id": patient_id,
                "eye": eye_label,
                "label": label,
                "thickness": thickness,
                "gender": gender,
                "group": group,
                "age": age,
            })

    if missing:
        print(f"WARNING: {len(missing)} image files not found on disk.")
        if strict_missing:
            print(f"  Skipped these (--strict-missing). First 5: {missing[:5]}")
        else:
            print(f"  Included anyway. First 5: {missing[:5]}")

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Convert CIMT JSON to labels CSV")
    parser.add_argument("--json", required=True, help="Path to data_info.json")
    parser.add_argument("--image-dir", required=True, help="Path to image directory")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--strict-missing", action="store_true",
        help="Skip rows where image file is missing on disk",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    image_dir = Path(args.image_dir)
    out_path = Path(args.out)

    if not json_path.exists():
        print(f"ERROR: JSON not found: {json_path}")
        sys.exit(1)
    if not image_dir.exists():
        print(f"ERROR: Image directory not found: {image_dir}")
        sys.exit(1)

    df = build_labels_csv(json_path, image_dir, strict_missing=args.strict_missing)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Labels CSV written: {out_path}")
    print(f"  Rows:     {len(df)}")
    print(f"  Patients: {df['patient_id'].nunique()}")
    print(f"  Label distribution:")
    print(f"    Positive: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.1%})")
    print(f"    Negative: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.1%})")


if __name__ == "__main__":
    main()
