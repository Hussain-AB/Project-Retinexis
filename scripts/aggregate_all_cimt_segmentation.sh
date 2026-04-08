#!/usr/bin/env bash
# Aggregate CV fold results for all segmentation models in outputs_segmentation/cimt,
# then combine into a single summary CSV.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CIMT_DIR="$PROJECT_ROOT/outputs_segmentation/cimt"
CIMT_MULTIMODAL_DIR="$PROJECT_ROOT/outputs_segmentation_multimodal/cimt"

# --- 1) Run per-model aggregation ---
for model_dir in "$CIMT_DIR"/*/ "$CIMT_MULTIMODAL_DIR"/*/; do
    [ -d "$model_dir" ] || continue

    model_name=$(basename "$model_dir")
    echo "=============================================="
    echo "  Aggregating: $model_name"
    echo "=============================================="

    python "$SCRIPT_DIR/aggregate_folds.py" \
        --run-dir "$model_dir" \
        --n-folds 5
done

# --- 2) Combine all cv_summary.csv into one table ---
OUTPUT_CSV="$CIMT_DIR/all_models_summary.csv"

python - "$CIMT_DIR" "$CIMT_MULTIMODAL_DIR" "$OUTPUT_CSV" <<'PYEOF'
import sys, pathlib, numpy as np, pandas as pd

cimt_dir = pathlib.Path(sys.argv[1])
cimt_multimodal_dir = pathlib.Path(sys.argv[2])
output_csv = pathlib.Path(sys.argv[3])

METRICS = ["auroc", "auprc", "ece", "f1", "sensitivity", "specificity", "harmonic_mean"]
rows = []

search_dirs = [cimt_dir]
if cimt_multimodal_dir.exists():
    search_dirs.append(cimt_multimodal_dir)

for search_dir in search_dirs:
    for csv_path in sorted(search_dir.glob("*/cv_summary.csv")):
        model = csv_path.parent.name
        df = pd.read_csv(csv_path)

        if "sensitivity" in df.columns and "specificity" in df.columns:
            denom = df["sensitivity"] + df["specificity"]
            df["harmonic_mean"] = np.where(denom > 0, 2 * (df["sensitivity"] * df["specificity"]) / denom, np.nan)

        row = {"model": model, "n_folds": len(df)}
        for m in METRICS:
            if m in df.columns:
                vals = df[m].dropna()
                row[f"{m}_mean"] = round(vals.mean(), 4)
                row[f"{m}_std"]  = round(vals.std(), 4)
        rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(output_csv, index=False)
print(f"\nCombined summary saved to: {output_csv}")
print(summary.to_string(index=False))
PYEOF

echo ""
echo "Done."
