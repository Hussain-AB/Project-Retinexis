#!/usr/bin/env bash
# Calibrates all models across all datasets by searching for val_predictions.csv files.
# It automatically identifies the model directories and invokes calibrate.py for each.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUTS_DIR="${1:-$PROJECT_ROOT/outputs}"

# Convert OUTPUTS_DIR to an absolute path for cleaner path manipulation later
OUTPUTS_DIR="$(cd "$OUTPUTS_DIR" && pwd)"

echo "========================================================"
echo " Starting Global Calibration for Models in: "
echo " $OUTPUTS_DIR"
echo "========================================================"

# Find all val_predictions.csv files.
# For a path like:  /outputs/hypertension/convnextv2/fold_1/predictions/val_predictions.csv
# the model directory is: /outputs/hypertension/convnextv2
# We use rev and cut to dynamically strip the last 3 path components regardless of depth.
# Added '-L' to follow symlinks (as 'outputs' is linked to scratch space).
MODEL_DIRS=$(find -L "$OUTPUTS_DIR" -type f -name "val_predictions.csv" | rev | cut -d'/' -f4- | rev | sort -u)

if [ -z "$MODEL_DIRS" ]; then
    echo "No val_predictions.csv files found. Did you finish phase 2 training?"
    exit 0
fi

declare -i count=0
for model_dir in $MODEL_DIRS; do
    # Display the relative path from the outputs directory for cleaner logs
    rel_path="${model_dir#$OUTPUTS_DIR/}"
    
    echo ""
    echo "--------------------------------------------------------"
    echo " Calibrating Model: $rel_path"
    echo " Directory: $model_dir"
    echo "--------------------------------------------------------"

    python "$SCRIPT_DIR/calibrate.py" --run_dir "$model_dir"
    count=$((count + 1))
done

echo ""
echo "========================================================"
echo " Calibration complete! Successfully calibrated $count models."
echo "========================================================"
