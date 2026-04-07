#!/bin/bash
# generate_vessel_masks.sh — Submit vessel mask generation as a single SLURM job.
#
# Runs scripts/generate_vessel_masks.py on the full CIMT dataset using a GPU.
# Must be run once before any vessel multimodal training.
#
# Usage (from project root):
#     bash scripts/ibex/generate_vessel_masks.sh

set -e

REPO_DIR="/home/x_albeshh/Project-Retinexis"
IMAGES_DIR="${REPO_DIR}/data/cimt/Fundus_CIMT_2903_Dataset/images"
SAVE_DIR="${REPO_DIR}/data/cimt/vessel_masks"
LOG_DIR="${REPO_DIR}/outputs/vessel_masks"

mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Vessel Mask Generation — Submitting job"
echo "Images: ${IMAGES_DIR}"
echo "Output: ${SAVE_DIR}"
echo "========================================"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vessel_masks
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00

set -e

cd ${REPO_DIR}

echo "============================================"
echo "Job:    vessel mask generation"
echo "Node:   \$(hostname)"
echo "GPU:    \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start:  \$(date)"
echo "============================================"

python scripts/generate_vessel_masks.py \
    --images-dir ${IMAGES_DIR} \
    --save-dir ${SAVE_DIR}

echo "============================================"
echo "Done: \$(date)"
echo "Masks saved to: ${SAVE_DIR}"
echo "============================================"
EOF

echo ""
echo "Submitted. Monitor with: squeue -u \$USER"
echo "Logs: ${LOG_DIR}/"
