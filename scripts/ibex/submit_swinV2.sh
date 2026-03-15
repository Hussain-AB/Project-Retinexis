#!/bin/bash
# submit_swinV2.sh — Submit SwinV2 Phase 1 (CIMT) as a SLURM array job.
#
# Usage (from project root):
#     bash scripts/ibex/submit_swinV2.sh

set -e

REPO_DIR="/home/x_albeshh/Project-Retinexis"
CONFIG_DIR="tasks/cimt/configs/phase1"
CONFIG_FILE="swinv2_phase1.yaml"
MODEL_NAME="swinv2"

LOG_DIR="${REPO_DIR}/outputs/phase1_${MODEL_NAME}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "CIMT Phase 1 — Submitting ${MODEL_NAME} × 5 folds"
echo "========================================"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=phase1_${MODEL_NAME}
#SBATCH --array=0-4
#SBATCH --output=${LOG_DIR}/%x_%A_%a.out
#SBATCH --error=${LOG_DIR}/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

set -e

cd ${REPO_DIR}

echo "============================================"
echo "Model:  ${MODEL_NAME}"
echo "Config: ${CONFIG_DIR}/${CONFIG_FILE}"
echo "Fold:   \${SLURM_ARRAY_TASK_ID}"
echo "Node:   \$(hostname)"
echo "GPU:    \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

python scripts/train.py \\
    --config ${CONFIG_DIR}/${CONFIG_FILE} \\
    --fold \${SLURM_ARRAY_TASK_ID} \\
    --phase1_only

echo "Done: ${MODEL_NAME} fold \${SLURM_ARRAY_TASK_ID}"
EOF

echo ""
echo "========================================"
echo "${MODEL_NAME} submitted (5 jobs total)."
echo "Monitor with: squeue -u \$USER"
echo "========================================"