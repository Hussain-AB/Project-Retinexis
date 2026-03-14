#!/bin/bash
# submit_phase1.sh — Submit all CIMT Phase 1 configs as SLURM array jobs.
#
# Usage (from project root):
#     bash scripts/ibex/submit_phase1.sh

set -e

REPO_DIR="/home/x_albeshh/Project-Retinexis"
CONFIG_DIR="tasks/cimt/configs/phase1"

configs=(${REPO_DIR}/${CONFIG_DIR}/*.yaml)

if [ ${#configs[@]} -eq 0 ]; then
    echo "No YAML configs found in ${CONFIG_DIR}/"
    exit 1
fi

echo "========================================"
echo "CIMT Phase 1 — Submitting ${#configs[@]} models × 5 folds"
echo "========================================"

for config_path in "${configs[@]}"; do
    config_file=$(basename "$config_path")
    model_name="${config_file%_phase1.yaml}"

    LOG_DIR="${REPO_DIR}/outputs/phase1_${model_name}"
    mkdir -p "${LOG_DIR}"

    echo "  Submitting: ${model_name}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=phase1_${model_name}
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
echo "Model:  ${model_name}"
echo "Config: ${CONFIG_DIR}/${config_file}"
echo "Fold:   \${SLURM_ARRAY_TASK_ID}"
echo "Node:   \$(hostname)"
echo "GPU:    \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

python scripts/train.py \
    --config ${CONFIG_DIR}/${config_file} \
    --fold \${SLURM_ARRAY_TASK_ID} \
    --phase1_only

echo "Done: ${model_name} fold \${SLURM_ARRAY_TASK_ID}"
EOF

done

echo ""
echo "========================================"
echo "All ${#configs[@]} models submitted ($((${#configs[@]} * 5)) jobs total)."
echo "Monitor with: squeue -u \$USER"
echo "========================================"