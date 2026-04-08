#!/bin/bash
# submit_phase2_w_metdata.sh — Submit all CIMT multimodal configs as SLURM array jobs.
#
# Usage (from project root):
#     bash scripts/ibex/submit_phase2_w_metdata.sh

set -e

REPO_DIR="/home/x_albeshh/Project-Retinexis"
CONFIG_DIR="tasks/cimt/configs/multimodal"

configs=(
    ${REPO_DIR}/${CONFIG_DIR}/dinov2_large_multimodal.yaml
)

if [ ${#configs[@]} -eq 0 ]; then
    echo "No YAML configs found in ${CONFIG_DIR}/"
    exit 1
fi

echo "=========================================="
echo "CIMT Multimodal — Submitting ${#configs[@]} models × 5 folds"
echo "=========================================="

for config_path in "${configs[@]}"; do
    config_file=$(basename "$config_path")
    model_name="${config_file%_multimodal.yaml}"

    # Get the architecture folder for the checkpoint override
    if [[ "$model_name" == *"dinov2_large"* ]]; then ARCHITECTURE="dinov2"
    elif [[ "$model_name" == *"dinov3_large"* ]]; then ARCHITECTURE="dinov3"
    elif [[ "$model_name" == *"efficientnet_b2"* ]]; then ARCHITECTURE="efficientnet"
    elif [[ "$model_name" == *"convnextv2"* ]]; then ARCHITECTURE="convnextv2"
    elif [[ "$model_name" == *"retfound_mae"* ]]; then ARCHITECTURE="retfound_mae"
    elif [[ "$model_name" == *"retfound_dinov2"* ]]; then ARCHITECTURE="retfound_dinov2"
    else ARCHITECTURE="$model_name"
    fi

    LOG_DIR="${REPO_DIR}/outputs_multimodal/multimodal_${model_name}"
    mkdir -p "${LOG_DIR}"

    echo "  Submitting: ${model_name} (using checkpoint from outputs/cimt/${ARCHITECTURE}/...)"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=multimodal_${model_name}
#SBATCH --array=0-4
#SBATCH --output=${LOG_DIR}/%x_%A_%a.out
#SBATCH --error=${LOG_DIR}/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=halbeshri@sdm.sa

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
    --skip_phase1 \
    --checkpoint_path outputs/cimt/${ARCHITECTURE}/fold_\${SLURM_ARRAY_TASK_ID}/checkpoints/best.pt \
    --resume_weights_only

echo "Done: ${model_name} fold \${SLURM_ARRAY_TASK_ID}"
EOF

done

echo ""
echo "=========================================="
echo "All ${#configs[@]} models submitted ($((${#configs[@]} * 5)) jobs total)."
echo "Monitor with: squeue -u \$USER"
echo "=========================================="