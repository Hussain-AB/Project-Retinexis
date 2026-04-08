#!/bin/bash
# submit_vessel.sh — Submit EfficientNet-B2 vessel segmentation models (with and without metadata).
#
# Both models warm-start the image backbone from trained efficientnet phase2 checkpoints.
# Run from project root:
#     bash scripts/ibex/submit_vessel.sh

set -e

REPO_DIR="/home/x_albeshh/Project-Retinexis"
CONFIG_DIR="tasks/cimt/configs/vessel"

configs=(
    #"${REPO_DIR}/${CONFIG_DIR}/efficientnet_b2_vessel.yaml"
    "${REPO_DIR}/${CONFIG_DIR}/efficientnet_b2_vessel_multimodal.yaml"
)

echo "=========================================="
echo "Vessel Segmentation — Submitting ${#configs[@]} models × 5 folds"
echo "=========================================="

for config_path in "${configs[@]}"; do
    config_file=$(basename "$config_path")
    model_name="${config_file%.yaml}"

    if [[ "$model_name" == *"multimodal"* ]]; then
        LOG_DIR="${REPO_DIR}/outputs_segmentation_multimodal/segmentation_${model_name}"
    else
        LOG_DIR="${REPO_DIR}/outputs_segmentation/segmentation_${model_name}"
    fi
    mkdir -p "${LOG_DIR}"

    echo "  Submitting: ${model_name}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${model_name}
#SBATCH --array=0
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

if [[ "${config_file}" == *"multimodal"* ]]; then
    CHECKPOINT_OVERRIDE="--override backbone_checkpoint_path outputs_segmentation/cimt/efficientnet_vessel/fold_\${SLURM_ARRAY_TASK_ID}/checkpoints/best.pt"
else
    CHECKPOINT_OVERRIDE=""
fi

python scripts/train.py \
    --config ${CONFIG_DIR}/${config_file} \
    --fold \${SLURM_ARRAY_TASK_ID} \
    --skip_phase1 \
    \${CHECKPOINT_OVERRIDE}

echo "Done: ${model_name} fold \${SLURM_ARRAY_TASK_ID}"
EOF

done

echo ""
echo "=========================================="
echo "Submitted ${#configs[@]} models ($(( ${#configs[@]} * 5 )) jobs total)."
echo "Logs:    outputs_segmentation/segmentation_efficientnet_b2_vessel/"
echo "         outputs_segmentation_multimodal/segmentation_efficientnet_b2_vessel_multimodal/"
echo "Models:  outputs_segmentation/cimt/efficientnet_vessel/"
echo "         outputs_segmentation_multimodal/cimt/efficientnet_vessel/"
echo "Monitor: squeue -u \$USER"
echo "=========================================="
