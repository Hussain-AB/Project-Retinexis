#!/bin/bash
# submit_phase1.sh — Submit all CIMT Phase 1 configs as parallel SLURM jobs.
#
# Usage (from project root):
#     bash scripts/ibex/submit_phase1.sh
#
# Each config in tasks/cimt/configs/phase1/*.yaml gets its own SLURM job
# with identical resource requests (1× A100, 64GB, 8 CPUs, 24h).

set -e

REPO_DIR="/home/x_almofaf/Project-Retinexis"
CONFIG_DIR="tasks/cimt/configs/phase1"

# Collect all phase1 YAML configs
configs=(${REPO_DIR}/${CONFIG_DIR}/*.yaml)

if [ ${#configs[@]} -eq 0 ]; then
    echo "No YAML configs found in ${CONFIG_DIR}/"
    exit 1
fi

echo "========================================"
echo "CIMT Phase 1 — Submitting ${#configs[@]} jobs"
echo "========================================"

for config_path in "${configs[@]}"; do
    # Extract model name from filename (e.g. efficientnet_b2_phase1.yaml → efficientnet_b2)
    config_file=$(basename "$config_path")
    model_name="${config_file%_phase1.yaml}"

    echo "  Submitting: ${model_name} (${config_file})"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${model_name}
#SBATCH --output=${model_name}_%j.out
#SBATCH --error=${model_name}_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00

set -e

cd ${REPO_DIR}

echo "============================================"
echo "Phase 1: ${model_name}"
echo "Config:  ${CONFIG_DIR}/${config_file}"
echo "============================================"

pip install timm==1.0.3 albumentations opencv-python

python scripts/train.py \\
    --config ${CONFIG_DIR}/${config_file} \\
    --phase1_only

echo "Training finished: ${model_name}"
EOF

done

echo ""
echo "========================================"
echo "All ${#configs[@]} jobs submitted."
echo "Use 'squeue -u \$USER' to monitor."
echo "========================================"
