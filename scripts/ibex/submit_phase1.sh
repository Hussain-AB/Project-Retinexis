#!/bin/bash
#SBATCH --job-name=phase1
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=45G
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase1_%A_%a.out
#SBATCH --error=logs/phase1_%A_%a.err

set -e

conda activate retinexis
cd /ibex/user/$USER/Project-Retinexis

# Config passed as argument: sbatch submit_phase1.sh <config_file>
CONFIG=${1:-"tasks/cimt/configs/phase1/efficientnet_b2_phase1.yaml"}

echo "============================================"
echo "Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Fold: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

python scripts/train.py \
    --config $CONFIG \
    --fold $SLURM_ARRAY_TASK_ID \
    --phase1_only
