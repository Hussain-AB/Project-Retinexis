#!/bin/bash
#SBATCH --job-name=phase1_efficientnet_b2
#SBATCH --output=phase1_efficientnet_b2/%x_%j.out # %x is the job name, %j is the job ID
#SBATCH --error=phase1_efficientnet_b2/%x_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:4 # Request an A100 GPU
#SBATCH --time=24:00:00 # Depending on training length, adjust time up to 72 hours for long jobs

# Ensure we fail fast if any command fails
set -e

# -----------------------------------------------------------------------------
# 1. Environment Setup
# -----------------------------------------------------------------------------

# Load required modules (e.g., machine learning tools, CUDA)
# module load machine_learning 

# -----------------------------------------------------------------------------
# 2. Define Paths
# -----------------------------------------------------------------------------
# Set up where to save your model checkpoints and logs
mkdir -p outputs/phase1_efficientnet_b2

echo "Checkpoints and logs will be saved to: /outputs/phase1_efficientnet_b2"

# -----------------------------------------------------------------------------
# 3. Model Training
# -----------------------------------------------------------------------------
echo "Running training script..."

# Run your Python training script
python scripts/train.py \
    --config tasks/cimt/configs/phase1/efficientnet_b2_phase1.yaml \
    --phase1_only

echo "Training finished successfully!"