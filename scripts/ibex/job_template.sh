#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=%x_%j.out  # %x is the job name, %j is the job ID
#SBATCH --error=%x_%j.err  
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1 # Request an A100 GPU
#SBATCH --time=24:00:00 # Depending on training length, adjust time up to 72 hours for long jobs

# Ensure we fail fast if any command fails
set -e

# -----------------------------------------------------------------------------
# 1. Environment Setup
# -----------------------------------------------------------------------------
echo "Setting up environment..."

# Load required modules (e.g., machine learning tools, CUDA)
# module load machine_learning 

# -----------------------------------------------------------------------------
# 2. Define Paths
# -----------------------------------------------------------------------------
# Set up where to save your model checkpoints and logs
SAVE_DIR="experiments/{{JOB_NAME}}_${SLURM_JOB_ID}"
mkdir -p "$SAVE_DIR"

echo "Checkpoints and logs will be saved to: $SAVE_DIR"

# -----------------------------------------------------------------------------
# 3. Model Training
# -----------------------------------------------------------------------------
echo "Running training script..."

# Run your Python training script
python path/to/train_script.py --save_dir "$SAVE_DIR"

echo "Training finished successfully!"
