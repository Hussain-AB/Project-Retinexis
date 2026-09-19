#!/bin/bash
# env_setup.sh — one place that defines the Ibex runtime for every job.
#
# Sourced, not executed:   source scripts/ibex/env_setup.sh
#
# Every SLURM job sources this so that a change to the module, the Python
# environment or a cache location happens once rather than in twenty scripts.
# It is idempotent and safe to source on a login node.

# --- project root ------------------------------------------------------------
# Resolved from this file's own location, so the scripts work from any cwd and
# in any user's home. Nothing here should ever hardcode /home/<someone>/.
_ENV_SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PROJECT_ROOT="$( cd "${_ENV_SETUP_DIR}/../.." && pwd )"
cd "${PROJECT_ROOT}" || return 1

export DATA_DIR="${PROJECT_ROOT}/data"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# --- python ------------------------------------------------------------------
module load machine_learning/2026.02 2>/dev/null || \
  echo "[env_setup] WARNING: could not load machine_learning/2026.02" >&2

# A project venv takes precedence over the module's python when present.
if [ -f "${PROJECT_ROOT}/myenv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/myenv/bin/activate"
fi

# --- caches ------------------------------------------------------------------
# /home is small on Ibex. Model downloads and W&B spool must not land there.
: "${SCRATCH_ROOT:=/ibex/scratch/${USER}/Project-Retinexis}"
export SCRATCH_ROOT
export HF_HOME="${SCRATCH_ROOT}/.cache/huggingface"
export TORCH_HOME="${SCRATCH_ROOT}/.cache/torch"
export WANDB_DIR="${SCRATCH_ROOT}/.wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/.cache/wandb"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

# Compute nodes may have no outbound network. Offline logging still produces a
# run that can be synced later with `wandb sync`.
: "${WANDB_MODE:=offline}"
export WANDB_MODE

# --- determinism -------------------------------------------------------------
# config.deterministic=true sets the torch-side flags; this is the CUDA-side
# requirement for deterministic cuBLAS reductions.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Keep BLAS from oversubscribing the 6 CPUs SLURM gives us per GPU.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

echo "[env_setup] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[env_setup] python=$(command -v python)  $(python --version 2>&1)"
echo "[env_setup] WANDB_MODE=${WANDB_MODE}  HF_HOME=${HF_HOME}"
