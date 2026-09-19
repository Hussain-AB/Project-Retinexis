#!/bin/bash
# env_setup.sh — one place that defines the Ibex runtime for every job.
#
# Sourced, not executed:   source scripts/ibex/env_setup.sh
#
# Returns non-zero if it cannot produce a working python. A job that sources
# this under `set -e` therefore dies at the top rather than 40 minutes later
# with an import error, and never trains against an environment nobody chose.
#
# Override the module without editing this file:
#   RETINEXIS_ML_MODULE=machine_learning/2025.10 source scripts/ibex/env_setup.sh

# --- project root ------------------------------------------------------------
# Resolved from this file's own location, so the scripts work from any cwd and
# in any user's home. Nothing here should ever hardcode /home/<someone>/.
_ENV_SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PROJECT_ROOT="$( cd "${_ENV_SETUP_DIR}/../.." && pwd )"
cd "${PROJECT_ROOT}" || return 1

export DATA_DIR="${PROJECT_ROOT}/data"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# --- module ------------------------------------------------------------------
# The version is NOT pinned here by default, because a pinned version that gets
# retired fails closed and blocks the grid -- which is what happened to
# machine_learning/2026.02, inherited from job_template.slurm. Set
# RETINEXIS_ML_MODULE to pin it once you know which version the grid ran on,
# and record that string in the paper's methods section.
: "${RETINEXIS_ML_MODULE:=machine_learning}"
if ! module load "${RETINEXIS_ML_MODULE}" 2>/dev/null; then
    echo "[env_setup] ERROR: could not load module '${RETINEXIS_ML_MODULE}'" >&2
    echo "[env_setup]        run: module av machine_learning" >&2
    return 1
fi
_ML_MODULE_LOADED="$(module -t list 2>&1 | grep -i machine_learning || true)"
echo "[env_setup] module=${_ML_MODULE_LOADED:-${RETINEXIS_ML_MODULE}}"
if [ "${RETINEXIS_ML_MODULE}" = "machine_learning" ]; then
    echo "[env_setup] NOTE: module version is unpinned -- whatever the site" >&2
    echo "[env_setup]       defaults to today. Pin RETINEXIS_ML_MODULE before" >&2
    echo "[env_setup]       the grid so all 96 runs share one environment." >&2
fi

# --- python ------------------------------------------------------------------
# A project venv takes precedence, but ONLY if it actually runs. myenv/ is built
# against the module's interpreter and borrows its libpython at runtime, so an
# orphaned venv fails with "libpython3.X.so.1.0: cannot open shared object
# file". Activating it anyway would hand every job a broken python.
if [ -f "${PROJECT_ROOT}/myenv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/myenv/bin/activate"
    if ! python -c "import sys" >/dev/null 2>&1; then
        echo "[env_setup] WARNING: ${PROJECT_ROOT}/myenv is not usable under" >&2
        echo "[env_setup]          this module; falling back to the module python." >&2
        if [ -n "${VIRTUAL_ENV:-}" ] && command -v deactivate >/dev/null 2>&1; then
            deactivate
        fi
    fi
fi

if ! python -c "import sys" >/dev/null 2>&1; then
    echo "[env_setup] ERROR: no working python after module load" >&2
    return 1
fi

# Fail here rather than inside a queued job.
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "[env_setup] ERROR: python runs but 'import torch' fails:" >&2
    python -c "import torch" 2>&1 | tail -3 >&2
    return 1
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
echo "[env_setup] torch=$(python -c 'import torch;print(torch.__version__, "cuda", torch.version.cuda)' 2>&1)"
echo "[env_setup] WANDB_MODE=${WANDB_MODE}  HF_HOME=${HF_HOME}"
