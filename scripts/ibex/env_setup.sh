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

# --- python ------------------------------------------------------------------
# myenv/ is a self-contained venv (include-system-site-packages = false) built
# from a standalone interpreter, NOT from a machine_learning module -- see
# myenv/pyvenv.cfg. The only thing it needs from outside is that interpreter's
# libpython, which is why it dies with "libpython3.11.so.1.0: cannot open
# shared object file" when nothing has put it on the linker path.
#
# So: read the base prefix out of pyvenv.cfg and add its lib dir to
# LD_LIBRARY_PATH. No module required, and the venv itself pins every package,
# which makes it a better reproducibility anchor than a site module that can be
# retired mid-grid (machine_learning/2026.02 already was).
#
# Set RETINEXIS_ML_MODULE to load a site module instead, for a machine where
# the venv is missing or unusable.

_VENV="${PROJECT_ROOT}/myenv"
_PYVENV_CFG="${_VENV}/pyvenv.cfg"

if [ -n "${RETINEXIS_ML_MODULE:-}" ]; then
    if ! module load "${RETINEXIS_ML_MODULE}" 2>/dev/null; then
        echo "[env_setup] ERROR: could not load module '${RETINEXIS_ML_MODULE}'" >&2
        return 1
    fi
    echo "[env_setup] module=${RETINEXIS_ML_MODULE}"
fi

if [ -f "${_PYVENV_CFG}" ]; then
    _BASE_BIN="$(awk -F'=' '/^[[:space:]]*home[[:space:]]*=/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${_PYVENV_CFG}")"
    if [ -n "${_BASE_BIN}" ]; then
        _BASE_PREFIX="$(dirname "${_BASE_BIN}")"
        for _libdir in "${_BASE_PREFIX}/lib" "${_BASE_PREFIX}/lib64"; do
            if [ -d "${_libdir}" ]; then
                export LD_LIBRARY_PATH="${_libdir}:${LD_LIBRARY_PATH:-}"
            fi
        done
        echo "[env_setup] venv base=${_BASE_PREFIX}"
    fi
    # shellcheck disable=SC1091
    source "${_VENV}/bin/activate"
fi

# Never hand a job an interpreter that cannot start.
if ! python -c "import sys" >/dev/null 2>&1; then
    echo "[env_setup] ERROR: python does not run." >&2
    echo "[env_setup]   tried venv: ${_VENV}" >&2
    echo "[env_setup]   LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}" >&2
    python -c "import sys" 2>&1 | tail -2 >&2
    echo "[env_setup]   fix: confirm the path in ${_PYVENV_CFG} still exists," >&2
    echo "[env_setup]        or rebuild myenv, or set RETINEXIS_ML_MODULE." >&2
    return 1
fi

# Fail here rather than inside a queued job.
#
# These are IMPORTED, not merely located. importlib.util.find_spec() only finds
# a package directory on disk -- it reported transformers and huggingface_hub as
# present in a venv where neither could import, because `packaging` and
# `filelock` were missing. Only an actual import proves the package works.
_IMPORT_REPORT="$(python - <<'PYCHK' 2>/dev/null
# required: the training path imports every one of these.
req = ["torch","torchvision","yaml","numpy","pandas","PIL","sklearn","scipy",
       "timm","transformers","huggingface_hub","tqdm","wandb"]
# optional: vessel-mask generation (src/rrwnet) and attribution (xai/) only.
# Missing these must not block the core grid, which never imports them.
opt = ["skimage"]
out = []
for m in req + opt:
    try:
        __import__(m)
    except Exception as exc:
        tag = "OPTIONAL " if m in opt else ""
        out.append(f"{tag}{m}: {type(exc).__name__}: {exc}")
print(chr(10).join(out))
PYCHK
)"
_OPT_MISSING="$(printf '%s\n' "${_IMPORT_REPORT}" | grep "^OPTIONAL " || true)"
_IMPORT_REPORT="$(printf '%s\n' "${_IMPORT_REPORT}" | grep -v "^OPTIONAL " | grep -v "^$" || true)"
if [ -n "${_OPT_MISSING}" ]; then
    echo "[env_setup] note: optional packages unavailable -- vessel masks and" >&2
    echo "[env_setup]       xai only; the core grid never imports them:" >&2
    echo "${_OPT_MISSING}" | sed "s/^OPTIONAL /[env_setup]   /" >&2
fi
if [ -n "${_IMPORT_REPORT}" ]; then
    echo "[env_setup] ERROR: packages present but not importable:" >&2
    echo "${_IMPORT_REPORT}" | sed 's/^/[env_setup]   /' >&2
    echo "[env_setup] repair (on a LOGIN node -- compute nodes have no network):" >&2
    echo "[env_setup]   python -m pip check            # list every broken dependency" >&2
    echo "[env_setup]   python -m pip install -r requirements.txt" >&2
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
