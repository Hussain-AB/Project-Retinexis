#!/bin/bash
# submit_grid.sh — the ISBI retrain grid, as one manifest-driven SLURM array.
#
#   bash scripts/ibex/submit_grid.sh --dry-run     # build + print the manifest
#   bash scripts/ibex/submit_grid.sh               # build it, then sbatch
#   bash scripts/ibex/submit_grid.sh --fold0-only  # just the attribution ceiling
#
# Why a manifest rather than one array per model, as submit_phase2.sh did:
#
#   * a single (task, model, fold, seed) row per job makes the grid auditable --
#     the manifest is a record of exactly what was queued, and it is written
#     into the output tree beside the runs
#   * fold 0 of every model is queued FIRST. Fold 0 carries all three seeds and
#     is what the attribution study needs, so a queue that drains slowly still
#     produces a usable paper
#   * the seed reaches the trainer explicitly via --override seed, so seeds
#     1337/2024 land in fold_N_seedXXXX directories and cannot overwrite seed 42
#   * --time comes from a per-model worst case, not a median. The observed
#     medians were measured across restarts with a reset patience counter, so
#     they under-forecast badly
#
# Read scripts/preflight_check.py's output before running this. It refuses to
# be helpful after the fact.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
cd "${PROJECT_ROOT}"

DRY_RUN=0
FOLD0_ONLY=0
# Re-run a subset without disturbing finished runs, e.g. after fixing one
# config:  --only-task cimt --only-model retfound_dinov2
ONLY_TASK=""
ONLY_MODEL=""
THROTTLE=6
# Left empty by default: Ibex resolves the account automatically, and
# passing an explicit one is rejected with "Invalid account or
# account/partition combination". job_template.slurm hardcoded
# --account=x_almofaf while running out of another user's home; do not
# copy that. Set SBATCH_ACCOUNT, or pass --account, only if the site
# later requires it.
ACCOUNT="${SBATCH_ACCOUNT:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)     DRY_RUN=1 ;;
        --fold0-only)  FOLD0_ONLY=1 ;;
        --throttle)    THROTTLE="$2"; shift ;;
        --only-task)   ONLY_TASK="$2"; shift ;;
        --only-model)  ONLY_MODEL="$2"; shift ;;
        --account)     ACCOUNT="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

MODELS=(efficientnet_b2 convnextv2 dinov3_large dinov2_large retfound_mae retfound_dinov2)
SEEDS=(42 1337 2024)
OUT_BASE="outputs_isbi"
REP_BASE="outputs_isbi_rep"     # the seed-42 replicate, for sigma_nondet
GRID_DIR="${OUT_BASE}/_grid"
LOG_DIR="${OUT_BASE}/_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="${GRID_DIR}/manifest_${STAMP}.tsv"

mkdir -p "${GRID_DIR}" "${LOG_DIR}"

# --- wall-clock ceilings -----------------------------------------------------
# Worst case = phase2_epochs (50) x measured per-epoch cost, never early
# stopping. CIMT is ~1/3 of hypertension's 20,928 images. RETFound-DINOv2 has
# NO phase-2 logs anywhere -- its figure is an estimate and the smoke test must
# replace it before the big jobs are trusted.
time_for () {   # task model -> HH:MM:SS
    case "$1:$2" in
        hypertension:retfound_dinov2) echo "72:00:00" ;;
        hypertension:dinov2_large)    echo "60:00:00" ;;
        hypertension:retfound_mae)    echo "48:00:00" ;;
        hypertension:dinov3_large)    echo "24:00:00" ;;
        hypertension:convnextv2)      echo "24:00:00" ;;
        hypertension:efficientnet_b2) echo "12:00:00" ;;
        cimt:retfound_dinov2)         echo "24:00:00" ;;
        cimt:dinov2_large)            echo "20:00:00" ;;
        cimt:retfound_mae)            echo "16:00:00" ;;
        *)                            echo "12:00:00" ;;
    esac
}

emit () {   # task model fold seed output_dir
    local task="$1" model="$2" fold="$3" seed="$4" outdir="$5"
    local cfgdir
    case "${task}" in
        cimt)         cfgdir="tasks/cimt/configs/phase2" ;;
        hypertension) cfgdir="tasks/hypertension/configs/phase2" ;;
    esac
    [ -n "${ONLY_TASK}"  ] && [ "${task}"  != "${ONLY_TASK}"  ] && return 0
    [ -n "${ONLY_MODEL}" ] && [ "${model}" != "${ONLY_MODEL}" ] && return 0
    local cfg="${cfgdir}/${model}_phase2.yaml"
    [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${task}" "${model}" "${fold}" "${seed}" "${outdir}" "${cfg}" >> "${MANIFEST}"
}

: > "${MANIFEST}"

# Pass 1 -- fold 0 everywhere, all seeds, plus the seed-42 replicate.
for task in cimt hypertension; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            emit "${task}" "${model}" 0 "${seed}" "${OUT_BASE}"
        done
        emit "${task}" "${model}" 0 42 "${REP_BASE}"
    done
done

# Pass 2 -- folds 1-4 at the reference seed only.
if [ "${FOLD0_ONLY}" -eq 0 ]; then
    for task in cimt hypertension; do
        for model in "${MODELS[@]}"; do
            for fold in 1 2 3 4; do
                emit "${task}" "${model}" "${fold}" 42 "${OUT_BASE}"
            done
        done
    done
fi

N=$(wc -l < "${MANIFEST}")
echo "manifest: ${MANIFEST}  (${N} jobs)"
echo
awk -F'\t' '{printf "  %-4d %-13s %-17s fold %s  seed %-5s -> %s\n", NR-1, $1, $2, $3, $4, $5}' "${MANIFEST}"
echo

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "dry run -- nothing submitted. Rerun without --dry-run to queue."
    exit 0
fi

# One array per (task, model) so --time can differ; rows are selected from the
# shared manifest by their global index, so the manifest stays the one record.
for task in cimt hypertension; do
    for model in "${MODELS[@]}"; do
        idx=$(awk -F'\t' -v t="${task}" -v m="${model}" \
              'ARGV[1]==FILENAME && $1==t && $2==m {printf "%s%d", sep, NR-1; sep=","}' "${MANIFEST}")
        [ -n "${idx}" ] || continue
        tlimit=$(time_for "${task}" "${model}")

        # A bare "#" rather than an empty string: sbatch stops reading #SBATCH
        # directives at the first line that is neither a comment nor blank, and
        # a comment line is unambiguously safe where a blank one is merely
        # probably safe.
        acct_line="#"
        [ -n "${ACCOUNT}" ] && acct_line="#SBATCH --account=${ACCOUNT}"

        sbatch <<SBATCH
#!/bin/bash
${acct_line}
#SBATCH --job-name=isbi_${task}_${model}
#SBATCH --array=${idx}%${THROTTLE}
#SBATCH --output=${PROJECT_ROOT}/${LOG_DIR}/%x_%A_%a.out
#SBATCH --error=${PROJECT_ROOT}/${LOG_DIR}/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=${tlimit}

set -euo pipefail
source "${PROJECT_ROOT}/scripts/ibex/env_setup.sh"

row=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${PROJECT_ROOT}/${MANIFEST}")
IFS=\$'\t' read -r r_task r_model r_fold r_seed r_out r_cfg <<< "\${row}"

echo "=============================================="
echo " manifest row \${SLURM_ARRAY_TASK_ID}: \${row}"
echo " job \${SLURM_JOB_ID}  node \$(hostname)"
echo " gpu  \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo " git  \$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
echo "=============================================="

srun python scripts/train.py \\
    --config "\${r_cfg}" \\
    --fold "\${r_fold}" \\
    --output_dir "\${r_out}" \\
    --override seed "\${r_seed}"

echo "done: \${r_task} \${r_model} fold \${r_fold} seed \${r_seed}"
SBATCH
        echo "queued ${task}/${model}  time=${tlimit}  rows=${idx}"
    done
done

echo
echo "manifest kept at ${MANIFEST} -- do not edit it; the running jobs read it."
