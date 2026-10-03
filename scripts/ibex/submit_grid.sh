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
# core       = the 6-model grid over both tasks
# vessel     = the dual-stream arm and its two parameter-matched controls,
#              hypertension only. Same 8-job design per configuration.
# resolution = DINOv2-L at 224/336 (native 504) and ConvNeXtV2-B at 448 (native
#              224), both tasks, same 8-job design, so resolution is varied
#              WITHIN a model rather than confounded with it.
# wbce       = EfficientNet-B2 with the weighted loss back on, both tasks,
#              5 folds at seed 42 -- isolates the loss as the cause of the
#              calibration change.
# Configs for resolution and wbce come from scripts/make_ablation_configs.py.
ARM="core"
PER_MODEL_OUT=0
DESIGN="full"     # full = fold 0 x 3 seeds + replicate, then folds 1-4; folds = 0-4 at seed 42
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
        --arm)         ARM="$2"; shift ;;
        --account)     ACCOUNT="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

case "${ARM}" in
    core)
        MODELS=(efficientnet_b2 convnextv2 dinov3_large dinov2_large retfound_mae retfound_dinov2)
        TASKS=(cimt hypertension)
        CFG_SUFFIX="_phase2.yaml"
        CFG_SUBDIR="phase2"
        OUT_BASE="outputs_isbi"
        REP_BASE="outputs_isbi_rep"
        ;;
    vessel)
        # treatment + two controls; identical architecture and parameter count,
        # differing only in what the second branch receives
        MODELS=(efficientnet_b2_vessel efficientnet_b2_vessel_shuffled efficientnet_b2_vessel_image)
        TASKS=(hypertension)
        CFG_SUFFIX=".yaml"
        CFG_SUBDIR="vessel"
        OUT_BASE="outputs_isbi_vessel"
        REP_BASE="outputs_isbi_vessel_rep"
        ;;
    resolution)
        MODELS=(dinov2_large_res224 dinov2_large_res336 convnextv2_res448)
        TASKS=(cimt hypertension)
        CFG_SUFFIX=".yaml"
        CFG_SUBDIR="resolution"
        OUT_BASE="outputs_isbi_res"
        REP_BASE="outputs_isbi_res_rep"
        # dinov2_large_res224 and _res336 share architecture "dinov2", so in one
        # tree they would resolve to ONE run_dir and the second would die on the
        # restart guard -- exactly how cimt/retfound_dinov2 lost eight jobs. Each
        # configuration gets its own subdirectory instead.
        PER_MODEL_OUT=1
        ;;
    wbce)
        MODELS=(efficientnet_b2_wbce)
        TASKS=(cimt hypertension)
        CFG_SUFFIX=".yaml"
        CFG_SUBDIR="wbce"
        OUT_BASE="outputs_isbi_wbce"
        REP_BASE="outputs_isbi_wbce_rep"
        # 5 folds at seed 42, paired fold-for-fold against the core grid's
        # EfficientNet-B2. No seed or replicate rows.
        DESIGN="folds"
        ;;
    *) echo "unknown --arm: ${ARM} (expected core, vessel, resolution or wbce)" >&2; exit 2 ;;
esac
SEEDS=(42 1337 2024)
GRID_DIR="${OUT_BASE}/_grid"
LOG_DIR="${OUT_BASE}/_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="${GRID_DIR}/manifest_${STAMP}.tsv"

# A dry run must change nothing on disk, so its manifest goes to a temp file
# that is removed before exit. Only a real submission leaves a record behind.
if [ "${DRY_RUN}" -eq 1 ]; then
    MANIFEST="$(mktemp "${TMPDIR:-/tmp}/isbi-dryrun-XXXXXX.tsv")"
else
    mkdir -p "${GRID_DIR}" "${LOG_DIR}"
fi

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
        # resolution arm: scaled from measured s/epoch at native resolution
        # (DINOv2-L@504 ~2480 s, ConvNeXtV2-B@224 ~158 s on hypertension), worst
        # case 50 epochs. CIMT has ~0.28x the training images.
        hypertension:dinov2_large_res336) echo "24:00:00" ;;
        hypertension:dinov2_large_res224) echo "12:00:00" ;;
        hypertension:convnextv2_res448)   echo "16:00:00" ;;
        cimt:dinov2_large_res336)         echo "08:00:00" ;;
        cimt:dinov2_large_res224)         echo "04:00:00" ;;
        cimt:convnextv2_res448)           echo "06:00:00" ;;
        # EfficientNet-B2 measured at ~44 s/epoch on hypertension
        hypertension:efficientnet_b2_wbce) echo "04:00:00" ;;
        cimt:efficientnet_b2_wbce)         echo "02:00:00" ;;
        *)                            echo "12:00:00" ;;
    esac
}

emit () {   # task model fold seed output_dir
    local task="$1" model="$2" fold="$3" seed="$4" outdir="$5"
    local cfgdir
    case "${task}" in
        cimt)         cfgdir="tasks/cimt/configs/${CFG_SUBDIR}" ;;
        hypertension) cfgdir="tasks/hypertension/configs/${CFG_SUBDIR}" ;;
    esac
    [ -n "${ONLY_TASK}"  ] && [ "${task}"  != "${ONLY_TASK}"  ] && return 0
    [ -n "${ONLY_MODEL}" ] && [ "${model}" != "${ONLY_MODEL}" ] && return 0
    local cfg="${cfgdir}/${model}${CFG_SUFFIX}"
    [ -f "${cfg}" ] || { echo "missing config: ${cfg}" >&2; exit 1; }
    [ "${PER_MODEL_OUT}" -eq 1 ] && outdir="${outdir}/${model}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${task}" "${model}" "${fold}" "${seed}" "${outdir}" "${cfg}" >> "${MANIFEST}"
}

: > "${MANIFEST}"

if [ "${DESIGN}" = "folds" ]; then
    # One row per fold at the reference seed; nothing else.
    for task in "${TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            for fold in 0 1 2 3 4; do
                emit "${task}" "${model}" "${fold}" 42 "${OUT_BASE}"
            done
        done
    done
else
# Pass 1 -- fold 0 everywhere, all seeds, plus the seed-42 replicate.
for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            emit "${task}" "${model}" 0 "${seed}" "${OUT_BASE}"
        done
        emit "${task}" "${model}" 0 42 "${REP_BASE}"
    done
done
fi

# Pass 2 -- folds 1-4 at the reference seed only.
if [ "${DESIGN}" = "full" ] && [ "${FOLD0_ONLY}" -eq 0 ]; then
    for task in "${TASKS[@]}"; do
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
awk -F'\t' '{printf "  %-4d %-13s %-22s fold %s  seed %-5s -> %s\n", NR-1, $1, $2, $3, $4, $5}' "${MANIFEST}"
echo

if [ "${DRY_RUN}" -eq 1 ]; then
    rm -f "${MANIFEST}"
    echo "dry run -- nothing submitted, no manifest written."
    exit 0
fi

# One array per (task, model) so --time can differ; rows are selected from the
# shared manifest by their global index, so the manifest stays the one record.
for task in "${TASKS[@]}"; do
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
#SBATCH --job-name=isbi_${ARM}_${task}_${model}
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
