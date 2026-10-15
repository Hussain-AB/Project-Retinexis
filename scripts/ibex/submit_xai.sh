#!/usr/bin/env bash
# Grad-CAM + Integrated Gradients over every held-out image, as a SLURM array.
#
#   bash scripts/ibex/submit_xai.sh --dry-run          # write + print the job list, submit nothing
#   bash scripts/ibex/submit_xai.sh                    # submit everything
#   bash scripts/ibex/submit_xai.sh --stage main --only-task cimt
#
# Stages (default: all)
#   main      every model x fold 0-4, seed 42, all held-out images     60 jobs
#   ceiling   fold 0: seeds 1337 and 2024 + the exact seed-42 repeat   36 jobs
#             (is cross-model disagreement bigger than retraining noise?)
#   randinit  fold 0, 500 images, freshly initialised weights          12 jobs
#             (model-randomisation null: what the maps look like when
#              nothing has been learned)
#
# Resumable: re-running this script resubmits only (task, model, run) folders
# without a DONE file, and a restarted job skips finished chunks.
#
# Environment overrides:
#   GPU_CONSTRAINT=a100   add --constraint (empty by default)
#   THROTTLE=8            max array tasks running at once
#   OUT_ROOT=outputs/xai_full
#   VESSEL_CIMT=data/cimt/vessel_masks
#   VESSEL_HYPERTENSION=<read from tasks/hypertension/configs/vessel/efficientnet_b2_vessel.yaml>
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$(pwd)"

DRY_RUN=0
STAGE="all"
ONLY_TASK=""
ONLY_MODEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --stage) STAGE="$2"; shift ;;
        --only-task) ONLY_TASK="$2"; shift ;;
        --only-model) ONLY_MODEL="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

OUT_ROOT="${OUT_ROOT:-outputs/xai_full}"
THROTTLE="${THROTTLE:-8}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-}"
VESSEL_CIMT="${VESSEL_CIMT:-data/cimt/vessel_masks}"
if [ -z "${VESSEL_HYPERTENSION:-}" ]; then
    VESSEL_HYPERTENSION="$(awk -F': *' '/^vessel_mask_dir:/{gsub(/["'\'']/, "", $2); print $2}' \
        tasks/hypertension/configs/vessel/efficientnet_b2_vessel.yaml 2>/dev/null || true)"
fi
[ -n "${VESSEL_HYPERTENSION}" ] || { echo "set VESSEL_HYPERTENSION (could not read it from the vessel config)" >&2; exit 1; }
for d in "${VESSEL_CIMT}" "${VESSEL_HYPERTENSION}"; do
    [ -d "${d}" ] || { echo "vessel mask folder not found: ${d}" >&2; exit 1; }
done
[ -d xai ] || { echo "xai/ package not found in ${ROOT}" >&2; exit 1; }

MODELS=(retfound_dinov2 retfound_mae dinov2_large dinov3_large convnextv2 efficientnet_b2)
TASKS=(cimt hypertension)

time_for () {   # task model -> HH:MM:SS. IG dominates; the 500 px ViTs are ~4x the rest.
    case "$1:$2" in
        hypertension:dinov2_large|hypertension:retfound_dinov2) echo "12:00:00" ;;
        hypertension:*)                                         echo "06:00:00" ;;
        cimt:dinov2_large|cimt:retfound_dinov2)                 echo "05:00:00" ;;
        *)                                                      echo "03:00:00" ;;
    esac
}
dataset_dir () { [ "$1" = "hypertension" ] && echo "hypertension_combined" || echo "cimt"; }
arch_dir () {
    case "$1" in
        retfound_dinov2) echo retfound_dinov2 ;; retfound_mae) echo retfound_mae ;;
        dinov2_large) echo dinov2 ;; dinov3_large) echo dinov3 ;;
        convnextv2) echo convnextv2 ;; efficientnet_b2) echo efficientnet ;;
    esac
}

GRID="${OUT_ROOT}/_grid"
mkdir -p "${GRID}" "${GRID}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="${GRID}/manifest_${STAMP}.tsv"
: > "${MANIFEST}"

emit () {   # task model fold seed tree extra_args done_tag
    local task=$1 model=$2 fold=$3 seed=$4 tree=$5 extra=$6 tag=$7
    [ -n "${ONLY_TASK}" ] && [ "${task}" != "${ONLY_TASK}" ] && return 0
    [ -n "${ONLY_MODEL}" ] && [ "${model}" != "${ONLY_MODEL}" ] && return 0
    local sfx=""; [ "${seed}" != "42" ] && sfx="_seed${seed}"
    local run="${tree}/$(dataset_dir "${task}")/$(arch_dir "${model}")/fold_${fold}${sfx}"
    if [ ! -f "${run}/checkpoints/best.pt" ] || [ ! -f "${run}/predictions/val_predictions.csv" ]; then
        echo "  missing run, skipped: ${run}" >&2; return 0
    fi
    [ -f "${OUT_ROOT}/${task}/${model}/${tag}/DONE" ] && return 0
    local vd="${VESSEL_CIMT}"; [ "${task}" = "hypertension" ] && vd="${VESSEL_HYPERTENSION}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${task}" "${model}" "${fold}" "${seed}" "${tree}" \
        "${vd}" "${extra}" "$(time_for "${task}" "${model}")" >> "${MANIFEST}"
}

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "main" ]; then
            for fold in 0 1 2 3 4; do emit "${task}" "${model}" "${fold}" 42 outputs_isbi "-" "fold${fold}_seed42"; done
        fi
        if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "ceiling" ]; then
            emit "${task}" "${model}" 0 1337 outputs_isbi "-" "fold0_seed1337"
            emit "${task}" "${model}" 0 2024 outputs_isbi "-" "fold0_seed2024"
            emit "${task}" "${model}" 0 42 outputs_isbi_rep "-" "fold0_seed42_rep"
        fi
        if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "randinit" ]; then
            emit "${task}" "${model}" 0 42 outputs_isbi "--random-init --limit 500" "fold0_seed42_randinit"
        fi
    done
done

N=$(wc -l < "${MANIFEST}")
echo "manifest: ${MANIFEST}  (${N} jobs)"
awk -F'\t' '{printf "  %-4d %-13s %-16s fold %s seed %-5s %-17s %-28s %s\n", NR-1, $1, $2, $3, $4, $5, ($7=="-"?"":$7), $8}' "${MANIFEST}"
[ "${N}" -gt 0 ] || { echo "nothing to do"; exit 0; }

WORKER="${GRID}/worker_${STAMP}.sh"
cat > "${WORKER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT}"
source scripts/ibex/env_setup.sh
row=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
IFS=\$'\t' read -r task model fold seed tree vd extra tlimit <<< "\${row}"
[ "\${extra}" = "-" ] && extra=""
echo "row \${SLURM_ARRAY_TASK_ID}: \${task} \${model} fold \${fold} seed \${seed} \${tree} \${extra}"
echo "node \$(hostname)  gpu \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "git  \$(git rev-parse HEAD) dirty=\$(git status --porcelain --untracked-files=no | wc -l)"
python -m xai.run_maps --task "\${task}" --model "\${model}" --fold "\${fold}" --seed "\${seed}" \\
    --tree "\${tree}" --vessel-dir "\${vd}" --out-root "${OUT_ROOT}" \${extra}
echo "done: \${task} \${model} fold \${fold} seed \${seed} \${tree} \${extra}"
EOF
chmod +x "${WORKER}"

# one array per distinct time limit, so short jobs do not wait in a long queue
CONSTRAINT_ARG=()
[ -n "${GPU_CONSTRAINT}" ] && CONSTRAINT_ARG=(--constraint="${GPU_CONSTRAINT}")
for tl in $(cut -f8 "${MANIFEST}" | sort -u); do
    idx=$(awk -F'\t' -v t="${tl}" '$8==t{printf "%s%d", (n++?",":""), NR-1}' "${MANIFEST}")
    cmd=(sbatch --job-name="xai_${tl//:/}" --array="${idx}%${THROTTLE}" --time="${tl}"
         --gres=gpu:1 --cpus-per-task=8 --mem=64G "${CONSTRAINT_ARG[@]}"
         --output="${GRID}/logs/xai_%A_%a.out" --error="${GRID}/logs/xai_%A_%a.err" "${WORKER}")
    echo; echo "${cmd[*]}"
    [ "${DRY_RUN}" -eq 1 ] || "${cmd[@]}"
done
[ "${DRY_RUN}" -eq 1 ] && echo && echo "dry run: nothing submitted"
exit 0
