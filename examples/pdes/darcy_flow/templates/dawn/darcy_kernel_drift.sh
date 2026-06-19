#!/bin/bash
# Direct empirical projected-NTK drift diagnostic for Darcy on Dawn.
#
# This is intentionally heavier than spectral diagnostics.  Each snapshot costs
# roughly PROBE_SAMPLES * N_PROJECTIONS backward passes.  For example, the
# default 64x16 probe is 1024 scalar-output gradients per kernel snapshot.
# The default probe is at the finest/base resolution, matching the usual
# evaluation convention.  Lower PROBE_RES values are cheaper smoke tests, not
# equivalent substitutes for the main diagnostic.
# Raise PROBE_SAMPLES only when the requested runtime/memory are intentional.

#SBATCH --job-name=darcy-kdrift-dawn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=AIRR-P67-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --mail-user=pr556@cam.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/modules.sh
fi

for arg in "$@"; do
    case "$arg" in
        *=*) export "$arg" ;;
    esac
done

MODULE_PURGE="${MODULE_PURGE:-1}"
MODULES="${MODULES:-rhel9/default-dawn intelpython-conda}"
DEVICE="${DEVICE:-xpu}"

if [ -n "$MODULES" ]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "ERROR: MODULES was set but the module command is not available."
        exit 2
    fi
    if [ "$MODULE_PURGE" = "1" ]; then
        module purge
    fi
    for mod in $MODULES; do
        module load "$mod"
    done
fi

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

if [ -z "${PYTHON:-}" ]; then
    # shellcheck disable=SC1091
    . scripts/select_runner_python.sh
    PYTHON=$(select_runner_python "$DEVICE")
fi

DATA_DIR="${DATA_DIR:-examples/pdes/darcy_flow/data}"
OUT_BASE="${OUT_BASE:-results/dawn_kernel_drift}"
RUN="${RUN:-ntk_500s}"
BASE_RES="${BASE_RES:-241}"
SEED="${SEED:-42}"

LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
FNO_WIDTH="${FNO_WIDTH:-32}"
FNO_MODES="${FNO_MODES:-8}"
ADD_COORDS="${ADD_COORDS:-1}"
LOSS_REDUCTION="${LOSS_REDUCTION:-sum}"
LOAD_GPU="${LOAD_GPU:-1}"
SPECTRAL="${SPECTRAL:-0}"

PROBE_SPLIT="${PROBE_SPLIT:-train}"
PROBE_RES="${PROBE_RES:-}"
PROBE_SAMPLES="${PROBE_SAMPLES:-64}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-4}"
N_PROJECTIONS="${N_PROJECTIONS:-16}"
FOURIER_RADIUS="${FOURIER_RADIUS:-8}"
MAX_ROWS="${MAX_ROWS:-2048}"
KERNEL_EPOCHS_JSON="${KERNEL_EPOCHS_JSON:-[0,1,2,5,10,20,50,100]}"
SAVE_KERNELS="${SAVE_KERNELS:-1}"
SAVE_PROJECTIONS="${SAVE_PROJECTIONS:-1}"
DENORMALIZE_OUTPUT="${DENORMALIZE_OUTPUT:-0}"

mkdir -p logs "$OUT_BASE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-24}}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

echo "Using PYTHON=${PYTHON}"
echo "Using DEVICE=${DEVICE}"
echo "Using RUN=${RUN}"
echo "Using probe: split=${PROBE_SPLIT} res=${PROBE_RES:-auto} samples=${PROBE_SAMPLES} projections=${N_PROJECTIONS}"
echo "Kernel epochs: ${KERNEL_EPOCHS_JSON}"

case "$RUN" in
    r241|baseline_r241)
        NAME="baseline_r241_kernel_drift"
        EPOCHS_JSON="${EPOCHS_JSON:-[500]}"
        RES_JSON="[[241]]"
        SUBSET_JSON="[[1024]]"
        LEVELS_JSON="[1]"
        BATCH_JSON="[[20]]"
        LR_JSON="[${LR}]"
        EVAL_EVERY="${EVAL_EVERY:-100}"
        ;;
    r120|baseline_r120)
        NAME="baseline_r120_kernel_drift"
        EPOCHS_JSON="${EPOCHS_JSON:-[500]}"
        RES_JSON="[[120]]"
        SUBSET_JSON="[[1024]]"
        LEVELS_JSON="[1]"
        BATCH_JSON="[[20]]"
        LR_JSON="[${LR}]"
        EVAL_EVERY="${EVAL_EVERY:-100}"
        ;;
    ntk|ntk_500s|ntk_optimal)
        NAME="ntk_optimal_500s_kernel_drift"
        EPOCHS_JSON="${EPOCHS_JSON:-[0, 1330, 1309, 91, 13]}"
        RES_JSON="[[15], [30], [60], [120], [241]]"
        SUBSET_JSON="[[1024], [1024], [1024], [1024], [1024]]"
        LEVELS_JSON="[1, 1, 1, 1, 1]"
        BATCH_JSON="[[160], [80], [80], [20], [20]]"
        LR_JSON="[${LR}, ${LR}, ${LR}, ${LR}, ${LR}]"
        EVAL_EVERY="${EVAL_EVERY:-160}"
        ;;
    *)
        echo "ERROR: unknown RUN=${RUN}; use r241, r120, or ntk_500s."
        exit 2
        ;;
esac

OUT_DIR="${OUT_DIR:-${OUT_BASE}/${NAME}_seed${SEED}_n${PROBE_SAMPLES}_p${N_PROJECTIONS}}"

args=(
    "$PYTHON" examples/pdes/darcy_flow/runners/run_experiment.py
    --seed "$SEED"
    --data_dir "$DATA_DIR"
    --base_res "$BASE_RES"
    --out_dir "$OUT_DIR"
    --device "$DEVICE"
    --optimizer sgd
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --loss_type lp
    --loss_reduction "$LOSS_REDUCTION"
    --fno_width "$FNO_WIDTH"
    --fno_modes "$FNO_MODES"
    --epochs_per_phase_json "$EPOCHS_JSON"
    --c2f_res_per_phase_json "$RES_JSON"
    --subset_size_per_phase_json "$SUBSET_JSON"
    --levels_per_phase_json "$LEVELS_JSON"
    --batch_size_per_phase_json "$BATCH_JSON"
    --lr_per_phase_json "$LR_JSON"
    --eval_every "$EVAL_EVERY"
    --no-use_scheduler
    --kernel_drift
    --kernel_drift_epochs_json "$KERNEL_EPOCHS_JSON"
    --kernel_drift_probe_split "$PROBE_SPLIT"
    --kernel_drift_probe_samples "$PROBE_SAMPLES"
    --kernel_drift_probe_batch_size "$PROBE_BATCH_SIZE"
    --kernel_drift_num_projections "$N_PROJECTIONS"
    --kernel_drift_fourier_radius "$FOURIER_RADIUS"
    --kernel_drift_max_rows "$MAX_ROWS"
    --save_final_checkpoint
)

if [ -n "$PROBE_RES" ]; then
    args+=(--kernel_drift_probe_res "$PROBE_RES")
fi

if [ "$LOAD_GPU" = "1" ]; then
    args+=(--load_gpu)
fi
if [ "$ADD_COORDS" = "1" ]; then
    args+=(--add_coords)
else
    args+=(--no-add_coords)
fi
if [ "$SPECTRAL" != "1" ]; then
    args+=(--no_spectral)
fi
if [ "$SAVE_KERNELS" != "1" ]; then
    args+=(--no-kernel_drift_save_kernels)
fi
if [ "$SAVE_PROJECTIONS" != "1" ]; then
    args+=(--no-kernel_drift_save_projections)
fi
if [ "$DENORMALIZE_OUTPUT" = "1" ]; then
    args+=(--kernel_drift_denormalize_output)
fi

echo "Output: $OUT_DIR"
"${args[@]}"

echo
echo "Kernel drift outputs:"
echo "  $OUT_DIR/kernel_drift/summary.csv"
echo "  $OUT_DIR/kernel_drift/metadata.json"
