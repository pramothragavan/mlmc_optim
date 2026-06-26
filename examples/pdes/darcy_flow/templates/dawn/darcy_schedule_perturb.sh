#!/bin/bash
# Schedule perturbation experiment on Dawn.
#
# Tests whether NTK-optimal schedule performance depends on rank order
# vs exact magnitudes of the epoch allocation.
#
# All conditions use the same total epoch budget (2743) and hyperparameters
# as the NTK-optimal 500s probe, varying only the epoch allocation across
# the 5 resolution phases [r15, r30, r60, r120, r241].
#
# Each condition is submitted as a separate SLURM job via the wrapper
# darcy_schedule_perturb_submit.sh, or individually:
#   sbatch darcy_schedule_perturb.sh RUN=rank_flat
#
# Conditions:
#   ntk         [0, 1330, 1309, 91, 13]   NTK-derived (control)
#   rank_flat   [0, 800, 750, 650, 543]   Rank preserved, compressed range
#   rank_steep  [0, 2200, 500, 30, 13]    Rank preserved, exaggerated range
#   uniform     [0, 686, 686, 686, 685]   Equal allocation, no rank info
#   reversed    [0, 13, 91, 1309, 1330]   Inverted rank (anti-NTK)

#SBATCH --job-name=darcy-sched-perturb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=AIRR-P67-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
OUT_BASE="${OUT_BASE:-results/dawn_schedule_perturb}"
BASE_RES="${BASE_RES:-241}"
SEED="${SEED:-42}"
RUN="${RUN:-ntk}"

LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
FNO_WIDTH="${FNO_WIDTH:-32}"
FNO_MODES="${FNO_MODES:-8}"
LOAD_GPU="${LOAD_GPU:-1}"
ADD_COORDS="${ADD_COORDS:-1}"
LOSS_REDUCTION="${LOSS_REDUCTION:-sum}"
SPECTRAL="${SPECTRAL:-1}"

EVAL_EVERY="${EVAL_EVERY:-160}"

RES_JSON="[[15], [30], [60], [120], [241]]"
SUBSET_JSON="[[1024], [1024], [1024], [1024], [1024]]"
LEVELS_JSON="[1, 1, 1, 1, 1]"
BATCH_JSON="[[160], [80], [80], [20], [20]]"
LR_JSON="[${LR}, ${LR}, ${LR}, ${LR}, ${LR}]"

mkdir -p logs "$OUT_BASE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-24}}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

echo "Using PYTHON=${PYTHON}"
echo "Using DEVICE=${DEVICE}"
echo "Using RUN=${RUN}"

case "$RUN" in
    ntk|ntk_optimal)
        NAME="ntk_optimal"
        EPOCHS_JSON="[0, 1330, 1309, 91, 13]"
        ;;
    rank_flat)
        NAME="rank_flat"
        EPOCHS_JSON="[0, 800, 750, 650, 543]"
        ;;
    rank_steep)
        NAME="rank_steep"
        EPOCHS_JSON="[0, 2200, 500, 30, 13]"
        ;;
    uniform)
        NAME="uniform"
        EPOCHS_JSON="[0, 686, 686, 686, 685]"
        ;;
    reversed)
        NAME="reversed"
        EPOCHS_JSON="[0, 13, 91, 1309, 1330]"
        ;;
    *)
        echo "ERROR: unknown RUN='$RUN'"
        echo "  expected: ntk, rank_flat, rank_steep, uniform, reversed"
        exit 2
        ;;
esac

OUT_DIR="${OUT_DIR:-${OUT_BASE}/${NAME}_seed${SEED}}"

if [ -f "$OUT_DIR/summary.json" ]; then
    echo "SKIP: $OUT_DIR/summary.json already exists"
    exit 0
fi

echo "Running $NAME"
echo "  out_dir=$OUT_DIR"
echo "  epochs=$EPOCHS_JSON"

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
    --normalize true
    --epochs_per_phase_json "$EPOCHS_JSON"
    --c2f_res_per_phase_json "$RES_JSON"
    --subset_size_per_phase_json "$SUBSET_JSON"
    --levels_per_phase_json "$LEVELS_JSON"
    --batch_size_per_phase_json "$BATCH_JSON"
    --lr_per_phase_json "$LR_JSON"
    --eval_every "$EVAL_EVERY"
    --no-use_scheduler
    --save_final_checkpoint
)

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

"${args[@]}"

echo
echo "Finished: $OUT_DIR"
