#!/bin/bash
# Port-correctness verification: r15 & r30 baselines with --normalize_input False
# to match mesh's accidental behaviour (raw inputs, normalised outputs).
# Epoch counts match mesh baselines script exactly.

#SBATCH --job-name=darcy-verify-port
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=LIO-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
MODULES="${MODULES:-rhel8/default-amp}"
DEVICE="${DEVICE:-cuda}"

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

DATA_DIR="${DATA_DIR:-examples/pdes/darcy_flow/data}"
OUT_BASE="${OUT_BASE:-results/verify_port/ampere}"
if [ -z "${PYTHON:-}" ]; then
    # shellcheck disable=SC1091
    . scripts/select_runner_python.sh
    PYTHON=$(select_runner_python "$DEVICE")
fi
BASE_RES="${BASE_RES:-241}"
SEED="${SEED:-42}"
RUNS="${RUNS:-r15 r30}"

LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
FNO_WIDTH="${FNO_WIDTH:-32}"
FNO_MODES="${FNO_MODES:-8}"
LOAD_GPU="${LOAD_GPU:-1}"
SPECTRAL="${SPECTRAL:-1}"
ADD_COORDS="${ADD_COORDS:-1}"
LOSS_REDUCTION="${LOSS_REDUCTION:-sum}"

# Epoch counts from mesh baselines script
R15_EPOCHS="${R15_EPOCHS:-37500}"
R30_EPOCHS="${R30_EPOCHS:-19000}"

R15_EVAL_EVERY="${R15_EVAL_EVERY:-750}"
R30_EVAL_EVERY="${R30_EVAL_EVERY:-380}"

mkdir -p logs "$OUT_BASE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset PYTHONPATH
unset PYTHONHOME

echo "Using PYTHON=${PYTHON}"
echo "Using DEVICE=${DEVICE}"

run_experiment() {
    local name="$1"
    local epochs_json="$2"
    local res_json="$3"
    local subset_json="$4"
    local levels_json="$5"
    local batch_json="$6"
    local lr_json="$7"
    local eval_every="$8"
    local spectral="$9"
    local out_dir="$OUT_BASE/${name}_seed${SEED}"

    if [ -f "$out_dir/summary.json" ]; then
        echo "SKIP: $out_dir/summary.json already exists"
        return 0
    fi

    echo
    echo "Running $name"
    echo "  out_dir=$out_dir"
    echo "  epochs=$epochs_json res=$res_json batch=$batch_json eval_every=$eval_every"

    args=(
        "$PYTHON" examples/pdes/darcy_flow/runners/run_experiment.py
        --seed "$SEED"
        --data_dir "$DATA_DIR"
        --base_res "$BASE_RES"
        --out_dir "$out_dir"
        --device "$DEVICE"
        --optimizer sgd
        --lr "$LR"
        --weight_decay "$WEIGHT_DECAY"
        --loss_type lp
        --loss_reduction "$LOSS_REDUCTION"
        --fno_width "$FNO_WIDTH"
        --fno_modes "$FNO_MODES"
        --epochs_per_phase_json "$epochs_json"
        --c2f_res_per_phase_json "$res_json"
        --subset_size_per_phase_json "$subset_json"
        --levels_per_phase_json "$levels_json"
        --batch_size_per_phase_json "$batch_json"
        --lr_per_phase_json "$lr_json"
        --eval_every "$eval_every"
        --no-use_scheduler
        --save_final_checkpoint
        --normalize_input False
    )

    if [ "$LOAD_GPU" = "1" ]; then
        args+=(--load_gpu)
    fi
    if [ "$ADD_COORDS" = "1" ]; then
        args+=(--add_coords)
    else
        args+=(--no-add_coords)
    fi
    if [ "$spectral" != "1" ]; then
        args+=(--no_spectral)
    fi

    "${args[@]}"
}

for run in $RUNS; do
    case "$run" in
        r15)
            run_experiment "verify_r15" "[${R15_EPOCHS}]" \
                "[[15]]" "[[1024]]" "[1]" "[[160]]" "[${LR}]" \
                "$R15_EVAL_EVERY" "$SPECTRAL"
            ;;
        r30)
            run_experiment "verify_r30" "[${R30_EPOCHS}]" \
                "[[30]]" "[[1024]]" "[1]" "[[80]]" "[${LR}]" \
                "$R30_EVAL_EVERY" "$SPECTRAL"
            ;;
        *)
            echo "ERROR: unknown run '$run' in RUNS='$RUNS' (expected r15, r30)"
            exit 2
            ;;
    esac
done

echo
echo "Finished port verification: $OUT_BASE"
