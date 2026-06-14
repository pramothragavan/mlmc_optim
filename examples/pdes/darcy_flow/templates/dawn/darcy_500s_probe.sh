#!/bin/bash
# Throwaway Darcy probe using mesh-derived empirical CUDA train-time costs.
# On Dawn these are smoke-test lengths, not calibrated Dawn wall-clock budgets.
#
#   r15:  0.053961 s/epoch  -> 9266 epochs for ~500s
#   r120: 0.758270 s/epoch  -> 659 epochs for ~500s
#   NTK 500s integer schedule: [0, 1330, 1309, 91, 13]

#SBATCH --job-name=darcy-500s-dawn
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=AIRR-P67-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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

DATA_DIR="${DATA_DIR:-examples/pdes/darcy_flow/data}"
OUT_BASE="${OUT_BASE:-results/dawn}"
if [ -z "${PYTHON:-}" ]; then
    # shellcheck disable=SC1091
    . scripts/select_runner_python.sh
    PYTHON=$(select_runner_python "$DEVICE")
fi
BASE_RES="${BASE_RES:-241}"
SEED="${SEED:-42}"
RUNS="${RUNS:-r15 r120 ntk}"

LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
FNO_WIDTH="${FNO_WIDTH:-32}"
FNO_MODES="${FNO_MODES:-8}"
LOAD_GPU="${LOAD_GPU:-1}"
FINAL_FOURIER_RELU="${FINAL_FOURIER_RELU:-0}"
SPECTRAL="${SPECTRAL:-1}"
ADD_COORDS="${ADD_COORDS:-1}"
LOSS_REDUCTION="${LOSS_REDUCTION:-sum}"

R15_EPOCHS="${R15_EPOCHS:-9266}"
R120_EPOCHS="${R120_EPOCHS:-659}"
NTK_EPOCHS_JSON="${NTK_EPOCHS_JSON:-[0, 1330, 1309, 91, 13]}"

R15_EVAL_EVERY="${R15_EVAL_EVERY:-750}"
R120_EVAL_EVERY="${R120_EVAL_EVERY:-100}"
NTK_EVAL_EVERY="${NTK_EVAL_EVERY:-160}"

mkdir -p logs "$OUT_BASE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-24}}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

echo "Using PYTHON=${PYTHON}"
echo "Using DEVICE=${DEVICE}"
echo "Using ADD_COORDS=${ADD_COORDS}"
echo "Using LOSS_REDUCTION=${LOSS_REDUCTION}"

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
    if [ "$FINAL_FOURIER_RELU" = "1" ]; then
        args+=(--fno_final_fourier_relu)
    fi

    "${args[@]}"
}

for run in $RUNS; do
    case "$run" in
        r15)
            run_experiment "baseline_r15_500s" "[${R15_EPOCHS}]" \
                "[[15]]" "[[1024]]" "[1]" "[[160]]" "[${LR}]" \
                "$R15_EVAL_EVERY" "$SPECTRAL"
            ;;
        r120)
            run_experiment "baseline_r120_500s" "[${R120_EPOCHS}]" \
                "[[120]]" "[[1024]]" "[1]" "[[20]]" "[${LR}]" \
                "$R120_EVAL_EVERY" "$SPECTRAL"
            ;;
        ntk|ntk_optimal)
            run_experiment "ntk_optimal_500s" "$NTK_EPOCHS_JSON" \
                "[[15], [30], [60], [120], [241]]" \
                "[[1024], [1024], [1024], [1024], [1024]]" \
                "[1, 1, 1, 1, 1]" \
                "[[160], [80], [80], [20], [20]]" \
                "[${LR}, ${LR}, ${LR}, ${LR}, ${LR}]" \
                "$NTK_EVAL_EVERY" "$SPECTRAL"
            ;;
        *)
            echo "ERROR: unknown run '$run' in RUNS='$RUNS' (expected r15, r120, ntk)"
            exit 2
            ;;
    esac
done

echo
echo "Finished Darcy 500s probe: $OUT_BASE"
