#!/bin/bash
#SBATCH --job-name=darcy-baseline-ampere
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=COMPUTERLAB-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

if [ -z "${PYTHON:-}" ]; then
  # shellcheck disable=SC1091
  . scripts/select_runner_python.sh
  PYTHON=$(select_runner_python "$DEVICE")
fi

DATA_DIR="${DATA_DIR:-examples/pdes/darcy_flow/data}"
OUT_DIR="${OUT_DIR:-results/darcy_ampere_baseline_r241_seed42}"
ADD_COORDS="${ADD_COORDS:-1}"
LOSS_REDUCTION="${LOSS_REDUCTION:-sum}"

echo "Using PYTHON=${PYTHON}"
echo "Using DEVICE=${DEVICE}"
echo "Using ADD_COORDS=${ADD_COORDS}"
echo "Using LOSS_REDUCTION=${LOSS_REDUCTION}"

coord_args=(--add_coords)
if [ "$ADD_COORDS" != "1" ]; then
  coord_args=(--no-add_coords)
fi

"${PYTHON}" examples/pdes/darcy_flow/runners/run_experiment.py \
  --seed 42 \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --loss_reduction "${LOSS_REDUCTION}" \
  --epochs_per_phase_json '[500]' \
  --c2f_res_per_phase_json '[[241]]' \
  --subset_size_per_phase_json '[[1024]]' \
  --levels_per_phase_json '[1]' \
  --batch_size_per_phase_json '[[20]]' \
  "${coord_args[@]}" \
  --save_final_checkpoint
