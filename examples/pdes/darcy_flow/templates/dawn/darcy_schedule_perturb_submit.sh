#!/bin/bash
# Submit all schedule perturbation conditions as separate SLURM jobs.
#
# Empirical per-epoch costs on Dawn (XPU, single PVC):
#   r15: 0.13s   r30: 0.22s   r60: 0.25s   r120: 1.16s   r241: 96s
#
# Estimated training wall-clock (+ headroom for eval/spectral):
#   ntk:        ~0.5h  ->  2h
#   rank_flat:  ~15h   -> 20h
#   rank_steep: ~0.5h  ->  2h
#   uniform:    ~19h   -> 24h
#   reversed:   ~36h   -> 42h
#
# Usage:
#   bash darcy_schedule_perturb_submit.sh
#   bash darcy_schedule_perturb_submit.sh rank_flat reversed   # subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/darcy_schedule_perturb.sh"

if [ $# -gt 0 ]; then
    RUNS=("$@")
else
    RUNS=(ntk rank_flat rank_steep uniform reversed)
fi

get_time_limit() {
    case "$1" in
        ntk|rank_steep) echo "02:00:00" ;;
        rank_flat)      echo "20:00:00" ;;
        uniform)        echo "24:00:00" ;;
        reversed)       echo "42:00:00" ;;
        *)              echo "06:00:00" ;;
    esac
}

for run in "${RUNS[@]}"; do
    tlimit=$(get_time_limit "$run")
    echo "Submitting: $run (--time=$tlimit)"
    sbatch --job-name="sched-${run}" --time="$tlimit" "$SCRIPT" "RUN=${run}"
done

echo
echo "Submitted ${#RUNS[@]} jobs"
