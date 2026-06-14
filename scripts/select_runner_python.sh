#!/bin/bash
# Helper for runner templates: choose an interpreter created by setup_*_env.sh.
#
# Source this file, then call:
#   PYTHON=$(select_runner_python "$DEVICE")
#
# Override paths with CUDA_PYTHON, XPU_PYTHON, CUDA_VENV, XPU_VENV, or PYTHON.

runner_python_has_cuda() {
    local python="$1"
    [ -x "$python" ] || return 1
    "$python" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

runner_python_has_xpu() {
    local python="$1"
    [ -x "$python" ] || return 1
    "$python" - <<'PY' >/dev/null 2>&1
import torch
xpu = getattr(torch, "xpu", None)
raise SystemExit(
    0 if xpu is not None and hasattr(xpu, "is_available") and xpu.is_available()
    else 1
)
PY
}

select_runner_python() {
    local device="${1:-auto}"
    local repo="${REPO:-$PWD}"
    local cuda_venv="${CUDA_VENV:-$repo/.venv-cuda}"
    local xpu_venv="${XPU_VENV:-$repo/.venv-xpu}"
    local cuda_python="${CUDA_PYTHON:-$cuda_venv/bin/python}"
    local xpu_python="${XPU_PYTHON:-$xpu_venv/bin/python}"
    local fallback="${PYTHON_FALLBACK:-python}"

    if [ -n "${PYTHON:-}" ]; then
        echo "$PYTHON"
        return 0
    fi

    case "$device" in
        cuda)
            if [ -x "$cuda_python" ]; then
                echo "$cuda_python"
            else
                echo "$fallback"
            fi
            ;;
        xpu)
            if [ -x "$xpu_python" ]; then
                echo "$xpu_python"
            else
                echo "$fallback"
            fi
            ;;
        auto|gpu)
            if runner_python_has_cuda "$cuda_python"; then
                echo "$cuda_python"
            elif runner_python_has_xpu "$xpu_python"; then
                echo "$xpu_python"
            elif [ -x "$cuda_python" ]; then
                echo "$cuda_python"
            elif [ -x "$xpu_python" ]; then
                echo "$xpu_python"
            else
                echo "$fallback"
            fi
            ;;
        *)
            echo "$fallback"
            ;;
    esac
}
