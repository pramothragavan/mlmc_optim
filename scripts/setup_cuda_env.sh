#!/bin/bash
# Build a CUDA PyTorch virtualenv for mlmc_optim.
#
# Defaults follow this repo's current torch pins, but install from the CUDA
# wheel index. Override versions or the CUDA wheel index with environment vars.
#
# Examples:
#   bash scripts/setup_cuda_env.sh
#   VENV=$HOME/venvs/mlmc-cuda MODULE_PURGE=1 MODULES="rhel8/default-amp" PYTHON_BIN=/path/to/python3.11 bash scripts/setup_cuda_env.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 TORCH_VERSION=2.7.0 TORCHVISION_VERSION=0.22.0 TORCHAUDIO_VERSION=2.7.0 bash scripts/setup_cuda_env.sh

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/modules.sh
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
VENV=${VENV:-$REPO/.venv-cuda}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODULES=${MODULES:-}
MODULE_PURGE=${MODULE_PURGE:-0}

TORCH_VERSION=${TORCH_VERSION:-2.4.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.19.1}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-2.4.1}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

INSTALL_EDITABLE=${INSTALL_EDITABLE:-1}

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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: could not find PYTHON_BIN=$PYTHON_BIN"
    exit 2
fi

"$PYTHON_BIN" - <<'PY'
import sys

version = sys.version_info
if not ((3, 9) <= version[:2] < (3, 13)):
    raise SystemExit(
        "ERROR: CUDA setup needs Python 3.9-3.12 for this repo's "
        "torch==2.4.1 wheels. Got Python "
        f"{version.major}.{version.minor}.{version.micro}. "
        "Load a newer Python module if available, or set PYTHON_BIN, for example: "
        'MODULE_PURGE=1 MODULES="rhel8/default-amp" PYTHON_BIN=/path/to/python3.11 '
        "bash scripts/setup_cuda_env.sh --rebuild"
    )
PY

if [ "${1:-}" = "--rebuild" ] && [ -e "$VENV" ]; then
    mv "$VENV" "$VENV.bad.$(date +%Y%m%d-%H%M%S)"
elif [ -e "$VENV" ]; then
    echo "ERROR: $VENV already exists."
    echo "Run with --rebuild to move it aside and create a clean one."
    exit 2
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

mkdir -p "$(dirname "$VENV")"

echo "Creating CUDA virtualenv"
echo "  repo:       $REPO"
echo "  venv:       $VENV"
echo "  python:     $PYTHON_BIN"
echo "  torch:      $TORCH_VERSION / torchvision $TORCHVISION_VERSION / torchaudio $TORCHAUDIO_VERSION"
echo "  torch idx:  $TORCH_INDEX_URL"

"$PYTHON_BIN" -m venv "$VENV"
PYTHON="$VENV/bin/python"

"$PYTHON" -m pip install --upgrade pip setuptools wheel

echo "Installing CUDA PyTorch wheel set"
"$PYTHON" -m pip install --no-cache-dir \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX_URL"

REQ_TMP=$(mktemp)
trap 'rm -f "$REQ_TMP"' EXIT

awk '
    /^[[:space:]]*(#|$)/ { print; next }
    {
        line = $0
        sub(/^[[:space:]]+/, "", line)
        lower = tolower(line)
        if (lower ~ /^torch([<=>!~ ].*)?$/) next
        if (lower ~ /^torchvision([<=>!~ ].*)?$/) next
        if (lower ~ /^torchaudio([<=>!~ ].*)?$/) next
        print
    }
' "$REPO/requirements.txt" > "$REQ_TMP"

echo "Installing non-torch requirements"
"$PYTHON" -m pip install --no-cache-dir -r "$REQ_TMP"

if [ "$INSTALL_EDITABLE" = "1" ]; then
    "$PYTHON" -m pip install --no-deps -e "$REPO"
fi

"$PYTHON" - <<'PY'
import sys

import torch
import torchaudio
import torchvision

print("env python", sys.executable)
print("python_version", sys.version.split()[0])
print("torch", torch.__version__)
print("torch file", torch.__file__)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)

if "+cu" not in torch.__version__:
    raise SystemExit("ERROR: expected a CUDA torch wheel with a +cu build tag")

if torch.cuda.is_available():
    print("cuda device", torch.cuda.get_device_name(0))
PY

echo "Created CUDA venv at $VENV"
echo "Use PYTHON=$VENV/bin/python for CUDA runs."
