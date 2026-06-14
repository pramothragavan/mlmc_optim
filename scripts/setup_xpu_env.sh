#!/bin/bash
# Build an Intel XPU PyTorch virtualenv for mlmc_optim.
#
# This intentionally does not use the repo's torch==2.4.1 pin. The FNO code
# relies on complex einsum in spectral layers. The known-good Dawn stack from
# the mesh experiments was Python 3.12 with torch==2.12.0+xpu.
# The post-install check verifies the XPU wheel build without requiring a
# visible device, so setup can run on CSD3 login nodes.
#
# Examples:
#   bash scripts/setup_xpu_env.sh
#   VENV=$HOME/venvs/mlmc-xpu MODULE_PURGE=1 MODULES="rhel9/default-dawn" PYTHON_BIN=/path/to/python3.12 bash scripts/setup_xpu_env.sh

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/modules.sh
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
VENV=${VENV:-$REPO/.venv-xpu}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODULES=${MODULES:-}
MODULE_PURGE=${MODULE_PURGE:-0}

TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/xpu}
TORCH_VERSION=${TORCH_VERSION:-2.12.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.27.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-none}

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
if not ((3, 10) <= version[:2] < (3, 14)):
    raise SystemExit(
        "ERROR: XPU setup needs Python 3.10-3.13 for the default "
        "torch==2.12.0 XPU wheels. Python 3.12 is the known-good Dawn "
        "choice. Got Python "
        f"{version.major}.{version.minor}.{version.micro}. "
        "Load a newer Python module or set PYTHON_BIN, for example: "
        'MODULE_PURGE=1 MODULES="rhel9/default-dawn" PYTHON_BIN=/path/to/python3.12 '
        "bash scripts/setup_xpu_env.sh --rebuild"
    )
PY

PYTHON_MINOR=$("$PYTHON_BIN" - <<'PY'
import sys
print(sys.version_info.minor)
PY
)

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

echo "Creating XPU virtualenv"
echo "  repo:       $REPO"
echo "  venv:       $VENV"
echo "  python:     $PYTHON_BIN"
echo "  python minor: 3.${PYTHON_MINOR}"
echo "  torch:      $TORCH_VERSION / torchvision $TORCHVISION_VERSION / torchaudio $TORCHAUDIO_VERSION"
echo "  torch idx:  $TORCH_INDEX_URL"

"$PYTHON_BIN" -m venv "$VENV"
PYTHON="$VENV/bin/python"

"$PYTHON" -m pip install --upgrade pip setuptools wheel

echo "Installing XPU PyTorch wheel set"
torch_packages=("torch==$TORCH_VERSION")
if [ -n "$TORCHVISION_VERSION" ] && [ "$TORCHVISION_VERSION" != "none" ]; then
    torch_packages+=("torchvision==$TORCHVISION_VERSION")
fi
if [ -n "$TORCHAUDIO_VERSION" ] && [ "$TORCHAUDIO_VERSION" != "none" ]; then
    torch_packages+=("torchaudio==$TORCHAUDIO_VERSION")
fi
"$PYTHON" -m pip install --no-cache-dir \
    "${torch_packages[@]}" \
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
import importlib
import sys

import torch

print("env python", sys.executable)
print("python_version", sys.version.split()[0])
print("torch", torch.__version__)
print("torch file", torch.__file__)
for package in ("torchvision", "torchaudio"):
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        print(f"{package} unavailable ({type(exc).__name__}: {exc})")
    else:
        print(package, module.__version__)

if "+xpu" not in torch.__version__:
    raise SystemExit("ERROR: expected an XPU torch wheel with a +xpu build tag")

xpu = getattr(torch, "xpu", None)
if xpu is None:
    raise SystemExit("ERROR: installed torch has no torch.xpu namespace")

print("xpu namespace present")
print("xpu available", xpu.is_available())
if xpu.is_available():
    inp = torch.randn(2, 4, 4, 4, device="xpu", dtype=torch.cfloat)
    weights = torch.randn(4, 8, 4, 4, device="xpu", dtype=torch.cfloat)
    out = torch.einsum("bixy,ioxy->boxy", inp, weights)
    xpu.synchronize()
    print("complex_einsum ok", tuple(out.shape))
else:
    print("complex_einsum check skipped: no XPU visible")
PY

echo "Created XPU venv at $VENV"
echo "Use PYTHON=$VENV/bin/python for XPU runs."
