#!/bin/bash
# Setup script for env_mioflow: MIOFlow (GAGA + neural ODE, Huguet 2022).
#
# Recipe for the env_mioflow conda environment used for the paper experiments.
# Linux/CUDA only — these recipes embed cuda-* / nvidia-* wheels.
# Invoke from the repo root: `bash scripts/setup_mioflow.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Setting up env_mioflow (Python 3.10) ==="

if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "env_mioflow"; then
    echo "conda env 'env_mioflow' already exists. Reusing."
else
    conda create -n env_mioflow python=3.10 -y
fi

conda activate env_mioflow

echo "Installing pinned dependencies from scripts/requirements/env_mioflow.txt..."
pip install --upgrade pip
# The requirements file is a verbatim `pip freeze` of the environment. Any line
# starting with `-e git+...` is informational only — we install
# the corresponding submodule from baselines/ below.
grep -vE '^-e git\+' "$REPO_ROOT/scripts/requirements/env_mioflow.txt" > /tmp/env_mioflow.requirements.txt
pip install -r /tmp/env_mioflow.requirements.txt
rm -f /tmp/env_mioflow.requirements.txt

# Install the MIOFlow submodule editable from the local
# checkout so the runtime monkey-patches in runners/_runtime_patches.py
# operate on the same files git tracks under our pinned commit.
pip install --no-deps -e "$REPO_ROOT/baselines/MIOFlow"

echo ""
echo "=== env_mioflow setup complete ==="
echo "To activate: conda activate env_mioflow"
