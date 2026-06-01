#!/bin/bash
# Setup script for env_deepruot: DeepRUOT (regularized unbalanced OT, Zhang 2025).
#
# Recipe for the env_deepruot conda environment used for the paper experiments.
# Linux/CUDA only — these recipes embed cuda-* / nvidia-* wheels.
# Invoke from the repo root: `bash scripts/setup_deepruot.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Setting up env_deepruot (Python 3.10) ==="

if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "env_deepruot"; then
    echo "conda env 'env_deepruot' already exists. Reusing."
else
    conda create -n env_deepruot python=3.10 -y
fi

conda activate env_deepruot

# Pre-patch DeepRUOT/setup.py to use modern setuptools (drop the
# pkg_resources.parse_version assertion).
sed -i.bak \
    -e 's/^from pkg_resources import parse_version$/import setuptools/' \
    -e '/^assert parse_version(setuptools.__version__)/d' \
    "$REPO_ROOT/baselines/DeepRUOT/setup.py"

echo "Installing pinned dependencies from scripts/requirements/env_deepruot.txt..."
pip install --upgrade pip
# The requirements file is a verbatim `pip freeze` of the environment. Any line
# starting with `-e git+...` is informational only — we install
# the corresponding submodule from baselines/ below.
grep -vE '^-e git\+' "$REPO_ROOT/scripts/requirements/env_deepruot.txt" > /tmp/env_deepruot.requirements.txt
pip install -r /tmp/env_deepruot.requirements.txt
rm -f /tmp/env_deepruot.requirements.txt

# Install the DeepRUOT submodule editable from the local
# checkout so the runtime monkey-patches in runners/_runtime_patches.py
# operate on the same files git tracks under our pinned commit.
pip install --no-deps -e "$REPO_ROOT/baselines/DeepRUOT"

echo ""
echo "=== env_deepruot setup complete ==="
echo "To activate: conda activate env_deepruot"
