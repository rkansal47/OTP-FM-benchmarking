#!/bin/bash
# Setup script for env_sf2m: [SF]^2M / I-CFM / OT-CFM (torchcfm, Tong 2023/2024).
#
# Recipe for the env_sf2m conda environment used for the paper experiments.
# Linux/CUDA only — these recipes embed cuda-* / nvidia-* wheels.
# Invoke from the repo root: `bash scripts/setup_sf2m.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Setting up env_sf2m (Python 3.10) ==="

if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "env_sf2m"; then
    echo "conda env 'env_sf2m' already exists. Reusing."
else
    conda create -n env_sf2m python=3.10 -y
fi

conda activate env_sf2m

echo "Installing pinned dependencies from scripts/requirements/env_sf2m.txt..."
pip install --upgrade pip
# The requirements file is a verbatim `pip freeze` of the environment. Any line
# starting with `-e git+...` is informational only — we install
# the corresponding submodule from baselines/ below.
grep -vE '^-e git\+' "$REPO_ROOT/scripts/requirements/env_sf2m.txt" > /tmp/env_sf2m.requirements.txt
pip install -r /tmp/env_sf2m.requirements.txt
rm -f /tmp/env_sf2m.requirements.txt

# Install the conditional-flow-matching submodule editable from the local
# checkout so the runtime monkey-patches in runners/_runtime_patches.py
# operate on the same files git tracks under our pinned commit.
pip install --no-deps -e "$REPO_ROOT/baselines/conditional-flow-matching"

echo ""
echo "=== env_sf2m setup complete ==="
echo "To activate: conda activate env_sf2m"
