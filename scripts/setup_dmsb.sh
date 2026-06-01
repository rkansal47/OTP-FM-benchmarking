#!/bin/bash
# Setup script for env_dmsb: DMSB (deep momentum Schrodinger bridge, Chen 2023).
#
# Recipe for the env_dmsb conda environment used for the paper experiments.
# Linux/CUDA only — these recipes embed cuda-* / nvidia-* wheels.
# Invoke from the repo root: `bash scripts/setup_dmsb.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Setting up env_dmsb (Python 3.10) ==="

if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "env_dmsb"; then
    echo "conda env 'env_dmsb' already exists. Reusing."
else
    conda create -n env_dmsb python=3.10 -y
fi

conda activate env_dmsb

echo "Installing pinned dependencies from scripts/requirements/env_dmsb.txt..."
pip install --upgrade pip
# The requirements file is a verbatim `pip freeze` of the environment. Any line
# starting with `-e git+...` is informational only — we install
# the corresponding submodule from baselines/ below.
grep -vE '^-e git\+' "$REPO_ROOT/scripts/requirements/env_dmsb.txt" > /tmp/env_dmsb.requirements.txt
pip install -r /tmp/env_dmsb.requirements.txt
rm -f /tmp/env_dmsb.requirements.txt

# NOTE: the runner for this method prepends baselines/<method>/
# to sys.path; no editable pip install is required.

echo ""
echo "=== env_dmsb setup complete ==="
echo "To activate: conda activate env_dmsb"
