#!/bin/bash
# Setup script for env_vgfm: VGFM (velocity-growth flow matching, Wang 2025).
#
# Recipe for the env_vgfm conda environment used for the paper experiments.
# Linux/CUDA only — these recipes embed cuda-* / nvidia-* wheels.
# Invoke from the repo root: `bash scripts/setup_vgfm.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Setting up env_vgfm (Python 3.10) ==="

if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "env_vgfm"; then
    echo "conda env 'env_vgfm' already exists. Reusing."
else
    conda create -n env_vgfm python=3.10 -y
fi

conda activate env_vgfm

echo "Installing pinned dependencies from scripts/requirements/env_vgfm.txt..."
pip install --upgrade pip
# The requirements file is a verbatim `pip freeze` of the environment. Any line
# starting with `-e git+...` is informational only — we install
# the corresponding submodule from baselines/ below.
grep -vE '^-e git\+' "$REPO_ROOT/scripts/requirements/env_vgfm.txt" > /tmp/env_vgfm.requirements.txt
pip install -r /tmp/env_vgfm.requirements.txt
rm -f /tmp/env_vgfm.requirements.txt

# NOTE: the runner for this method prepends baselines/<method>/
# to sys.path; no editable pip install is required.

echo ""
echo "=== env_vgfm setup complete ==="
echo "To activate: conda activate env_vgfm"
