#!/bin/bash
# Setup script for env_mfm: OT-MFM (metric flow matching, Kapusniak 2024).
#
# Recipe for the env_mfm conda environment used for the paper experiments.
# Linux/CUDA only — these recipes embed cuda-* / nvidia-* wheels.
# Invoke from the repo root: `bash scripts/setup_mfm.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Setting up env_mfm (Python 3.11) ==="

if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "env_mfm"; then
    echo "conda env 'env_mfm' already exists. Reusing."
else
    conda create -n env_mfm python=3.11 -y
fi

conda activate env_mfm

echo "Installing pinned dependencies from scripts/requirements/env_mfm.txt..."
pip install --upgrade pip
# The requirements file is a verbatim `pip freeze` of the environment. Any line
# starting with `-e git+...` is informational only — we install
# the corresponding submodule from baselines/ below.
grep -vE '^-e git\+' "$REPO_ROOT/scripts/requirements/env_mfm.txt" > /tmp/env_mfm.requirements.txt
pip install -r /tmp/env_mfm.requirements.txt
rm -f /tmp/env_mfm.requirements.txt

# NOTE: the runner for this method prepends baselines/<method>/
# to sys.path; no editable pip install is required.

echo ""
echo "=== env_mfm setup complete ==="
echo "To activate: conda activate env_mfm"
