#!/bin/bash
# Setup script for env_jkonet: JKOnet* (Terpin 2024).
#
# Recipe for the env_jkonet conda environment used for the paper experiments.
# Linux/CUDA only — these recipes embed cuda-* / nvidia-* wheels.
# Invoke from the repo root: `bash scripts/setup_jkonet.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Setting up env_jkonet (Python 3.10) ==="

if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "env_jkonet"; then
    echo "conda env 'env_jkonet' already exists. Reusing."
else
    conda create -n env_jkonet python=3.10 -y
fi

conda activate env_jkonet

echo "Installing pinned dependencies from scripts/requirements/env_jkonet.txt..."
pip install --upgrade pip
# The requirements file is a verbatim `pip freeze` of the environment. Any line
# starting with `-e git+...` is informational only — we install
# the corresponding submodule from baselines/ below.
grep -vE '^-e git\+' "$REPO_ROOT/scripts/requirements/env_jkonet.txt" > /tmp/env_jkonet.requirements.txt
pip install -r /tmp/env_jkonet.requirements.txt
rm -f /tmp/env_jkonet.requirements.txt

# NOTE: the runner for this method prepends baselines/<method>/
# to sys.path; no editable pip install is required.

echo ""
echo "=== env_jkonet setup complete ==="
echo "To activate: conda activate env_jkonet"
