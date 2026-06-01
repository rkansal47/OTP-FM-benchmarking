#!/bin/bash
# Setup script for env_ijkonet: iJKOnet (adversarial inverse-JKO, Persiianov 2026).
#
# Recipe for the env_ijkonet conda environment used for the paper experiments.
# Linux/CUDA only — these recipes embed cuda-* / nvidia-* wheels.
# Invoke from the repo root: `bash scripts/setup_ijkonet.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Setting up env_ijkonet (Python 3.10) ==="

if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "env_ijkonet"; then
    echo "conda env 'env_ijkonet' already exists. Reusing."
else
    conda create -n env_ijkonet python=3.10 -y
fi

conda activate env_ijkonet

echo "Installing pinned dependencies from scripts/requirements/env_ijkonet.txt..."
pip install --upgrade pip
# The requirements file is a verbatim `pip freeze` of the environment. Any line
# starting with `-e git+...` is informational only — we install
# the corresponding submodule from baselines/ below.
grep -vE '^-e git\+' "$REPO_ROOT/scripts/requirements/env_ijkonet.txt" > /tmp/env_ijkonet.requirements.txt
pip install -r /tmp/env_ijkonet.requirements.txt
rm -f /tmp/env_ijkonet.requirements.txt

# NOTE: the runner for this method prepends baselines/<method>/
# to sys.path; no editable pip install is required.

echo ""
echo "=== env_ijkonet setup complete ==="
echo "To activate: conda activate env_ijkonet"
