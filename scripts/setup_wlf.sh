#!/bin/bash
# Setup script for wlf: WLF-UOT (Wasserstein Lagrangian Flows, Neklyudov 2024).
#
# Recipe for the wlf conda environment used for the paper experiments.
# Linux/CUDA only — these recipes embed cuda-* / nvidia-* wheels.
# Invoke from the repo root: `bash scripts/setup_wlf.sh`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Setting up wlf (Python 3.10) ==="

if ! command -v conda > /dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "wlf"; then
    echo "conda env 'wlf' already exists. Reusing."
else
    conda create -n wlf python=3.10 -y
fi

conda activate wlf

echo "Installing pinned dependencies from scripts/requirements/wlf.txt..."
pip install --upgrade pip
# The requirements file is a verbatim `pip freeze` of the environment. Any line
# starting with `-e git+...` is informational only — we install
# the corresponding submodule from baselines/ below.
grep -vE '^-e git\+' "$REPO_ROOT/scripts/requirements/wlf.txt" > /tmp/wlf.requirements.txt
pip install -r /tmp/wlf.requirements.txt
rm -f /tmp/wlf.requirements.txt

# NOTE: the runner for this method prepends baselines/<method>/
# to sys.path; no editable pip install is required.

echo ""
echo "=== wlf setup complete ==="
echo "To activate: conda activate wlf"
