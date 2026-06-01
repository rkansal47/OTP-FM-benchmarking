#!/bin/bash
# Setup script for 3MSBM environment.
# Invoke from the repo root, e.g. `bash scripts/setup_3msbm.sh`.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMODULE_DIR="$REPO_ROOT/baselines/3MSBM"

if [ ! -d "$SUBMODULE_DIR" ]; then
    echo "ERROR: 3MSBM submodule not found at $SUBMODULE_DIR" >&2
    echo "Run 'git submodule update --init baselines/3MSBM' first." >&2
    exit 1
fi

echo "=== Setting up 3MSBM environment ==="

echo "Creating conda environment 'env_3msbm'..."
conda create -n env_3msbm python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate env_3msbm

echo "Installing PyTorch (CUDA 11.8 wheel)..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "Installing 3MSBM requirements..."
(cd "$SUBMODULE_DIR" && pip install -r requirements.txt)

echo "Installing extras..."
pip install ipdb colored_traceback

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To use this environment:"
echo "  conda activate env_3msbm"
echo ""
echo "To train on EB data:"
echo "  conda run -n env_3msbm python runners/run_3msbm_original.py --dim 100 --epochs 40"
