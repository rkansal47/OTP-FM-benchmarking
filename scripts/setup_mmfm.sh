#!/bin/bash
# Setup script for MMFM environment.
# Invoke from the repo root, e.g. `bash scripts/setup_mmfm.sh`.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Setting up MMFM environment ==="

echo "Creating conda environment 'env_mmfm' (CPU-only — MMFM timing runs are CPU)..."
conda create -n env_mmfm python=3.11 -y

echo "Installing dependencies..."
conda run -n env_mmfm pip install torch --index-url https://download.pytorch.org/whl/cpu
conda run -n env_mmfm pip install numpy scipy tqdm torchdiffeq

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To use this environment:"
echo "  conda activate env_mmfm"
echo ""
echo "Training commands (timing benchmarks):"
echo "  conda run -n env_mmfm python $REPO_ROOT/runners/train_mmfm_timing.py --dataset eb5 --epochs 300"
echo "  conda run -n env_mmfm python $REPO_ROOT/runners/train_mmfm_timing.py --dataset eb100 --epochs 300"
echo "  conda run -n env_mmfm python $REPO_ROOT/runners/train_mmfm_timing.py --dataset cite50 --epochs 300"
