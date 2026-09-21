#!/usr/bin/env python
"""
MFM (Metric Flow Matching) / OT-MFM timing benchmark.

Wraps the existing MFM training pipeline (geopath + flow matching) with
wall-clock timing. Uses WANDB_MODE=disabled to avoid network overhead.

Usage:
    WANDB_MODE=disabled conda run -n env_mfm python train_mfm_timing.py \
        --dataset eb5 --epochs 1000 --working-dir <repo_root>
"""

import argparse
import os
import sys
import time

os.environ["WANDB_MODE"] = "disabled"

import torch

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

MFM_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baselines", "metric-flow-matching"
)
sys.path.insert(0, MFM_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _runtime_patches import patch_mfm_rbf_eps

patch_mfm_rbf_eps()  # guard against zero sigma in RBFNetwork

from mfm.train.main import main as mfm_main
from mfm.train.parsers import parse_args as mfm_parse_args
from mfm.train.train_utils import (
    dataset_name2datapath,
    load_config,
    merge_config,
)

DATASET_CONFIGS = {
    "eb5": {
        "config_yaml": "configs/single_cell/5dims/ot-mfm_eb.yaml",
        "data_name": "eb",
        "dim": 5,
    },
    "eb100": {
        "config_yaml": None,
        "data_name": "eb",
        "dim": 100,
        "overrides": {
            "hidden_dims_geopath": [1024, 1024, 1024],
            "hidden_dims_flow": [1024, 1024, 1024],
            "velocity_metric": "rbf",
            "time_geopath": True,
            "whiten": False,
            "patience": 25,
            "n_centers": 150,
            "kappa": 1.5,
            "rho": -2.75,
            "alpha_metric": 1,
            "metric_epochs": 200,
            "mfm": True,
            "t_exclude": [1, 2, 3],
            "gammas": [0.125, 0.125, 0.125],
        },
    },
    "cite50": {
        "config_yaml": "configs/single_cell/50dims/ot-mfm_cite.yaml",
        "data_name": "cite",
        "dim": 50,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["eb5", "eb100", "cite50"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--working-dir",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dc = DATASET_CONFIGS[args.dataset]

    os.chdir(MFM_ROOT)

    sys.argv = ["train_mfm_timing.py"]
    mfm_args = mfm_parse_args()

    if dc.get("config_yaml"):
        config = load_config(os.path.join(MFM_ROOT, dc["config_yaml"]))
        mfm_args = merge_config(mfm_args, config)

    mfm_args.data_type = "scrna"
    mfm_args.data_name = dc["data_name"]
    mfm_args.dim = dc["dim"]
    mfm_args.epochs = args.epochs
    mfm_args.seeds = [args.seed]
    mfm_args.accelerator = "gpu"
    mfm_args.working_dir = args.working_dir

    for k, v in dc.get("overrides", {}).items():
        setattr(mfm_args, k, v)

    mfm_args.group_name = "timing_benchmark"
    mfm_args.data_path = dataset_name2datapath(mfm_args.data_name, mfm_args.working_dir)

    t_exclude_list = mfm_args.t_exclude if mfm_args.t_exclude else [None]
    t_exclude = t_exclude_list[0]
    if mfm_args.gammas:
        mfm_args.gamma_current = mfm_args.gammas[0]
    mfm_args.t_exclude_current = t_exclude
    mfm_args.seed_current = args.seed

    print(f"=== MFM {args.dataset} (dim={dc['dim']}, t_exclude={t_exclude}) ===")
    start = time.time()
    mfm_main(mfm_args, seed=args.seed, t_exclude=t_exclude)
    elapsed = time.time() - start
    print(f"\nMFM {args.dataset}: {elapsed:.2f}s total (geopath + flow, epochs={args.epochs})")


if __name__ == "__main__":
    main()
