#!/usr/bin/env python
"""
NLSB (Neural Lagrangian Schrödinger Bridge) timing benchmark.

Runs a few epochs of NLSB training using the paper's scRNA configs,
measures per-epoch wall-clock time, and extrapolates to full training.

Usage:
    conda run -n env_mmfm python -u train_nlsb_timing.py --dataset eb5
    conda run -n env_mmfm python -u train_nlsb_timing.py --dataset eb100
    conda run -n env_mmfm python -u train_nlsb_timing.py --dataset cite50
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

NLSB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baselines", "NLSB"
)
sys.path.insert(0, NLSB_DIR)
os.chdir(NLSB_DIR)

from dataset import BalancedBatchSampler, scRNASeq
from model import LAGRANGIAN_NAME, SDE_MODEL_NAME, SDENet

CONFIGS = {
    "eb5": "config/rna/NLSB/D/train.json",
    "eb100": "config/rna/NLSB/D/train_eb100.json",
    "cite50": "config/rna/NLSB/D/train_cite50.json",
}

TOTAL_EPOCHS = {
    "eb5": 2500,
    "eb100": 2500,
    "cite50": 2500,
}


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["eb5", "eb100", "cite50"])
    parser.add_argument("--timing-epochs", type=int, default=10, help="Number of epochs to time")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config_path = CONFIGS[args.dataset]
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["seed"] = 57

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fix_seed(cfg["seed"])
    print(f"=== NLSB {args.dataset} (dim={cfg['dataset']['dim']}) ===")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Timing epochs: {args.timing_epochs}")

    tr_ds = scRNASeq(
        [cfg["dataset"]["train_data_path"]],
        cfg["dataset"]["dim"],
        use_v=cfg["dataset"]["use_v"],
        LMT=cfg["LMT"],
    )
    va_ds = scRNASeq(
        [cfg["dataset"]["val_data_path"]],
        cfg["dataset"]["dim"],
        use_v=cfg["dataset"]["use_v"],
        LMT=cfg["LMT"],
        scaler=tr_ds.get_scaler(),
    )

    t_set = tr_ds.get_label_set()
    train_t_set = t_set[:]

    batch_sampler_tr = BalancedBatchSampler(tr_ds, cfg["dataset"]["batch_size"])
    batch_sampler_va = BalancedBatchSampler(va_ds, cfg["dataset"]["val_batch_size"])
    tr_dl = DataLoader(tr_ds, batch_sampler=batch_sampler_tr)
    va_dl = DataLoader(va_ds, batch_sampler=batch_sampler_va)

    L = LAGRANGIAN_NAME["cellular"](
        tr_ds.full_data["X"],
        tr_ds.full_data["t"],
        **cfg["lagrangian"],
        device=device,
    )
    net = SDE_MODEL_NAME["ito"](**cfg["model"], lagrangian=L)
    model = SDENet(net, device)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Train batches/epoch: {len(tr_dl)}")
    print(f"Val batches/epoch: {len(va_dl)}")

    optimizer = optim.Adam(model.parameters_lr(), lr=cfg["optim"]["lr"])

    epoch_times = []
    for epoch in range(1, args.timing_epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        # Training
        outputs = []
        for batch_idx, train_batch in enumerate(tr_dl):
            train_batch["base"] = tr_ds.base_sample(cfg["dataset"]["batch_size"])
            optimizer.zero_grad()
            out = model.training_step(train_batch, batch_idx, train_t_set, tr_ds.T0)
            outputs.append(out)
            loss = out["loss"]
            loss.backward()
            del train_batch
            optimizer.step()
            if hasattr(model, "clamp_parameters"):
                model.clamp_parameters()

        train_result = model.training_epoch_end(outputs)

        # Validation
        val_outputs = []
        for batch_idx, val_batch in enumerate(va_dl):
            val_batch["base"] = va_ds.base_sample(cfg["dataset"]["val_batch_size"])
            out = model.validation_step(val_batch, batch_idx, train_t_set, va_ds.T0)
            val_outputs.append(out)
            del val_batch

        model.validation_epoch_end(val_outputs)
        emd_result = model.validation(va_ds, train_t_set)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        epoch_times.append(elapsed)

        print(
            f"Epoch {epoch}/{args.timing_epochs}: "
            f"{elapsed:.2f}s  train_loss={train_result['avg_loss']:.4f}  "
            f"avg_emd={emd_result['avg_emd']:.4f}"
        )

    avg_epoch = np.mean(epoch_times)
    std_epoch = np.std(epoch_times)
    total_epochs = TOTAL_EPOCHS[args.dataset]
    extrapolated_min = avg_epoch * total_epochs / 60.0

    print(f"\n=== NLSB {args.dataset} Timing Summary ===")
    print(f"Per-epoch: {avg_epoch:.2f} +/- {std_epoch:.2f} s")
    print(f"Total epochs (paper): {total_epochs}")
    print(f"Extrapolated total: {extrapolated_min:.1f} min ({extrapolated_min/60:.1f} h)")


if __name__ == "__main__":
    main()
