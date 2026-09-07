#!/usr/bin/env python
"""
Standalone VGFM (Velocity-Growth Flow Matching) training script for timing benchmarks.

Extracts the training pipeline from the VGFM notebooks (eb5, eb50, cite50)
and exposes it as a CLI tool with configurable dataset, epochs, and device.

Usage:
    python train_vgfm_timing.py --dataset eb5   --n-pretrain-epochs 2000 --n-train-epochs 30
    python train_vgfm_timing.py --dataset eb100  --n-pretrain-epochs 2000 --n-train-epochs 30
    python train_vgfm_timing.py --dataset cite50 --n-pretrain-epochs 5000 --n-train-epochs 30
"""

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

VGFM_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "baselines", "VGFM")
sys.path.insert(0, VGFM_ROOT)

from VGFM.losses import OT_loss1
from VGFM.models import FNet
from VGFM.train import pretrain, train

DATASET_CONFIGS = {
    "eb5": dict(
        csv="eb_pca50.csv",
        dim=5,
        hidden_dim=128,
        n_hiddens=3,
        activation="leakyrelu",
        n_pretrain_epochs=2000,
        n_train_epochs=30,
        batch_size=256,
        lr_pretrain=1e-3,
        lr_train=1e-4,
        reg=0.01,
        reg_m=5,
        norm_cost=True,
        hold_one_out=False,
        hold_out=-1,
        scale_data=True,
    ),
    "eb50": dict(
        csv="eb_pca50.csv",
        dim=50,
        hidden_dim=256,
        n_hiddens=5,
        activation="leakyrelu",
        n_pretrain_epochs=2000,
        n_train_epochs=30,
        batch_size=256,
        lr_pretrain=1e-3,
        lr_train=1e-4,
        reg=0.01,
        reg_m=5,
        norm_cost=True,
        hold_one_out=False,
        hold_out=-1,
        scale_data=False,
    ),
    "eb100": dict(
        csv="eb_pca50.csv",
        dim=100,
        hidden_dim=256,
        n_hiddens=5,
        activation="leakyrelu",
        n_pretrain_epochs=2000,
        n_train_epochs=30,
        batch_size=256,
        lr_pretrain=1e-3,
        lr_train=1e-4,
        reg=0.01,
        reg_m=5,
        norm_cost=True,
        hold_one_out=False,
        hold_out=-1,
        scale_data=False,
    ),
    "cite50": dict(
        csv="cite_pca50.csv",
        dim=50,
        hidden_dim=256,
        n_hiddens=5,
        activation="leakyrelu",
        n_pretrain_epochs=5000,
        n_train_epochs=30,
        batch_size=256,
        lr_pretrain=1e-3,
        lr_train=1e-4,
        reg=0.01,
        reg_m=5,
        norm_cost=True,
        hold_one_out=False,
        hold_out=-1,
        scale_data=False,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="VGFM timing benchmark")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument(
        "--n-pretrain-epochs", type=int, default=None, help="Override pretrain epochs"
    )
    parser.add_argument(
        "--n-train-epochs", type=int, default=None, help="Override ODE-train epochs (0 to skip)"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory for saving model/logs"
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.txt")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s:%(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def load_data(cfg, data_dir):
    csv_path = os.path.join(data_dir, cfg["csv"])
    df = pd.read_csv(csv_path)
    dim = cfg["dim"]
    df = df.iloc[:, : dim + 1]

    if cfg["scale_data"]:
        from sklearn.preprocessing import StandardScaler

        cols = [c for c in df.columns if c != "samples"]
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])

    return df


def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset].copy()

    if args.n_pretrain_epochs is not None:
        cfg["n_pretrain_epochs"] = args.n_pretrain_epochs
    if args.n_train_epochs is not None:
        cfg["n_train_epochs"] = args.n_train_epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    data_dir = args.data_dir or os.path.join(VGFM_ROOT, "data")
    output_dir = args.output_dir or os.path.join(VGFM_ROOT, "results", f"timing_{args.dataset}")

    set_seed(args.seed)
    logger = setup_logger(output_dir, f"vgfm_{args.dataset}")

    print(f"{'='*60}")
    print(f"VGFM Timing Benchmark — {args.dataset}")
    print(f"{'='*60}")
    print(f"  dim             = {cfg['dim']}")
    print(f"  hidden_dim      = {cfg['hidden_dim']}")
    print(f"  n_hiddens       = {cfg['n_hiddens']}")
    print(f"  activation      = {cfg['activation']}")
    print(f"  pretrain_epochs = {cfg['n_pretrain_epochs']}")
    print(f"  train_epochs    = {cfg['n_train_epochs']}")
    print(f"  batch_size      = {cfg['batch_size']}")
    print(f"  lr_pretrain     = {cfg['lr_pretrain']}")
    print(f"  lr_train        = {cfg['lr_train']}")
    print(f"  reg             = {cfg['reg']}")
    print(f"  reg_m           = [{cfg['reg_m']}, inf]")
    print(f"  norm_cost       = {cfg['norm_cost']}")
    print(f"  hold_one_out    = {cfg['hold_one_out']}")
    print(f"  hold_out        = {cfg['hold_out']}")
    print(f"  scale_data      = {cfg['scale_data']}")
    print(f"  device          = {device}")
    print(f"  output_dir      = {output_dir}")
    print(f"{'='*60}")

    # --- Load data ---
    logger.info(f"Loading dataset: {args.dataset}")
    df = load_data(cfg, data_dir)
    print(f"Data shape: {df.shape}")
    print(f"Time points: {sorted(df['samples'].unique())}")
    print(f"Samples per time: {dict(df.groupby('samples').size())}")

    # --- Build model ---
    f_net = FNet(
        in_out_dim=cfg["dim"],
        hidden_dim=cfg["hidden_dim"],
        n_hiddens=cfg["n_hiddens"],
        activation=cfg["activation"],
    ).to(device)
    n_params = sum(p.numel() for p in f_net.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Compute relative mass ---
    sample_sizes = df.groupby("samples").size()
    relative_mass = torch.tensor((sample_sizes / sample_sizes.iloc[0]).values, dtype=torch.float32)
    print(f"Relative mass: {relative_mass.tolist()}")

    groups = sorted([g for g in df.samples.unique() if g != cfg["hold_out"]])

    # ========== PRETRAIN (flow + growth matching) ==========
    print(f"\n{'='*60}")
    print("Phase 1: Pretrain (flow + growth matching)")
    print(f"{'='*60}")
    optimizer = torch.optim.Adam(f_net.parameters(), lr=cfg["lr_pretrain"])

    t_start = time.time()
    f_net, v_losses, g_losses, losses = pretrain(
        f_net,
        df,
        optimizer,
        n_epoch=cfg["n_pretrain_epochs"],
        hold_out=cfg["hold_out"],
        logger=logger,
        device=device,
        relative_mass=relative_mass,
        reg=cfg["reg"],
        reg_m=[cfg["reg_m"], np.inf],
        norm_cost=cfg["norm_cost"],
        batch_size=cfg["batch_size"],
    )
    pretrain_time = time.time() - t_start
    print(f"\nPretrain completed in {pretrain_time:.2f}s")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Final vloss: {v_losses[-1]:.6f}")
    print(f"  Final gloss: {g_losses[-1]:.6f}")

    pretrain_model_path = os.path.join(output_dir, "pretrain_best_model")
    torch.save(f_net.state_dict(), pretrain_model_path)

    # ========== TRAIN (ODE-based, optional) ==========
    train_time = 0.0
    if cfg["n_train_epochs"] > 0:
        print(f"\n{'='*60}")
        print("Phase 2: Train (ODE-based refinement)")
        print(f"{'='*60}")
        f_net.load_state_dict(torch.load(pretrain_model_path, map_location=device))
        optimizer2 = torch.optim.Adam(f_net.parameters(), lr=cfg["lr_train"])

        criterion = OT_loss1()
        sample_size = (df[df["samples"] == 0.0].values.shape[0],)
        initial_size = df[df["samples"] == 0].iloc[:, 1].shape[0]

        t_start = time.time()
        l_loss, b_loss, g_loss = train(
            f_net,
            df,
            groups,
            optimizer2,
            cfg["n_train_epochs"],
            criterion=criterion,
            use_cuda=(device.type == "cuda"),
            apply_losses_in_time=True,
            hold_one_out=cfg["hold_one_out"],
            hold_out=cfg["hold_out"],
            sample_size=sample_size,
            relative_mass=relative_mass,
            initial_size=initial_size,
            sample_with_replacement=False,
            logger=logger,
            device=device,
            best_model_path=os.path.join(output_dir, "best_model"),
            stepsize=0.1,
        )
        train_time = time.time() - t_start
        print(f"\nTrain completed in {train_time:.2f}s")
        if b_loss:
            print(f"  Final batch loss: {b_loss[-1]:.6f}")

    # ========== Summary ==========
    total_time = pretrain_time + train_time
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset:        {args.dataset}")
    print(f"  Pretrain time:  {pretrain_time:.2f}s ({cfg['n_pretrain_epochs']} epochs)")
    if cfg["n_train_epochs"] > 0:
        print(f"  Train time:     {train_time:.2f}s ({cfg['n_train_epochs']} epochs)")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"{'='*60}")

    results = {
        "dataset": args.dataset,
        "dim": cfg["dim"],
        "hidden_dim": cfg["hidden_dim"],
        "n_hiddens": cfg["n_hiddens"],
        "n_pretrain_epochs": cfg["n_pretrain_epochs"],
        "n_train_epochs": cfg["n_train_epochs"],
        "batch_size": cfg["batch_size"],
        "pretrain_time_s": pretrain_time,
        "train_time_s": train_time,
        "total_time_s": total_time,
        "final_pretrain_loss": losses[-1] if losses else None,
        "n_params": n_params,
        "device": str(device),
    }

    import json

    results_path = os.path.join(output_dir, "timing_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
