#!/usr/bin/env python
"""
MIOFlow timing benchmark.

Trains the full MIOFlow pipeline (GAGA autoencoder + Neural ODE) and reports
wall-clock training time per epoch.

Usage:
    conda run -n env_mioflow python train_mioflow_timing.py --dataset eb5 --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines" / "MIOFlow"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runtime_patches import patch_mioflow_odeint
from mioflow.gaga import fit_gaga
from mioflow.mioflow import MIOFlow

patch_mioflow_odeint()  # force RK4 step_size=0.1 (paper-run integrator config)


def load_data(dataset: str, data_dir: str) -> anndata.AnnData:
    if dataset.startswith("eb"):
        dim = int(dataset[2:])
        raw = np.load(f"{data_dir}/eb_velocity_v5.npz", allow_pickle=True)
        pca = raw["pcs"][:, :dim].astype(np.float32)
        phate = raw["phate"].astype(np.float32)
        labels = raw["sample_labels"].astype(int)
    elif dataset == "cite50":
        dim = 50
        raw = np.load(f"{data_dir}/cite_pca50.npz")
        pca = raw["pca"].astype(np.float32)
        phate = None
        labels = raw["sample_labels"].astype(int)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    adata = anndata.AnnData(X=np.zeros((pca.shape[0], 1), dtype=np.float32))
    adata.obsm["X_pca"] = pca
    adata.obs["time_bin"] = pd.Categorical(labels)
    if phate is not None:
        adata.obsm["X_phate"] = phate[:, :2]
    else:
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
        sc.tl.umap(adata)
        adata.obsm["X_phate"] = adata.obsm["X_umap"][:, :2].astype(np.float32)

    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["eb5", "eb100", "cite50"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gaga-encoder-epochs", type=int, default=300)
    parser.add_argument("--gaga-decoder-epochs", type=int, default=300)
    parser.add_argument("--ode-epochs", type=int, default=300)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = args.device == "cuda" and torch.cuda.is_available()

    data_dir = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "data")
    adata = load_data(args.dataset, data_dir)
    dim = adata.obsm["X_pca"].shape[1]
    print(f"Dataset: {args.dataset}  dim={dim}  cells={adata.n_obs}")

    # Phase 1: GAGA
    gaga_start = time.time()
    gaga_model = fit_gaga(
        X_pca=adata.obsm["X_pca"],
        X_phate=adata.obsm["X_phate"],
        latent_dim=min(dim, 2),
        hidden_dims=[128, 64],
        batch_size=1024,
        encoder_epochs=args.gaga_encoder_epochs,
        decoder_epochs=args.gaga_decoder_epochs,
        learning_rate=1e-3,
    )
    gaga_elapsed = time.time() - gaga_start
    gaga_total_epochs = args.gaga_encoder_epochs + args.gaga_decoder_epochs
    print(
        f"GAGA: {gaga_elapsed:.2f}s for {gaga_total_epochs} epochs = {gaga_elapsed/gaga_total_epochs*1000:.2f} ms/epoch"
    )

    gaga_model.cpu()

    # Phase 2: MIOFlow ODE
    ode_start = time.time()
    mf = MIOFlow(
        adata,
        gaga_model=gaga_model,
        obs_time_key="time_bin",
        debug_level="warning",
        hidden_dim=64,
        use_cuda=use_cuda,
        momentum_beta=0.0,
        scheduler_type="cosine",
        learning_rate=1e-3,
        scheduler_t_max=args.ode_epochs,
        scheduler_min_lr=1e-5,
        growth_rate_model=None,
        n_epochs=args.ode_epochs,
        use_density_loss=False,
        lambda_ot=1.0,
        lambda_energy=0.1,
        energy_time_steps=20,
        sample_size=args.sample_size,
        n_trajectories=0,
        n_bins=10,
        exp_dir="/tmp/mioflow_timing",
    )
    mf.fit()
    ode_elapsed = time.time() - ode_start
    print(
        f"ODE: {ode_elapsed:.2f}s for {args.ode_epochs} epochs = {ode_elapsed/args.ode_epochs*1000:.2f} ms/epoch"
    )

    total = gaga_elapsed + ode_elapsed
    total_epochs = gaga_total_epochs + args.ode_epochs
    print(f"\n=== MIOFlow {args.dataset} ===")
    print(f"Total: {total:.2f}s for {total_epochs} total epochs")
    print(
        f"  GAGA:    {gaga_elapsed:.2f}s ({gaga_total_epochs} ep, {gaga_elapsed/gaga_total_epochs*1000:.2f} ms/ep)"
    )
    print(
        f"  ODE:     {ode_elapsed:.2f}s ({args.ode_epochs} ep, {ode_elapsed/args.ode_epochs*1000:.2f} ms/ep)"
    )
    print(f"  Overall: {total/total_epochs*1000:.2f} ms/epoch")


if __name__ == "__main__":
    main()
