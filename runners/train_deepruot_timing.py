#!/usr/bin/env python3
"""
Training time benchmark for DeepRUOT on EB and CITE-seq datasets.

DeepRUOT = "Learning stochastic dynamics from snapshots through
regularized unbalanced optimal transport" (ICLR'25 oral)

Full training pipeline (4 phases, following the emt.ipynb notebook):
  Phase 1: velocity + growth  (train_un1, 30 iters)
  Phase 2: velocity only      (train_un1, growth frozen, 10 iters)
  Phase 3: score pretraining   (scoreNet2 via SchrodingerBridgeCFM, 3001 iters)
  Phase 4: joint training      (train_all with PINN loss, 10 iters)

Usage:
    python train_deepruot_timing.py --dataset eb5   --device cuda
    python train_deepruot_timing.py --dataset eb100  --device cuda
    python train_deepruot_timing.py --dataset cite50 --device cuda
"""

import argparse
import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

DEEPRUOT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "baselines", "DeepRUOT")
sys.path.insert(0, DEEPRUOT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _runtime_patches import shim_tqdm_notebook  # noqa: E402

shim_tqdm_notebook()  # DeepRUOT/train.py uses `from tqdm.notebook import tqdm`

from DeepRUOT.models import FNet, scoreNet2  # noqa: E402
from DeepRUOT.losses import OT_loss1  # noqa: E402
from DeepRUOT.train import train_un1, train_all  # noqa: E402
from DeepRUOT.utils import (  # noqa: E402
    generate_state_trajectory,
    get_batch,
    SchrodingerBridgeConditionalFlowMatcher,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EB_DATA_PATH = os.path.join(REPO_ROOT, "OTP-FM", "data", "eb_velocity_v5.npz")
CITE_DATA_PATH = os.path.join(REPO_ROOT, "OTP-FM", "data", "cite_pca50.csv")


def load_eb_data(n_dims: int) -> pd.DataFrame:
    data = np.load(EB_DATA_PATH, allow_pickle=True)
    pcs = data["pcs"][:, :n_dims]
    labels = data["sample_labels"]
    cols = {f"x{i+1}": pcs[:, i] for i in range(n_dims)}
    cols["samples"] = labels.astype(float)
    return pd.DataFrame(cols)


def load_cite_data(n_dims: int = 50) -> pd.DataFrame:
    df = pd.read_csv(CITE_DATA_PATH)
    df["samples"] = df["samples"].astype(float)
    if n_dims < 50:
        keep = ["samples"] + [f"x{i+1}" for i in range(n_dims)]
        df = df[keep]
    return df


def compute_relative_mass(df: pd.DataFrame) -> torch.Tensor:
    sample_sizes = df.groupby("samples").size()
    ref0 = sample_sizes / sample_sizes.iloc[0]
    return torch.tensor(ref0.values)


def build_X_list(df, dim, groups):
    """Build list of numpy arrays per timepoint (for score pretraining)."""
    X = []
    for g in range(len(groups)):
        mask = df["samples"] == groups[g]
        cols = [f"x{i+1}" for i in range(dim)]
        X.append(df.loc[mask, cols].values.astype(np.float32))
    return X


def build_datatime0(df, dim):
    """Build the t=0 data tensor used by density1 / PINN loss."""
    mask = df["samples"] == 0
    cols = [f"x{i+1}" for i in range(dim)]
    return torch.tensor(df.loc[mask, cols].values, dtype=torch.float32)


def run_training(args):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if args.dataset == "eb5":
        df = load_eb_data(5)
        dim = 5
        tag = "EB-5D"
    elif args.dataset == "eb100":
        df = load_eb_data(100)
        dim = 100
        tag = "EB-100D"
    elif args.dataset == "cite50":
        df = load_cite_data(50)
        dim = 50
        tag = "CITE-50D"
    elif args.dataset == "cite5":
        df = load_cite_data(5)
        dim = 5
        tag = "CITE-5D"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    groups = sorted(df["samples"].unique())
    n_time_steps = len(groups) - 1
    n_times = len(groups)
    device = torch.device(args.device)

    print(f"Dataset: {tag}")
    print(f"  Cells: {len(df)}, Dims: {dim}, Time points: {groups}")
    print(f"  Device: {device}")

    use_cuda = device.type == "cuda"

    # ---- Model ----
    f_net = FNet(in_out_dim=dim, hidden_dim=128, n_hiddens=3, activation="leakyrelu")
    f_net = f_net.to(device)

    criterion = OT_loss1(which="emd", use_cuda=use_cuda)
    initial_size = len(df[df["samples"] == 0])
    relative_mass = compute_relative_mass(df)
    sample_size = (initial_size,)

    results_dir = os.path.join(os.path.dirname(__file__), "deepruot_timing_results")
    os.makedirs(results_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, f"best_{args.dataset}")

    phase_times = {}

    # ==================================================================
    # Phase 1: velocity + growth (30 iterations)
    # ==================================================================
    p1_iters = args.p1_iters
    print(f"\n{'='*60}")
    print(f"Phase 1: Train velocity + growth ({p1_iters} iterations)")
    print(f"{'='*60}")

    optimizer1 = optim.Adam(f_net.parameters(), lr=1e-3)
    t0 = time.time()

    train_un1(
        f_net,
        df,
        groups,
        optimizer1,
        p1_iters,
        criterion=criterion,
        use_cuda=use_cuda,
        local_loss=True,
        global_loss=False,
        apply_losses_in_time=True,
        hold_one_out=False,
        hold_out="random",
        hinge_value=0.01,
        lambda_ot=0.1,
        lambda_mass=1.0,
        lambda_energy=0.001,
        use_pinn=False,
        use_penalty=False,
        use_density_loss=False,
        lambda_density=10.0,
        top_k=5,
        sample_size=sample_size,
        relative_mass=relative_mass,
        initial_size=initial_size,
        sample_with_replacement=False,
        device=device,
        best_model_path=best_model_path,
    )

    phase_times["p1"] = time.time() - t0
    print(f"Phase 1 time: {phase_times['p1']:.2f}s")

    # ==================================================================
    # Phase 2: velocity only, growth frozen (10 iterations)
    # ==================================================================
    p2_iters = args.p2_iters
    print(f"\n{'='*60}")
    print(f"Phase 2: Train velocity only ({p2_iters} iterations)")
    print(f"{'='*60}")

    for param in f_net.g_net.parameters():
        param.requires_grad = False

    optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, f_net.parameters()), lr=1e-3)
    t0 = time.time()

    train_un1(
        f_net,
        df,
        groups,
        optimizer2,
        p2_iters,
        criterion=criterion,
        use_cuda=use_cuda,
        local_loss=True,
        global_loss=False,
        apply_losses_in_time=True,
        hold_one_out=False,
        hold_out="random",
        hinge_value=0.01,
        lambda_ot=0.1,
        lambda_mass=0.0,
        lambda_energy=0.001,
        use_pinn=False,
        use_penalty=False,
        use_density_loss=False,
        lambda_density=10.0,
        top_k=5,
        sample_size=sample_size,
        relative_mass=relative_mass,
        initial_size=initial_size,
        sample_with_replacement=False,
        device=device,
        best_model_path=best_model_path,
    )

    phase_times["p2"] = time.time() - t0
    print(f"Phase 2 time: {phase_times['p2']:.2f}s")

    # Unfreeze growth for later phases
    for param in f_net.g_net.parameters():
        param.requires_grad = True

    # ==================================================================
    # Phase 3: score model pretraining (3001 iterations)
    # ==================================================================
    p3_iters = args.p3_iters
    sigma = 0.05
    print(f"\n{'='*60}")
    print(f"Phase 3: Score model pretraining ({p3_iters} iterations)")
    print(f"{'='*60}")

    X = build_X_list(df, dim, groups)
    batch_size_score = len(df[df["samples"] == 0])
    time_tensor = torch.Tensor(groups)

    f_net.eval()
    with torch.no_grad():
        trajectory = generate_state_trajectory(
            X, n_times, batch_size_score, f_net, time_tensor, device
        )
    f_net.train()

    SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
    sf2m_score_model = (
        scoreNet2(in_out_dim=dim, hidden_dim=128, activation="leakyrelu").float().to(device)
    )
    sf2m_optimizer = optim.Adam(sf2m_score_model.parameters(), lr=1e-4)

    lambda_penalty = 0
    t0 = time.time()

    for i in tqdm(range(p3_iters), desc="Score pretraining"):
        sf2m_optimizer.zero_grad()
        t_batch, xt, ut, eps = get_batch(
            SF2M, X, trajectory, batch_size_score, n_times, return_noise=True
        )
        t_batch = torch.unsqueeze(t_batch, 1)
        lambda_t = SF2M.compute_lambda(t_batch % 1)
        xt = xt.to(device)
        t_batch = t_batch.to(device)
        eps = eps.to(device)
        lambda_t = lambda_t.to(device)

        value_st = sf2m_score_model(t_batch, xt)
        st = sf2m_score_model.compute_gradient(t_batch, xt)
        positive_st = torch.relu(value_st)
        penalty = lambda_penalty * torch.max(positive_st)
        score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
        loss = score_loss + penalty
        loss.backward()
        sf2m_optimizer.step()

    phase_times["p3"] = time.time() - t0
    print(f"Phase 3 time: {phase_times['p3']:.2f}s")

    # ==================================================================
    # Phase 4: joint training with PINN loss (10 iterations)
    # ==================================================================
    p4_iters = args.p4_iters
    print(f"\n{'='*60}")
    print(f"Phase 4: Joint training with PINN ({p4_iters} iterations)")
    print(f"{'='*60}")

    datatime0 = build_datatime0(df, dim)

    optimizer4 = optim.SGD(list(f_net.parameters()) + list(sf2m_score_model.parameters()), lr=1e-5)
    t0 = time.time()

    train_all(
        f_net,
        df,
        groups,
        optimizer4,
        p4_iters,
        criterion=criterion,
        use_cuda=use_cuda,
        local_loss=True,
        global_loss=False,
        apply_losses_in_time=True,
        hold_one_out=False,
        hold_out="random",
        sf2m_score_model=sf2m_score_model,
        hinge_value=0.01,
        datatime0=datatime0,
        device=device,
        lambda_initial=0.1,
        use_pinn=True,
        use_penalty=True,
        use_density_loss=False,
        lambda_density=10.0,
        top_k=5,
        sample_size=sample_size,
        relative_mass=relative_mass,
        initial_size=initial_size,
        sample_with_replacement=False,
        sigmaa=sigma,
        lambda_pinn=1,
    )

    phase_times["p4"] = time.time() - t0
    print(f"Phase 4 time: {phase_times['p4']:.2f}s")

    # ==================================================================
    # Summary
    # ==================================================================
    total = sum(phase_times.values())
    total_grad_steps = (
        p1_iters * n_time_steps + p2_iters * n_time_steps + p3_iters + p4_iters * n_time_steps
    )

    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY: {tag}")
    print(f"{'='*60}")
    print(f"  Phase 1 (v+g, {p1_iters} iters):     {phase_times['p1']:.2f}s")
    print(f"  Phase 2 (v only, {p2_iters} iters):   {phase_times['p2']:.2f}s")
    print(f"  Phase 3 (score, {p3_iters} iters):    {phase_times['p3']:.2f}s")
    print(f"  Phase 4 (joint, {p4_iters} iters):    {phase_times['p4']:.2f}s")
    print(f"  Total training time: {total:.2f}s  ({total/60:.2f} min)")
    print(f"  Total gradient steps: {total_grad_steps}")


def main():
    parser = argparse.ArgumentParser(description="DeepRUOT training time benchmark (all 4 phases)")
    parser.add_argument(
        "--dataset",
        choices=["eb5", "eb100", "cite50", "cite5"],
        required=True,
    )
    parser.add_argument(
        "--p1-iters", type=int, default=30, help="Phase 1 (v+g) iterations. Default: 30"
    )
    parser.add_argument(
        "--p2-iters", type=int, default=10, help="Phase 2 (v only) iterations. Default: 10"
    )
    parser.add_argument(
        "--p3-iters",
        type=int,
        default=3001,
        help="Phase 3 (score pretraining) iterations. Default: 3001",
    )
    parser.add_argument(
        "--p4-iters", type=int, default=10, help="Phase 4 (joint / PINN) iterations. Default: 10"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu, cuda, cuda:0, etc.",
    )
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
