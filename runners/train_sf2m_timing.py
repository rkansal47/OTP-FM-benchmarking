"""
Training script for SF2M (Simulation-Free Schrödinger Bridges via Score and Flow Matching)
using the "M-exact" variant from torchcfm.

M-exact = SchrodingerBridgeConditionalFlowMatcher with ot_method="exact" (Earth Mover's Distance
for minibatch OT coupling, as opposed to Sinkhorn entropic OT).

SF2M trains two networks jointly:
  - A velocity (flow) network v_t(x)
  - A score network s_t(x)

The combined loss is:
  L = E[||v_t(x_t) - u_t||^2] + E[||lambda_t * s_t(x_t) + epsilon||^2]

where lambda_t = 2*sigma_t / sigma^2, sigma_t = sigma * sqrt(t*(1-t)).

Reference: Tong et al., "Simulation-free Schrödinger bridges via score and flow matching", 2024.

Usage:
    conda run -n env_sf2m python train_sf2m_timing.py --dataset eb5 --epochs 100
    conda run -n env_sf2m python train_sf2m_timing.py --dataset eb100 --epochs 100
    conda run -n env_sf2m python train_sf2m_timing.py --dataset cite50 --epochs 100
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "baselines" / "conditional-flow-matching")
)

from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher


# ---------------------------------------------------------------------------
# MLP architecture (matches torchcfm examples: 4-layer SELU MLP)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256, time_varying: bool = True):
        super().__init__()
        in_dim = dim + (1 if time_varying else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eb_data(data_path: Path, max_dim: int, normalize: bool = True):
    """Load EB (embryoid body) data from eb_velocity_v5.npz."""
    data = np.load(data_path, allow_pickle=True)
    pcs = data["pcs"][:, :max_dim].astype(np.float32)
    labels = data["sample_labels"].astype(np.int64)

    if normalize:
        mean = pcs.mean(axis=0)
        std = pcs.std(axis=0) + 1e-8
        pcs = (pcs - mean) / std

    times = sorted(np.unique(labels).tolist())
    X = [pcs[labels == t] for t in times]
    return X, times


def load_cite_data(data_path: Path, normalize: bool = True):
    """Load CITE-seq PCA50 data from cite_pca50.csv."""
    import pandas as pd

    df = pd.read_csv(data_path)
    labels = df["samples"].values.astype(np.int64)
    feature_cols = [c for c in df.columns if c != "samples"]
    pcs = df[feature_cols].values.astype(np.float32)

    if normalize:
        mean = pcs.mean(axis=0)
        std = pcs.std(axis=0) + 1e-8
        pcs = (pcs - mean) / std

    times = sorted(np.unique(labels).tolist())
    X = [pcs[labels == t] for t in times]
    return X, times


# ---------------------------------------------------------------------------
# Batching (follows the single-cell notebook pattern)
# ---------------------------------------------------------------------------


def get_batch(
    fm: SchrodingerBridgeConditionalFlowMatcher,
    X: list[np.ndarray],
    batch_size: int,
    device: torch.device,
):
    """
    Build a batch spanning all consecutive timepoint pairs, exactly as in the
    torchcfm single-cell notebook. Returns (t, xt, ut, eps).
    """
    n_times = len(X)
    ts, xts, uts, epss = [], [], [], []

    for t_start in range(n_times - 1):
        idx0 = np.random.randint(X[t_start].shape[0], size=batch_size)
        idx1 = np.random.randint(X[t_start + 1].shape[0], size=batch_size)
        x0 = torch.from_numpy(X[t_start][idx0]).float().to(device)
        x1 = torch.from_numpy(X[t_start + 1][idx1]).float().to(device)

        t, xt, ut, eps = fm.sample_location_and_conditional_flow(x0, x1, return_noise=True)

        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
        epss.append(eps)

    return torch.cat(ts), torch.cat(xts), torch.cat(uts), torch.cat(epss)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    flow_model: nn.Module,
    score_model: nn.Module,
    fm: SchrodingerBridgeConditionalFlowMatcher,
    X: list[np.ndarray],
    n_times: int,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    iters_per_epoch: int,
    device: torch.device,
) -> dict:
    """
    Joint training of velocity and score networks for SF2M.
    Returns dict with timing and loss info.
    """
    optimizer = torch.optim.AdamW(
        list(flow_model.parameters()) + list(score_model.parameters()),
        lr=lr,
    )

    flow_model.train()
    score_model.train()

    all_losses = []
    total_iters = 0
    t_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(range(iters_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for _ in pbar:
            optimizer.zero_grad()

            t, xt, ut, eps = get_batch(fm, X, batch_size, device)

            inp = torch.cat([xt, t[:, None]], dim=-1)
            vt = flow_model(inp)
            st = score_model(inp)

            flow_loss = torch.mean((vt - ut) ** 2)

            lambda_t = fm.compute_lambda(t % 1)
            score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)

            loss = flow_loss + score_loss
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            total_iters += 1
            pbar.set_postfix(fl=f"{flow_loss.item():.3f}", sl=f"{score_loss.item():.3f}")

        avg = np.mean(epoch_losses)
        all_losses.append(avg)
        elapsed = time.perf_counter() - t_start
        print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}  elapsed={elapsed:.1f}s")

    total_time = time.perf_counter() - t_start
    return {
        "total_time_s": total_time,
        "total_iters": total_iters,
        "losses": all_losses,
        "time_per_iter_ms": total_time / total_iters * 1000 if total_iters else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SF2M M-exact training for timing benchmarks")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["eb5", "eb100", "cite50"],
        help="Dataset: eb5 (EB 5D), eb100 (EB 100D), cite50 (CITE-seq 50D)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--iters-per-epoch", type=int, default=100, help="Training iterations per epoch"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=0.1, help="SB diffusion sigma (must be > 0)")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # --- resolve data paths ---
    base_dir = Path(__file__).resolve().parents[1]  # repo root
    eb_path = base_dir / "data" / "eb_velocity_v5.npz"
    cite_path = base_dir / "OTP-FM" / "data" / "cite_pca50.csv"

    normalize = not args.no_normalize

    if args.dataset == "eb5":
        X, times = load_eb_data(eb_path, max_dim=5, normalize=normalize)
        dim = 5
    elif args.dataset == "eb100":
        X, times = load_eb_data(eb_path, max_dim=100, normalize=normalize)
        dim = 100
    elif args.dataset == "cite50":
        X, times = load_cite_data(cite_path, normalize=normalize)
        dim = 50
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    n_times = len(times)
    print(f"Dataset: {args.dataset}  dim={dim}  n_times={n_times}")
    for i, t in enumerate(times):
        print(f"  time {t}: {X[i].shape[0]} cells")

    # --- models ---
    flow_model = MLP(dim=dim, hidden_dim=args.hidden_dim, time_varying=True).to(device)
    score_model = MLP(dim=dim, hidden_dim=args.hidden_dim, time_varying=True).to(device)

    n_params = sum(p.numel() for p in flow_model.parameters()) + sum(
        p.numel() for p in score_model.parameters()
    )
    print(f"Total parameters (flow + score): {n_params:,}")

    # --- SF2M M-exact matcher ---
    fm = SchrodingerBridgeConditionalFlowMatcher(sigma=args.sigma, ot_method="exact")
    print(
        f"Flow matcher: SchrodingerBridgeConditionalFlowMatcher(sigma={args.sigma}, ot_method='exact')"
    )

    # --- train ---
    results = train(
        flow_model,
        score_model,
        fm,
        X,
        n_times,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        iters_per_epoch=args.iters_per_epoch,
        device=device,
    )

    print("\n=== Timing Results ===")
    print(f"Total time:       {results['total_time_s']:.2f} s")
    print(f"Total iterations: {results['total_iters']}")
    print(f"Time per iter:    {results['time_per_iter_ms']:.2f} ms")
    print(f"Final loss:       {results['losses'][-1]:.4f}")

    # --- save ---
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / "outputs" / "sf2m"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_path = args.output_dir / f"sf2m_mexact_{args.dataset}.pt"
    torch.save(
        {
            "flow_model_state": flow_model.state_dict(),
            "score_model_state": score_model.state_dict(),
            "config": vars(args),
            "results": results,
        },
        save_path,
    )
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
