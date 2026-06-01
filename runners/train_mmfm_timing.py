"""
Training script for MMFM (Multi-Marginal Flow Matching) timing benchmarks.

Uses cubic spline interpolation through multiple marginals to define a
conditional velocity field, then trains a simple MLP to regress onto that
velocity (MSE loss).  This is the core MMFM approach from:

    Rohbeck et al., "Multi-Marginal Flow Matching for Single Cell Perturbation Prediction", 2024.

Supports three dataset configurations:
  - eb5:    Embryoid Body, 5 PCA dims,  timepoints [0,1,2,3,4]
  - eb100:  Embryoid Body, 100 PCA dims, timepoints [0,1,2,3,4]
  - cite50: CITE-seq, 50 PCA dims, timepoints [0,1,2,3]

Usage:
    conda run -n env_mmfm python train_mmfm_timing.py --dataset eb5 --epochs 300
    conda run -n env_mmfm python train_mmfm_timing.py --dataset eb100 --epochs 300
    conda run -n env_mmfm python train_mmfm_timing.py --dataset cite50 --epochs 300
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import interpolate
from tqdm import tqdm


# ---------------------------------------------------------------------------
# MLP velocity network (matches the architecture in train_mmfm_eb.py)
# ---------------------------------------------------------------------------


class SimpleVelocityNet(nn.Module):
    """Simple MLP velocity network for MMFM.  Input: [x, t]  Output: v."""

    def __init__(self, dim: int, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim + 1, hidden_dim))
        layers.append(nn.SELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eb_data(data_path: Path, max_dim: int, normalize: bool = True):
    """Load EB (embryoid body) data.  Returns (X_by_time, times)."""
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
    """Load CITE-seq PCA50 data from cite_pca50.npz.  Returns (X_by_time, times)."""
    data = np.load(data_path)
    pcs = data["pca"].astype(np.float32)
    labels = data["sample_labels"].astype(np.int64)

    if normalize:
        mean = pcs.mean(axis=0)
        std = pcs.std(axis=0) + 1e-8
        pcs = (pcs - mean) / std

    times = sorted(np.unique(labels).tolist())
    X = [pcs[labels == t] for t in times]
    return X, times


# ---------------------------------------------------------------------------
# Multi-marginal flow matching via cubic spline
# ---------------------------------------------------------------------------


def sample_flow_matching(
    xs: torch.Tensor,
    timepoints: np.ndarray,
    sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample (t, x_t, u_t) for multi-marginal flow matching using cubic splines.

    Args:
        xs: (batch_size, n_times, dim) samples at each marginal
        timepoints: (n_times,) normalised time locations in [0, 1]
        sigma: optional Gaussian noise variance around the spline mean

    Returns:
        t:  (batch_size,)  uniformly sampled in [0,1]
        xt: (batch_size, dim)  interpolated location
        ut: (batch_size, dim)  conditional velocity (spline derivative)
    """
    batch_size, n_times, dim = xs.shape
    device = xs.device

    t = torch.rand(batch_size, device=device)

    xs_np = xs.cpu().numpy()
    t_np = t.cpu().numpy()

    xt = np.zeros((batch_size, dim))
    ut = np.zeros((batch_size, dim))

    for i in range(batch_size):
        spline = interpolate.CubicSpline(timepoints, xs_np[i])
        xt[i] = spline(t_np[i])
        ut[i] = spline(t_np[i], 1)

    if sigma > 0:
        xt += np.random.randn(*xt.shape) * sigma

    return (
        t,
        torch.from_numpy(xt).float().to(device),
        torch.from_numpy(ut).float().to(device),
    )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def get_batch(
    X: list[np.ndarray],
    timepoints_norm: np.ndarray,
    batch_size: int,
    device: torch.device,
    sigma: float = 0.0,
):
    """
    Build one multi-marginal batch.

    Samples `batch_size` cells from each timepoint, stacks them into
    (batch_size, n_times, dim), then calls the spline-based flow matcher.

    Returns (t, xt, ut).
    """
    samples = []
    for i in range(len(X)):
        idx = np.random.randint(X[i].shape[0], size=batch_size)
        samples.append(X[i][idx])

    xs = np.stack(samples, axis=1)  # (batch_size, n_times, dim)
    xs = torch.from_numpy(xs).float().to(device)

    t, xt, ut = sample_flow_matching(xs, timepoints_norm, sigma=sigma)
    return t, xt, ut


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    X: list[np.ndarray],
    timepoints_norm: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    iters_per_epoch: int,
    device: torch.device,
    sigma: float = 0.0,
) -> dict:
    """Train MMFM velocity model.  Returns timing & loss dict."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model.train()
    all_losses: list[float] = []
    total_iters = 0
    t_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_losses: list[float] = []
        pbar = tqdm(range(iters_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for _ in pbar:
            optimizer.zero_grad()

            t, xt, ut = get_batch(X, timepoints_norm, batch_size, device, sigma)

            inp = torch.cat([xt, t[:, None]], dim=1).to(device)
            vt = model(inp)
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            total_iters += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg = np.mean(epoch_losses)
        all_losses.append(avg)
        elapsed = time.perf_counter() - t_start
        print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.6f}  elapsed={elapsed:.1f}s")

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
    parser = argparse.ArgumentParser(description="MMFM training for timing benchmarks")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["eb5", "eb100", "cite50"],
        help="Dataset: eb5 (EB 5D), eb100 (EB 100D), cite50 (CITE-seq 50D)",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--iters-per-epoch", type=int, default=100, help="Training iterations per epoch"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--sigma", type=float, default=0.0, help="Flow variance (0 = deterministic spline)"
    )
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
    cite_path = base_dir / "data" / "cite_pca50.npz"

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
    t_min, t_max = min(times), max(times)
    timepoints_norm = np.array([(t - t_min) / (t_max - t_min) for t in times])

    print(f"Dataset: {args.dataset}  dim={dim}  n_times={n_times}")
    print(f"Raw timepoints: {times}")
    print(f"Normalised timepoints: {timepoints_norm}")
    for i, t in enumerate(times):
        print(f"  time {t}: {X[i].shape[0]} cells")

    # --- model ---
    model = SimpleVelocityNet(
        dim=dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- train ---
    results = train(
        model,
        X,
        timepoints_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        iters_per_epoch=args.iters_per_epoch,
        device=device,
        sigma=args.sigma,
    )

    print("\n=== Timing Results ===")
    print(f"Total time:       {results['total_time_s']:.2f} s")
    print(f"Total iterations: {results['total_iters']}")
    print(f"Time per iter:    {results['time_per_iter_ms']:.2f} ms")
    print(f"Final loss:       {results['losses'][-1]:.6f}")

    # --- save ---
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / "outputs" / "mmfm"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_path = args.output_dir / f"mmfm_{args.dataset}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "results": results,
        },
        save_path,
    )
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
