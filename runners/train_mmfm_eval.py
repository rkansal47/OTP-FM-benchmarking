"""
MMFM (Multi-Marginal Flow Matching) training and evaluation.

Trains MMFM with cubic spline interpolation on single-cell trajectory datasets
and evaluates with W1, W2, and MMD metrics following the OTP-FM evaluation protocol.

Model architecture matches OTP-FM's FlowNetMLP (PositionalEmbedding + LayerNorm +
SiLU + residual connections + dropout) for fair parameter-count comparison.

Evaluation spaces (matching OTP-FM):
  EB W1:  normalized (standardized) space
  EB W2:  original PCA space, first min(dim, 10) dims
  EB MMD: original PCA space
  CITE W1: original PCA space (raw; OTP-FM CSVs report the same)

Experiments:
  eb5_loo:    EB 5D leave-one-out   -> avg W1
  cite50_loo: CITE 50D leave-one-out -> avg W1
  cite5_loo:  CITE 5D  leave-one-out -> avg W1
  eb5_l2o:    EB 5D leave-two-out   -> avg W2
  eb100_loo:  EB 100D leave-one-out + train-on-all -> per-config avg MMD
"""

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.nn as nn
import torchdiffeq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OTP_FM_DIR = BASE_DIR / "OTP-FM"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "mmfm_eval"


def _import_from(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eb_data = _import_from("eb_data", OTP_FM_DIR / "experiments" / "singlecell" / "data.py")
_cite_data = _import_from("cite_data", OTP_FM_DIR / "experiments" / "citeseq" / "data.py")

EXPERIMENT_CONFIGS = {
    "eb5_loo": {
        "dataset": "eb",
        "pca_dim": 5,
        "folds": [[1], [2], [3]],
        "fold_epochs": [100, 100, 100],
        "primary_metric": "w1",
        "primary_metric_space": "normalized",
        "hidden_dim": 768,
        "num_hidden_layers": 8,
        "dropout": 0.2,
        "residual_every": 2,
        "lr": 3e-3,
        "iters_per_epoch": 100,
        "batch_size": 256,
    },
    "cite50_loo": {
        "dataset": "cite",
        "pca_dim": 50,
        "folds": [[1], [2]],
        "fold_epochs": [200, 200],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "hidden_dim": 768,
        "num_hidden_layers": 10,
        "dropout": 0.2,
        "residual_every": 2,
        "lr": 3e-3,
        "iters_per_epoch": 100,
        "batch_size": 256,
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "folds": [[1], [2]],
        "fold_epochs": [200, 200],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "hidden_dim": 768,
        "num_hidden_layers": 10,
        "dropout": 0.2,
        "residual_every": 2,
        "lr": 3e-3,
        "iters_per_epoch": 100,
        "batch_size": 256,
    },
    "eb5_l2o": {
        "dataset": "eb",
        "pca_dim": 5,
        "folds": [[1, 3]],
        "fold_epochs": [300],
        "primary_metric": "w2",
        "primary_metric_space": "normalized",
        "hidden_dim": 768,
        "num_hidden_layers": 8,
        "dropout": 0.2,
        "residual_every": 2,
        "lr": 3e-3,
        "iters_per_epoch": 100,
        "batch_size": 256,
    },
    "eb100_loo": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1], [2], [3], []],
        "fold_epochs": [500, 300, 500, 300],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "hidden_dim": 768,
        "num_hidden_layers": 8,
        "dropout": 0.2,
        "residual_every": 2,
        "lr": 1e-3,
        "iters_per_epoch": 100,
        "batch_size": 256,
    },
}


# ── Model (matches OTP-FM's FlowNetMLP architecture) ─────────────────────────


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (DDPM++ / ADM style)."""

    def __init__(self, num_channels, max_positions=10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions

    def forward(self, x):
        half = self.num_channels // 2
        freqs = torch.arange(0, half, dtype=torch.float32, device=x.device)
        freqs = freqs / (half - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.view(-1).float().outer(freqs)
        return torch.cat([x.cos(), x.sin()], dim=1)


class ResidualMLP(nn.Module):
    """MLP with pre-activation LayerNorm, SiLU, residual connections, and dropout.

    Matches OTP-FM's MLP class architecture exactly.
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, num_hidden_layers, dropout=0.0, residual_every=0
    ):
        super().__init__()
        self.residual_every = residual_every
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    *([] if dropout <= 0 else [nn.Dropout(dropout)]),
                )
            )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        h = self.input_proj(x)
        if self.residual_every > 0:
            h_res = h
            for i, layer in enumerate(self.hidden_layers):
                h = layer(h)
                if (i + 1) % self.residual_every == 0:
                    h = h + h_res
                    h_res = h
        else:
            for layer in self.hidden_layers:
                h = layer(h)
        return self.output_proj(h)


class VelocityNet(nn.Module):
    """Velocity network matching OTP-FM's FlowNetMLP parameter count.

    Architecture: x_emb(Linear) + t_emb(Positional+Linear) -> ResidualMLP -> v
    For MMFM we have no dt, so we project t_emb to 128 dims via a linear layer
    to match the 192-dim input of OTP-FM's main MLP (64 x_emb + 128 t_emb).
    """

    def __init__(
        self,
        dim,
        hidden_dim=768,
        num_hidden_layers=8,
        x_emb_dim=64,
        t_emb_dim=64,
        dropout=0.2,
        residual_every=2,
    ):
        super().__init__()
        self.x_emb = nn.Linear(dim, x_emb_dim)
        self.t_pos_emb = PositionalEmbedding(t_emb_dim)
        self.t_proj = nn.Linear(t_emb_dim, 2 * t_emb_dim)
        self.v = ResidualMLP(
            input_dim=x_emb_dim + 2 * t_emb_dim,
            hidden_dim=hidden_dim,
            output_dim=dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            residual_every=residual_every,
        )

    def forward(self, x_and_t):
        x = x_and_t[:, :-1]
        t = x_and_t[:, -1]
        x_emb = self.x_emb(x)
        t_emb = self.t_proj(self.t_pos_emb(t))
        return self.v(torch.cat([x_emb, t_emb], dim=1))


# ── Data Loading (uses OTP-FM's data infrastructure) ─────────────────────────


def load_data(dataset, pca_dim, holdout_times, ot_coupling=True):
    """Load data and optionally compute OT couplings using OTP-FM's data loaders."""
    data_dir = OTP_FM_DIR / "data"
    if dataset == "eb":
        raw = _eb_data.load_eb_data(
            data_dir=data_dir,
            pca_dim=pca_dim,
            normalize=True,
            ot_coupling=ot_coupling,
            holdout_times=holdout_times,
        )
    elif dataset == "cite":
        raw = _cite_data.load_citeseq_data(
            data_dir=OTP_FM_DIR / "data",
            pca_dim=pca_dim,
            normalize=True,
            ot_coupling=ot_coupling,
            holdout_times=holdout_times,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    all_times = sorted(raw["marginals"].keys())
    # Convert marginals to numpy for evaluation
    marginals_np = {t: raw["marginals"][t].numpy() for t in all_times}
    return {
        "marginals": marginals_np,
        "scaler": raw["scaler"],
        "all_times": all_times,
        "pcs": raw["pcs"],
        "labels": raw["labels"],
        "ot_alignments": raw["ot_alignments"],
        "train_times": raw["train_times"],
    }


def build_coupled_data(data, train_times, dataset, batch_size=256):
    """Build OT-coupled dataset using OTP-FM's Dataset classes."""
    DatasetCls = (
        _eb_data.EBMultiMarginalDataset
        if dataset == "eb"
        else _cite_data.CiteSeqMultiMarginalDataset
    )
    holdout_times = [t for t in data["all_times"] if t not in train_times]
    ds = DatasetCls(
        data["pcs"],
        data["labels"],
        holdout_times=holdout_times,
        ot_alignments=data["ot_alignments"],
    )

    # Extract the full coupled array: (n_samples, n_times, dim)
    n = len(ds)
    dim = data["pcs"].shape[1]
    n_times = len(train_times)
    coupled = np.empty((n, n_times, dim), dtype=np.float32)
    for i in range(n):
        samples = ds[i]  # list of n_times tensors
        for t_idx in range(n_times):
            coupled[i, t_idx] = samples[t_idx].numpy()

    logger.info(f"    OT-coupled data: {coupled.shape}")
    return coupled


# ── Cubic Spline Sampling ────────────────────────────────────────────────────


def sample_spline_batch(coupled_data, timepoints_norm, batch_size, device, chunk_size=32):
    """Sample (t, x_t, u_t) via cubic spline interpolation through OT-coupled trajectories."""
    from scipy import interpolate

    n_total, n_times, dim = coupled_data.shape
    idx = np.random.randint(n_total, size=batch_size)
    samples = coupled_data[idx].transpose(1, 0, 2)  # (n_times, batch, dim)

    t_vals = np.random.rand(batch_size).astype(np.float64)
    xt = np.empty((batch_size, dim), dtype=np.float32)
    ut = np.empty((batch_size, dim), dtype=np.float32)
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        c = end - start
        chunk_y = samples[:, start:end, :].reshape(n_times, c * dim)
        spline = interpolate.CubicSpline(timepoints_norm, chunk_y)
        chunk_t = t_vals[start:end]
        xt_flat = spline(chunk_t).reshape(c, c, dim)
        ut_flat = spline(chunk_t, 1).reshape(c, c, dim)
        cidx = np.arange(c)
        xt[start:end] = xt_flat[cidx, cidx].astype(np.float32)
        ut[start:end] = ut_flat[cidx, cidx].astype(np.float32)
    return (
        torch.from_numpy(t_vals.astype(np.float32)).to(device),
        torch.from_numpy(xt).to(device),
        torch.from_numpy(ut).to(device),
    )


# ── Training ──────────────────────────────────────────────────────────────────


def train_model(
    model, X_by_time, timepoints_norm, *, epochs, batch_size, lr, iters_per_epoch, device
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    model.train()
    losses = []
    t_start = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(iters_per_epoch):
            optimizer.zero_grad()
            t, xt, ut = sample_spline_batch(X_by_time, timepoints_norm, batch_size, device)
            pred = model(torch.cat([xt, t[:, None]], dim=1))
            loss = torch.mean((pred - ut) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / iters_per_epoch
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")
    return time.perf_counter() - t_start, losses


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def transport(model, source, t_start, t_end, device):
    model.eval()
    x0 = torch.from_numpy(source).float().to(device)

    def ode_fn(t, x):
        t_vec = t.expand(x.shape[0], 1)
        return model(torch.cat([x, t_vec], dim=1))

    t_span = torch.tensor([t_start, t_end], device=device)
    traj = torchdiffeq.odeint(ode_fn, x0, t_span, method="dopri5", atol=1e-5, rtol=1e-5)
    return traj[-1].cpu().numpy()


def compute_w1(gen, gt):
    n = min(len(gen), len(gt))
    g, r = gen[:n].astype(np.float64), gt[:n].astype(np.float64)
    M = ot.dist(g, r, metric="euclidean")
    a, b = np.ones(n) / n, np.ones(n) / n
    return float(ot.emd2(a, b, M, numItermax=int(1e7)))


def compute_w2(gen, gt, max_dim=10):
    d = min(gen.shape[1], max_dim)
    g, r = gen[:, :d], gt[:, :d]
    n = min(len(g), len(r))
    g, r = g[:n].astype(np.float64), r[:n].astype(np.float64)
    M = ot.dist(g, r, metric="sqeuclidean")
    a, b = np.ones(n) / n, np.ones(n) / n
    return float(np.sqrt(ot.emd2(a, b, M, numItermax=int(1e7))))


def compute_mmd(gen, gt, kernel_mul=2.0, kernel_num=5):
    source = torch.from_numpy(gen.astype(np.float32))
    target = torch.from_numpy(gt.astype(np.float32))
    n_all = len(source) + len(target)
    total = torch.cat([source, target], dim=0)
    L2 = torch.cdist(total, total, p=2).pow(2)
    bandwidth = L2.sum() / (n_all**2 - n_all)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    kernels = sum(torch.exp(-L2 / (bandwidth * kernel_mul**i)) for i in range(kernel_num))
    ns = len(source)
    XX = kernels[:ns, :ns].mean()
    YY = kernels[ns:, ns:].mean()
    XY = kernels[:ns, ns:].mean()
    YX = kernels[ns:, :ns].mean()
    return (XX + YY - XY - YX).item()


METRIC_FNS = {"w1": compute_w1, "w2": compute_w2, "mmd": compute_mmd}


def evaluate_fold(
    model,
    data,
    holdout_times,
    all_times,
    primary_metric,
    primary_metric_space,
    scaler,
    device,
    eval_all_times=False,
):
    """Evaluate model. Returns dict of {t: metric_value} and transported samples at all times."""
    t_min, t_max = min(all_times), max(all_times)

    def norm_t(t):
        return (t - t_min) / (t_max - t_min)

    if eval_all_times:
        metric_times = [t for t in all_times if t != all_times[0]]
    elif holdout_times:
        metric_times = holdout_times
    else:
        metric_times = [t for t in all_times if t != all_times[0]]

    source = data["marginals"][all_times[0]]
    metric_fn = METRIC_FNS[primary_metric]
    results = {}
    transported = {}

    # Transport to ALL non-source times (for plotting)
    for t_eval in all_times:
        if t_eval == all_times[0]:
            continue
        gen = transport(
            model, source, t_start=norm_t(all_times[0]), t_end=norm_t(t_eval), device=device
        )
        transported[t_eval] = gen

        if t_eval in metric_times:
            gt = data["marginals"][t_eval]
            gen_metric, gt_metric = gen, gt
            if primary_metric_space == "original" and scaler is not None:
                gen_metric = scaler.inverse_transform(gen)
                gt_metric = scaler.inverse_transform(gt)
            val = metric_fn(gen_metric, gt_metric)
            results[t_eval] = val
            logger.info(f"    t={t_eval}: {primary_metric}={val:.6f}")

    return results, transported


# ── Plotting (matches OTP-FM's 2-panel layout) ───────────────────────────────

TIME_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]


def plot_trajectories(
    data, transported, all_times, holdout_times, fold_label, out_dir, pcs=(0, 1), n_scatter=2000
):
    """2-panel plot matching OTP-FM: Left = Ground Truth, Right = Learned."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pc1, pc2 = pcs

    fig, (ax_gt, ax_gen) = plt.subplots(1, 2, figsize=(14, 6))

    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf

    for i, t in enumerate(all_times):
        color = TIME_COLORS[i % len(TIME_COLORS)]
        held = t in holdout_times
        label = f"t={t}" + (" *" if held else "")

        gt = data["marginals"][t]
        idx = np.random.choice(len(gt), size=min(n_scatter, len(gt)), replace=False)
        ax_gt.scatter(gt[idx, pc1], gt[idx, pc2], s=2, alpha=0.4, c=color, label=label)
        x_min = min(x_min, gt[idx, pc1].min())
        x_max = max(x_max, gt[idx, pc1].max())
        y_min = min(y_min, gt[idx, pc2].min())
        y_max = max(y_max, gt[idx, pc2].max())

        if t in transported:
            gen = transported[t]
            idx_g = np.random.choice(len(gen), size=min(n_scatter, len(gen)), replace=False)
            ax_gen.scatter(gen[idx_g, pc1], gen[idx_g, pc2], s=2, alpha=0.4, c=color, label=label)
            x_min = min(x_min, gen[idx_g, pc1].min())
            x_max = max(x_max, gen[idx_g, pc1].max())
            y_min = min(y_min, gen[idx_g, pc2].min())
            y_max = max(y_max, gen[idx_g, pc2].max())

    pad = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    lims = dict(
        xlim=(x_min - pad * x_range, x_max + pad * x_range),
        ylim=(y_min - pad * y_range, y_max + pad * y_range),
    )
    for ax, title in [(ax_gt, "Ground Truth"), (ax_gen, "MMFM Learned")]:
        ax.set(**lims)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"PC{pc1+1}", fontsize=11)
        ax.set_ylabel(f"PC{pc2+1}", fontsize=11)
        ax.legend(markerscale=4, fontsize=9, loc="best")

    fig.suptitle(fold_label, fontsize=13)
    fig.tight_layout()
    fname = out_dir / f"{fold_label.replace(' ', '_')}_pc{pc1+1}_pc{pc2+1}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    logger.info(f"    Saved trajectory plot: {fname}")


def plot_loss_curve(losses, fold_label, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss — {fold_label}")
    ax.set_yscale("log")
    fig.tight_layout()
    fname = out_dir / f"loss_{fold_label.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=100)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────


def _serialize_fold(holdout, fold_metrics):
    """Convert per-time {t: metric} into JSON-friendly dict."""
    holdout_label = "_".join(map(str, holdout)) if holdout else "none"
    return {
        "holdout": list(holdout),
        "holdout_label": holdout_label,
        "per_time": {str(t): float(v) for t, v in fold_metrics.items()},
        "fold_avg": float(np.mean(list(fold_metrics.values()))) if fold_metrics else None,
    }


def run_experiment(exp_name, seed=42, device_str="cuda", save_dir=None):
    """Train + eval all folds for one experiment with a single seed.

    Persists metrics to ``<save_dir>/<exp_name>/seed<seed>.json``.
    Plots are written next to the JSON.
    """
    cfg = EXPERIMENT_CONFIGS[exp_name]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Experiment: {exp_name} seed={seed} on {device} ===")

    if save_dir is None:
        save_dir = DEFAULT_OUTPUT_DIR
    save_dir = Path(save_dir)
    out_dir = save_dir / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"seed{seed}.json"

    dim = cfg["pca_dim"]
    eval_all_times = exp_name == "eb100_loo"
    all_fold_metrics = []
    all_train_times_sec = []

    t_run_start = time.perf_counter()
    for fold_idx, holdout in enumerate(cfg["folds"]):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load data with OT coupling for this fold's holdout
        data = load_data(cfg["dataset"], cfg["pca_dim"], holdout_times=holdout, ot_coupling=True)
        all_times = data["all_times"]
        t_min, t_max = min(all_times), max(all_times)
        train_times = data["train_times"]

        tp_norm = np.array([(t - t_min) / (t_max - t_min) for t in train_times], dtype=np.float32)

        fold_label = (
            f"{exp_name}_seed{seed}_holdout{'_'.join(map(str, holdout)) if holdout else 'none'}"
        )
        logger.info(
            f"\n--- Fold {fold_idx+1}/{len(cfg['folds'])}: holdout={holdout}, train={train_times} ---"
        )
        logger.info(f"    Normalized training timepoints: {tp_norm.tolist()}")

        coupled = build_coupled_data(data, train_times, cfg["dataset"])

        model = VelocityNet(
            dim=dim,
            hidden_dim=cfg["hidden_dim"],
            num_hidden_layers=cfg["num_hidden_layers"],
            dropout=cfg["dropout"],
            residual_every=cfg["residual_every"],
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    Model params: {n_params:,}")

        fold_epochs = cfg["fold_epochs"][fold_idx]

        train_sec, losses = train_model(
            model,
            coupled,
            tp_norm,
            epochs=fold_epochs,
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            iters_per_epoch=cfg["iters_per_epoch"],
            device=device,
        )
        all_train_times_sec.append(train_sec)
        logger.info(f"    Training time: {train_sec:.1f}s  Final loss: {losses[-1]:.6f}")

        plot_loss_curve(losses, fold_label, out_dir)

        fold_results, transported = evaluate_fold(
            model,
            data,
            holdout,
            all_times,
            primary_metric=cfg["primary_metric"],
            primary_metric_space=cfg["primary_metric_space"],
            scaler=data["scaler"],
            device=device,
            eval_all_times=eval_all_times,
        )
        all_fold_metrics.append(fold_results)

        plot_trajectories(data, transported, all_times, holdout, fold_label, out_dir, pcs=(0, 1))
        if dim >= 4:
            plot_trajectories(
                data, transported, all_times, holdout, fold_label, out_dir, pcs=(2, 3)
            )

    # Reporting
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULT: {exp_name} seed={seed}")
    logger.info(f"  Metric: {cfg['primary_metric'].upper()}")

    folds_serialized = [
        _serialize_fold(holdout, fold_res)
        for holdout, fold_res in zip(cfg["folds"], all_fold_metrics)
    ]

    if exp_name == "eb100_loo":
        # Per-config average MMD across all timepoints
        for fold_idx, (holdout, fold_res) in enumerate(zip(cfg["folds"], all_fold_metrics)):
            fold_avg = np.mean(list(fold_res.values()))
            logger.info(f"  holdout={holdout}: per-time {dict(fold_res)}")
            logger.info(f"    -> avg MMD = {fold_avg:.6f}")
        grand_avg = float(np.mean([np.mean(list(f.values())) for f in all_fold_metrics]))
        logger.info(f"  Grand avg MMD (across all configs): {grand_avg:.6f}")
    else:
        for fold_idx, (holdout, fold_res) in enumerate(zip(cfg["folds"], all_fold_metrics)):
            for t, v in fold_res.items():
                logger.info(f"    holdout={holdout}, t={t}: {v:.6f}")
        all_values = [v for fold in all_fold_metrics for v in fold.values()]
        grand_avg = float(np.mean(all_values))
        logger.info(f"  Average {cfg['primary_metric'].upper()}: {grand_avg:.6f}")

    total_sec = time.perf_counter() - t_run_start
    avg_train = float(np.mean(all_train_times_sec))
    logger.info(f"  Avg training time per fold: {avg_train:.1f}s ({avg_train/60:.1f} min)")
    logger.info(f"  Total wall-clock (incl. eval): {total_sec:.1f}s ({total_sec/60:.1f} min)")
    logger.info(f"{'='*60}")

    payload = {
        "experiment": exp_name,
        "seed": seed,
        "primary_metric": cfg["primary_metric"],
        "primary_metric_space": cfg["primary_metric_space"],
        "pca_dim": cfg["pca_dim"],
        "folds": folds_serialized,
        "grand_avg": grand_avg,
        "train_sec_per_fold": [float(s) for s in all_train_times_sec],
        "total_sec": float(total_sec),
        "config": {k: v for k, v in cfg.items() if k != "folds"},
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"  Saved metrics to {metrics_path}")

    return all_fold_metrics


def main():
    parser = argparse.ArgumentParser(description="MMFM training + evaluation")
    parser.add_argument(
        "--experiment", type=str, required=True, choices=list(EXPERIMENT_CONFIGS.keys())
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save metrics JSON and plots (default: src/.../outputs/mmfm_eval)",
    )
    args = parser.parse_args()
    run_experiment(
        args.experiment, seed=args.seed, device_str=args.device, save_dir=Path(args.save_dir)
    )


if __name__ == "__main__":
    main()
