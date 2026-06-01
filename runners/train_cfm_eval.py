"""
Unified evaluation harness for torchcfm-style flow matching baselines:
  --method icfm   : I-CFM  (Tong et al., 2024) — independent CFM, no OT coupling
  --method otcfm  : OT-CFM (Tong et al., 2024) — minibatch exact OT coupling
  --method sf2m   : SF^2M  (Tong et al., 2023) — Schrodinger Bridge CFM (velocity + score)

Trains one velocity model per fold using `torchcfm`'s
`{Conditional, ExactOptimalTransport, SchrodingerBridge}ConditionalFlowMatcher`
classes, then evaluates with W1 / W2 / MMD using the same protocol as
`train_mmfm_eval.py` (which is the OTP-FM evaluation convention).

Output JSONs follow the MMFM format and live at
    results/<method>/<experiment>/seed<N>.json
so that `scripts/aggregate_*.py` and the table generator can consume them.

The per-segment training loop (one (t_i, t_{i+1}) pair at a time) follows
`train_sf2m_timing.py`. The global velocity is rescaled by 1 / segment_duration
so that integrating dx/dt = v(x, t_global) over t_global in [0, 1] reproduces
the expected sample evolution.
"""

import argparse
import importlib.util
import json
import logging
import sys
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
DEFAULT_OUTPUT_DIR_TPL = Path(__file__).resolve().parent / "outputs" / "{method}_eval"
TORCHCFM_DIR = Path(__file__).resolve().parents[1] / "baselines" / "conditional-flow-matching"
sys.path.insert(0, str(TORCHCFM_DIR))

from torchcfm.conditional_flow_matching import (  # noqa: E402
    ConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _traj_utils import save_trajectory_and_checkpoint_torchdiffeq  # noqa: E402


# ── Import OTP-FM data loaders without triggering the experiments package ───


def _import_from(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eb_data = _import_from("eb_data", OTP_FM_DIR / "experiments" / "singlecell" / "data.py")
_cite_data = _import_from("cite_data", OTP_FM_DIR / "experiments" / "citeseq" / "data.py")


# ── Experiment configs (mirror train_mmfm_eval.py + new eb100_l2o) ──────────


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
    # NEW — EB 100D leave-two-out (holdout = {t1, t3}). Eval at all non-source
    # times (t1, t2, t3, t4) in MMD; the table generator computes
    # Avg MMD = (t1 + t3 + 2*rest)/4 from these.
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1, 3]],
        "fold_epochs": [500],
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


# ── Model (matches OTP-FM's FlowNetMLP architecture; copy of MMFM script) ──


class PositionalEmbedding(nn.Module):
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


# ── Data loading (identical to train_mmfm_eval.py) ──────────────────────────


def load_data(dataset, pca_dim, holdout_times, ot_coupling=False):
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


# ── Training (segment-wise CFM/SF2M with global-time velocity rescaling) ────


def make_flow_matcher(method, sigma=0.1):
    """Return the appropriate torchcfm matcher for the chosen method.
    For otcfm with precomputed couplings, we use the plain CFM since the
    OT pairing is already handled in sample_segment_batch."""
    if method in ("icfm", "otcfm"):
        # otcfm uses precomputed couplings, so at training time we just
        # need the standard conditional flow (linear interpolation + noise)
        return ConditionalFlowMatcher(sigma=sigma)
    if method == "sf2m":
        return SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="exact")
    raise ValueError(f"Unknown CFM method: {method}")


def sample_segment_batch(
    method, fm, X_by_train_time, tp_norm, batch_size, device, ot_alignments=None, train_times=None
):
    """Sample (t_global, xt, ut_global, eps) across all consecutive train-time
    segments and concatenate. Returns 4 tensors of shape (batch_size * n_segs,).
    The velocity ut is divided by the segment duration so that
    dx/dt_global = v(x, t_global) holds globally.

    For otcfm: uses precomputed ot_alignments to pair source->target instead
    of random pairing. This replaces the expensive per-batch OT solve.
    """
    n_train = len(X_by_train_time)
    ts, xts, uts, epss = [], [], [], []

    for seg in range(n_train - 1):
        x0_arr = X_by_train_time[seg]
        x1_arr = X_by_train_time[seg + 1]
        idx0 = np.random.randint(x0_arr.shape[0], size=batch_size)

        if method == "otcfm" and ot_alignments is not None:
            # Use precomputed OT mapping: idx1 = mapping[idx0]
            t_src = train_times[seg]
            t_tgt = train_times[seg + 1]
            mapping = ot_alignments[(t_src, t_tgt)]
            idx1 = mapping[idx0]
        else:
            idx1 = np.random.randint(x1_arr.shape[0], size=batch_size)

        x0 = torch.from_numpy(x0_arr[idx0]).float().to(device)
        x1 = torch.from_numpy(x1_arr[idx1]).float().to(device)

        t_local, xt, ut_local, eps = fm.sample_location_and_conditional_flow(
            x0,
            x1,
            return_noise=True,
        )
        seg_lo = float(tp_norm[seg])
        seg_hi = float(tp_norm[seg + 1])
        seg_dur = seg_hi - seg_lo
        t_global = seg_lo + t_local * seg_dur
        ut_global = ut_local / seg_dur
        ts.append(t_global)
        xts.append(xt)
        uts.append(ut_global)
        epss.append(eps)

    return (
        torch.cat(ts),
        torch.cat(xts),
        torch.cat(uts),
        torch.cat(epss),
    )


def train_model(
    method,
    flow_model,
    score_model,
    X_by_train_time,
    tp_norm,
    *,
    epochs,
    batch_size,
    lr,
    iters_per_epoch,
    sigma,
    device,
    ot_alignments=None,
    train_times=None,
):
    fm = make_flow_matcher(method, sigma=sigma)

    params = list(flow_model.parameters())
    if method == "sf2m":
        params = params + list(score_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5,
    )

    flow_model.train()
    if score_model is not None:
        score_model.train()

    losses = []
    t_start = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(iters_per_epoch):
            optimizer.zero_grad()
            t, xt, ut, eps = sample_segment_batch(
                method,
                fm,
                X_by_train_time,
                tp_norm,
                batch_size,
                device,
                ot_alignments=ot_alignments,
                train_times=train_times,
            )
            inp = torch.cat([xt, t[:, None]], dim=1)
            v_pred = flow_model(inp)
            flow_loss = torch.mean((v_pred - ut) ** 2)
            if method == "sf2m":
                # The SB lambda is in segment-local time: torchcfm uses
                # t in [0, 1] for compute_lambda. We need to recover the
                # local time inside each segment to apply the right weight.
                # Since each segment got the same shape of `t_local` (uniform
                # in [0, 1]), the concatenation order is segment-by-segment;
                # decode it from the global t and tp_norm bin edges.
                t_local_dec = []
                bs = batch_size
                for seg in range(len(tp_norm) - 1):
                    seg_lo = float(tp_norm[seg])
                    seg_hi = float(tp_norm[seg + 1])
                    chunk = t[seg * bs : (seg + 1) * bs]
                    t_local_dec.append((chunk - seg_lo) / (seg_hi - seg_lo))
                t_local_all = torch.cat(t_local_dec)
                t_local_all = t_local_all.clamp(1e-4, 1 - 1e-4)
                s_pred = score_model(inp)
                lambda_t = fm.compute_lambda(t_local_all)
                score_loss = torch.mean((lambda_t[:, None] * s_pred + eps) ** 2)
                loss = flow_loss + score_loss
            else:
                loss = flow_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / iters_per_epoch
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")
    return time.perf_counter() - t_start, losses


# ── Evaluation (matches MMFM script) ────────────────────────────────────────


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


# ── Plotting (lightweight, mirrors MMFM script) ─────────────────────────────


TIME_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]


def plot_loss_curve(losses, fold_label, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"Training Loss — {fold_label}")
    fig.tight_layout()
    fig.savefig(out_dir / f"loss_{fold_label.replace(' ', '_')}.png", dpi=100)
    plt.close(fig)


def plot_trajectories(
    data, transported, all_times, holdout_times, fold_label, out_dir, pcs=(0, 1), n_scatter=2000
):
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
    xr = x_max - x_min
    yr = y_max - y_min
    lims = dict(
        xlim=(x_min - pad * xr, x_max + pad * xr), ylim=(y_min - pad * yr, y_max + pad * yr)
    )
    for ax, title in [(ax_gt, "Ground Truth"), (ax_gen, "Generated")]:
        ax.set(**lims)
        ax.set_title(title)
        ax.set_xlabel(f"PC{pc1+1}")
        ax.set_ylabel(f"PC{pc2+1}")
        ax.legend(markerscale=4, fontsize=8)
    fig.suptitle(fold_label)
    fig.tight_layout()
    fig.savefig(out_dir / f"{fold_label.replace(' ', '_')}_pc{pc1+1}_pc{pc2+1}.png", dpi=100)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────


def _serialize_fold(holdout, fold_metrics):
    return {
        "holdout": list(holdout),
        "holdout_label": "_".join(map(str, holdout)) if holdout else "none",
        "per_time": {str(t): float(v) for t, v in fold_metrics.items()},
        "fold_avg": float(np.mean(list(fold_metrics.values()))) if fold_metrics else None,
    }


def run_experiment(method, exp_name, seed=42, device_str="cuda", save_dir=None, sigma=0.1):
    cfg = EXPERIMENT_CONFIGS[exp_name]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"=== {method.upper()} {exp_name} seed={seed} on {device} ===")

    if save_dir is None:
        save_dir = Path(str(DEFAULT_OUTPUT_DIR_TPL).format(method=method))
    save_dir = Path(save_dir)
    out_dir = save_dir / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"seed{seed}.json"

    dim = cfg["pca_dim"]
    eval_all_times = exp_name in ("eb100_loo", "eb100_l2o")
    all_fold_metrics = []
    all_train_times_sec = []

    t_run_start = time.perf_counter()
    for fold_idx, holdout in enumerate(cfg["folds"]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        use_ot = method == "otcfm"
        data = load_data(cfg["dataset"], cfg["pca_dim"], holdout_times=holdout, ot_coupling=use_ot)
        all_times = data["all_times"]
        t_min, t_max = min(all_times), max(all_times)
        train_times = data["train_times"]
        tp_norm = np.array([(t - t_min) / (t_max - t_min) for t in train_times], dtype=np.float32)

        fold_label = (
            f"{method}_{exp_name}_seed{seed}_holdout"
            f"{'_'.join(map(str, holdout)) if holdout else 'none'}"
        )
        logger.info(
            f"\n--- Fold {fold_idx+1}/{len(cfg['folds'])}: holdout={holdout}, "
            f"train={train_times}, tp_norm={tp_norm.tolist()} ---"
        )
        if use_ot:
            logger.info("    Using precomputed OT couplings")

        X_by_train_time = [data["marginals"][t] for t in train_times]

        flow_model = VelocityNet(
            dim=dim,
            hidden_dim=cfg["hidden_dim"],
            num_hidden_layers=cfg["num_hidden_layers"],
            dropout=cfg["dropout"],
            residual_every=cfg["residual_every"],
        ).to(device)
        score_model = None
        if method == "sf2m":
            score_model = VelocityNet(
                dim=dim,
                hidden_dim=cfg["hidden_dim"],
                num_hidden_layers=cfg["num_hidden_layers"],
                dropout=cfg["dropout"],
                residual_every=cfg["residual_every"],
            ).to(device)
        n_params = sum(p.numel() for p in flow_model.parameters())
        if score_model is not None:
            n_params += sum(p.numel() for p in score_model.parameters())
        logger.info(f"    Model params: {n_params:,}")

        fold_epochs = cfg["fold_epochs"][fold_idx]
        train_sec, losses = train_model(
            method,
            flow_model,
            score_model,
            X_by_train_time,
            tp_norm,
            epochs=fold_epochs,
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            iters_per_epoch=cfg["iters_per_epoch"],
            sigma=sigma,
            device=device,
            ot_alignments=data["ot_alignments"] if use_ot else None,
            train_times=train_times if use_ot else None,
        )
        all_train_times_sec.append(train_sec)
        logger.info(f"    Training time: {train_sec:.1f}s  Final loss: {losses[-1]:.6f}")
        plot_loss_curve(losses, fold_label, out_dir)

        fold_results, transported = evaluate_fold(
            flow_model,
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

        # Save checkpoint + fine trajectory for downstream PCA plots (normalized space)
        source_init = data["marginals"][all_times[0]]

        def _ode_fn(t, x, _model=flow_model):
            t_vec = t.expand(x.shape[0], 1)
            return _model(torch.cat([x, t_vec], dim=1))

        sd_pieces = {"flow_model": flow_model.state_dict()}
        if score_model is not None:
            sd_pieces["score_model"] = score_model.state_dict()
        save_trajectory_and_checkpoint_torchdiffeq(
            model=flow_model,
            source_np=source_init,
            ode_fn=_ode_fn,
            out_dir=out_dir,
            fold_idx=fold_idx,
            holdout=holdout,
            seed=seed,
            method=method,
            dataset=cfg["dataset"],
            dim=cfg["pca_dim"],
            marginal_times=all_times,
            config={k: v for k, v in cfg.items() if k != "folds"},
            state_dict_pieces=sd_pieces,
            extras={
                "train_sec": train_sec,
                "fold_idx": fold_idx,
                "holdout": list(holdout),
                "seed": seed,
            },
            t_start=0.0,
            t_end=1.0,
            n_steps=101,
            device=str(device),
        )

    folds_serialized = [
        _serialize_fold(holdout, fold_res)
        for holdout, fold_res in zip(cfg["folds"], all_fold_metrics)
    ]
    if exp_name in ("eb100_loo", "eb100_l2o"):
        for holdout, fold_res in zip(cfg["folds"], all_fold_metrics):
            fold_avg = float(np.mean(list(fold_res.values())))
            logger.info(f"  holdout={holdout}: per-time {dict(fold_res)} -> avg = {fold_avg:.6f}")
        grand_avg = float(np.mean([np.mean(list(f.values())) for f in all_fold_metrics]))
    else:
        all_values = [v for fold in all_fold_metrics for v in fold.values()]
        grand_avg = float(np.mean(all_values))
    logger.info(f"  Grand avg {cfg['primary_metric'].upper()}: {grand_avg:.6f}")

    total_sec = time.perf_counter() - t_run_start
    payload = {
        "experiment": exp_name,
        "method": method,
        "seed": seed,
        "primary_metric": cfg["primary_metric"],
        "primary_metric_space": cfg["primary_metric_space"],
        "pca_dim": cfg["pca_dim"],
        "folds": folds_serialized,
        "grand_avg": grand_avg,
        "train_sec_per_fold": [float(s) for s in all_train_times_sec],
        "total_sec": float(total_sec),
        "config": {k: v for k, v in cfg.items() if k != "folds"},
        "sigma": sigma,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"  Saved metrics to {metrics_path}")
    return all_fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Unified torchcfm baseline eval")
    parser.add_argument("--method", required=True, choices=["icfm", "otcfm", "sf2m"])
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="ICFM/OT-CFM use sigma for variance of probability path; "
        "SF2M uses sigma>0 for the SB diffusion.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Output dir; defaults to results/<method>/"
    )
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = str(BASE_DIR / "results" / args.method)
    run_experiment(
        args.method,
        args.experiment,
        seed=args.seed,
        device_str=args.device,
        save_dir=Path(args.save_dir),
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
