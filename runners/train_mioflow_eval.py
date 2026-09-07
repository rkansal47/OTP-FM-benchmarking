"""
MIOFlow evaluation harness.

Trains MIOFlow and evaluates transported samples
against ground truth at holdout/all times using W1/W2/MMD.

Uses the MIOFlow library API with OTP-FM-compatible data loading.
For the PCA-space models used here, GAGA maps from PCA_dim -> 2D PHATE latent,
and the ODE operates in that latent space. For eval, we transport in the
ODE's normalized space and then denormalize.

We use a "direct" mode that skips GAGA and trains the ODE directly in PCA space.

Usage:
    conda run -n env_mioflow python train_mioflow_eval.py --experiment eb5_l2o --seed 42
    conda run -n env_mioflow python train_mioflow_eval.py --experiment eb100_loo --seed 42 --mode direct
"""

import argparse
import importlib.util
import json
import logging
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import ot
import torch
from torchdiffeq import odeint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OTP_FM_DIR = BASE_DIR / "OTP-FM"
MIOFLOW_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "MIOFlow"
sys.path.insert(0, str(MIOFLOW_ROOT))

from mioflow.core.models.ode_model import ODEFunc
from mioflow.mioflow import train_mioflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runtime_patches import patch_mioflow_odeint

patch_mioflow_odeint()  # force RK4 step_size=0.1 (paper-run integrator config)

from _traj_utils import save_torch_checkpoint, save_trajectory_npz
from mioflow.core.datasets import TimeSeriesDataset


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
        "primary_metric": "w1",
        "primary_metric_space": "normalized",
        "hidden_dim": 64,
        "n_epochs": 300,
        "gaga_encoder_epochs": 300,
        "gaga_decoder_epochs": 300,
        "sample_size": 100,
        "lr": 1e-3,
    },
    "eb5_l2o": {
        "dataset": "eb",
        "pca_dim": 5,
        "folds": [[1, 3]],
        "primary_metric": "w2",
        "primary_metric_space": "normalized",
        "hidden_dim": 64,
        "n_epochs": 300,
        "gaga_encoder_epochs": 300,
        "gaga_decoder_epochs": 300,
        "sample_size": 100,
        "lr": 1e-3,
    },
    "eb100_loo": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1], [2], [3], []],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "hidden_dim": 128,
        "n_epochs": 500,
        "gaga_encoder_epochs": 300,
        "gaga_decoder_epochs": 300,
        "sample_size": 100,
        "lr": 1e-3,
    },
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1, 3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "hidden_dim": 128,
        "n_epochs": 500,
        "gaga_encoder_epochs": 300,
        "gaga_decoder_epochs": 300,
        "sample_size": 100,
        "lr": 1e-3,
    },
    "cite50_loo": {
        "dataset": "cite",
        "pca_dim": 50,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "hidden_dim": 128,
        "n_epochs": 500,
        "gaga_encoder_epochs": 300,
        "gaga_decoder_epochs": 300,
        "sample_size": 100,
        "lr": 1e-3,
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "normalized",
        "hidden_dim": 64,
        "n_epochs": 300,
        "gaga_encoder_epochs": 300,
        "gaga_decoder_epochs": 300,
        "sample_size": 100,
        "lr": 1e-3,
    },
}


# ── Metrics ──────────────────────────────────────────────────────────────────


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


# ── Data loading ─────────────────────────────────────────────────────────────


def load_data(dataset, pca_dim, holdout_times):
    """Load via OTP-FM loaders."""
    data_dir = OTP_FM_DIR / "data"
    if dataset == "eb":
        raw = _eb_data.load_eb_data(
            data_dir=data_dir,
            pca_dim=pca_dim,
            normalize=True,
            ot_coupling=False,
            holdout_times=holdout_times,
        )
    elif dataset == "cite":
        raw = _cite_data.load_citeseq_data(
            data_dir=OTP_FM_DIR / "data",
            pca_dim=pca_dim,
            normalize=True,
            ot_coupling=False,
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
        "train_times": raw["train_times"],
    }


# ── Direct-mode training (skip GAGA, train ODE in PCA space) ────────────────


def train_direct(data, train_times, cfg, device, seed):
    """Train a MIOFlow-style ODE directly in the normalized PCA space."""
    dim = cfg["pca_dim"]
    marginals_np = data["marginals"]

    # Build TimeSeriesDataset from train data
    # MIOFlow normalizes internally, so we pass raw normalized data and
    # handle mean/std ourselves for eval.
    all_train_data = np.concatenate([marginals_np[t] for t in train_times])
    mean_vals = all_train_data.mean(axis=0)
    std_vals = all_train_data.std(axis=0)
    std_vals = np.where(std_vals == 0, 1.0, std_vals)

    time_series = []
    for idx, t in enumerate(train_times):
        arr = (marginals_np[t] - mean_vals) / std_vals
        time_series.append((arr.astype(np.float32), float(idx)))
    dataset = TimeSeriesDataset(time_series)

    model = ODEFunc(input_dim=dim, hidden_dim=cfg["hidden_dim"])
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"    ODEFunc params: {n_params:,}")

    t_start = time.perf_counter()
    train_mioflow(
        model=model,
        dataset=dataset,
        num_epochs=cfg["n_epochs"],
        batch_size=cfg["sample_size"],
        learning_rate=cfg["lr"],
        device=device,
        lambda_ot=1.0,
        lambda_density=0.0,
        lambda_energy=0.01,
        energy_time_steps=10,
        sde_dt=0.1,
        lambda_energy_f=1.0,
        lambda_energy_g=0.0,
        grad_clip=1.0,
        scheduler_type="cosine",
        scheduler_step_size=30,
        scheduler_gamma=0.5,
        scheduler_t_max=cfg["n_epochs"],
        scheduler_min_lr=1e-5,
        growth_rate_model=None,
        growth_rate_lr=1e-4,
    )
    train_sec = time.perf_counter() - t_start
    logger.info(f"    Training time: {train_sec:.1f}s")

    return model, mean_vals, std_vals, train_sec


@torch.no_grad()
def transport_direct(model, source_np, mean_vals, std_vals, t_start, t_end, device):
    """Integrate ODE from t_start to t_end in normalized space, denormalize."""
    model.eval()
    x0_norm = (source_np - mean_vals) / std_vals
    x0 = torch.from_numpy(x0_norm.astype(np.float32)).to(device)
    t_span = torch.tensor([float(t_start), float(t_end)], device=device)
    traj = odeint(model, x0, t_span, method="rk4", options=dict(step_size=0.1))
    gen_norm = traj[-1].cpu().numpy()
    return gen_norm * std_vals + mean_vals


# ── Main ─────────────────────────────────────────────────────────────────────


def _serialize_fold(holdout, fold_metrics):
    return {
        "holdout": list(holdout),
        "holdout_label": "_".join(map(str, holdout)) if holdout else "none",
        "per_time": {str(t): float(v) for t, v in fold_metrics.items()},
        "fold_avg": float(np.mean(list(fold_metrics.values()))) if fold_metrics else None,
    }


def run_experiment(exp_name, seed=42, device_str="cuda", save_dir=None, mode="direct"):
    cfg = EXPERIMENT_CONFIGS[exp_name]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"=== MIOFlow ({mode}) {exp_name} seed={seed} on {device} ===")

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "mioflow"
    save_dir = Path(save_dir)
    out_dir = save_dir / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"seed{seed}.json"

    eval_all_times = exp_name in ("eb100_loo", "eb100_l2o")
    all_fold_metrics = []
    all_train_times_sec = []

    t_run_start = time.perf_counter()
    for fold_idx, holdout in enumerate(cfg["folds"]):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data = load_data(cfg["dataset"], cfg["pca_dim"], holdout_times=holdout)
        all_times = data["all_times"]
        train_times = data["train_times"]
        n_train = len(train_times)

        logger.info(
            f"\n--- Fold {fold_idx+1}/{len(cfg['folds'])}: holdout={holdout}, "
            f"train={train_times} ---"
        )

        model, mean_vals, std_vals, train_sec = train_direct(data, train_times, cfg, device, seed)
        all_train_times_sec.append(train_sec)

        # Evaluate
        if eval_all_times:
            metric_times = [t for t in all_times if t != all_times[0]]
        elif holdout:
            metric_times = holdout
        else:
            metric_times = [t for t in all_times if t != all_times[0]]

        source = data["marginals"][all_times[0]]
        metric_fn = METRIC_FNS[cfg["primary_metric"]]
        fold_results = {}

        t_real_min = min(all_times)
        t_real_max = max(all_times)

        for t_eval in all_times:
            if t_eval == all_times[0]:
                continue
            # Map real time to ODE index space [0, n_train-1]
            frac = (t_eval - t_real_min) / (t_real_max - t_real_min)
            ode_t = frac * (n_train - 1)
            gen = transport_direct(model, source, mean_vals, std_vals, 0.0, ode_t, device)
            if t_eval in metric_times:
                gt = data["marginals"][t_eval]
                gen_metric, gt_metric = gen, gt
                if cfg["primary_metric_space"] == "original" and data["scaler"] is not None:
                    gen_metric = data["scaler"].inverse_transform(gen)
                    gt_metric = data["scaler"].inverse_transform(gt)
                val = metric_fn(gen_metric, gt_metric)
                fold_results[t_eval] = val
                logger.info(
                    f"    t={t_eval} (ode_t={ode_t:.2f}): " f"{cfg['primary_metric']}={val:.6f}"
                )

        all_fold_metrics.append(fold_results)

        # Save checkpoint + fine trajectory (denormalized -> normalized PCA space)
        model.eval()
        n_traj = min(2000, source.shape[0])
        x0_norm = (source[:n_traj] - mean_vals) / std_vals
        x0 = torch.from_numpy(x0_norm.astype(np.float32)).to(device)
        ode_t_max = float(n_train - 1)
        t_span_ode = torch.linspace(0.0, ode_t_max, 101, device=device)
        with torch.no_grad():
            traj = odeint(model, x0, t_span_ode, method="rk4", options=dict(step_size=0.1))
        traj_np = traj.cpu().numpy()
        traj_np = traj_np * std_vals + mean_vals
        traj_for_save = traj_np.transpose(1, 0, 2).astype(np.float32)
        t_eval_norm = (t_span_ode.cpu().numpy() / ode_t_max).astype(np.float32)

        holdout_str = "_".join(map(str, holdout)) if holdout else "none"
        base = f"fold{fold_idx}_holdout{holdout_str}_seed{seed}"
        save_trajectory_npz(
            trajectories=traj_for_save,
            t_eval=t_eval_norm,
            marginal_times=all_times,
            method="mioflow",
            dataset=cfg["dataset"],
            dim=cfg["pca_dim"],
            output_path=out_dir / "trajectories" / f"{base}_trajectories.npz",
            config={k: v for k, v in cfg.items() if k != "folds"},
        )
        save_torch_checkpoint(
            {
                "ode_func": model.state_dict(),
                "mean_vals": torch.tensor(mean_vals, dtype=torch.float32),
                "std_vals": torch.tensor(std_vals, dtype=torch.float32),
            },
            out_dir / "checkpoints" / f"{base}.pt",
            train_sec=float(train_sec),
            fold_idx=fold_idx,
            holdout=list(holdout),
            seed=seed,
            n_train=n_train,
        )

    folds_serialized = [
        _serialize_fold(holdout, fold_res)
        for holdout, fold_res in zip(cfg["folds"], all_fold_metrics)
    ]
    if exp_name in ("eb100_loo", "eb100_l2o"):
        grand_avg = float(np.mean([np.mean(list(f.values())) for f in all_fold_metrics]))
    else:
        all_values = [v for fold in all_fold_metrics for v in fold.values()]
        grand_avg = float(np.mean(all_values))
    logger.info(f"  Grand avg {cfg['primary_metric'].upper()}: {grand_avg:.6f}")

    total_sec = time.perf_counter() - t_run_start
    payload = {
        "experiment": exp_name,
        "method": "mioflow",
        "mode": mode,
        "seed": seed,
        "primary_metric": cfg["primary_metric"],
        "primary_metric_space": cfg["primary_metric_space"],
        "pca_dim": cfg["pca_dim"],
        "folds": folds_serialized,
        "grand_avg": grand_avg,
        "train_sec_per_fold": [float(s) for s in all_train_times_sec],
        "total_sec": float(total_sec),
        "config": {k: v for k, v in cfg.items() if k not in ("folds",)},
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"  Saved metrics to {metrics_path}")
    return all_fold_metrics


def main():
    parser = argparse.ArgumentParser(description="MIOFlow evaluation harness")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "gaga"])
    args = parser.parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else BASE_DIR / "results" / "mioflow"
    run_experiment(
        args.experiment, seed=args.seed, device_str=args.device, save_dir=save_dir, mode=args.mode
    )


if __name__ == "__main__":
    main()
