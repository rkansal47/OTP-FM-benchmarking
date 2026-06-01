"""
TrajectoryNet evaluation harness.

Trains TrajectoryNet (CNF) using its native training loop and evaluates by
forward-integrating the learned ODE from t=0 to held-out timepoints.

Uses OTP-FM data loaders for consistent evaluation.

Usage:
    conda run -n env_trajectorynet python train_trajectorynet_eval.py --experiment eb100_loo --seed 42
"""

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OTP_FM_DIR = BASE_DIR / "OTP-FM"
TNET_DIR = Path(__file__).resolve().parents[1] / "baselines" / "TrajectoryNet"

sys.path.insert(0, str(TNET_DIR))
from TrajectoryNet.train_misc import (
    build_model_tabular,
    set_cnf_options,
    create_regularization_fns,
)


def _import_from(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eb_data = _import_from("eb_data", OTP_FM_DIR / "experiments" / "singlecell" / "data.py")
_cite_data = _import_from("cite_data", OTP_FM_DIR / "experiments" / "citeseq" / "data.py")


EXPERIMENT_CONFIGS = {
    "eb100_loo": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1], [2], [3], []],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "niters": 5000,
        "hidden_dims": "128-128-128",
    },
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1, 3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "niters": 5000,
        "hidden_dims": "128-128-128",
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "folds": [[1], [2], [3]],
        "primary_metric": "w1",
        "primary_metric_space": "normalized",
        "niters": 1000,
        "hidden_dims": "64-64-64",
    },
}


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_w1(gen, gt):
    import ot as pot_lib

    n = min(len(gen), len(gt))
    g, r = gen[:n].astype(np.float64), gt[:n].astype(np.float64)
    M = pot_lib.dist(g, r, metric="euclidean")
    a, b = np.ones(n) / n, np.ones(n) / n
    return float(pot_lib.emd2(a, b, M, numItermax=int(1e7)))


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


METRIC_FNS = {"w1": compute_w1, "mmd": compute_mmd}


# ── Data ─────────────────────────────────────────────────────────────────────


def load_data(dataset, pca_dim, holdout_times):
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


class SCDataWrapper:
    """Wraps OTP-FM data in the format TrajectoryNet expects."""

    def __init__(self, marginals, train_times, pca_dim):
        all_data = []
        all_labels = []
        for t in train_times:
            arr = marginals[t]
            all_data.append(arr)
            all_labels.extend([float(t)] * len(arr))
        self.data = np.vstack(all_data).astype(np.float32)
        self.labels = np.array(all_labels, dtype=np.float32)
        self.unique_times = sorted(set(all_labels))
        self.dim = pca_dim

    def get_data(self):
        return self.data

    def get_times(self):
        return self.labels

    def get_unique_times(self):
        return self.unique_times

    def sample_index(self, batch_size, tp):
        mask = self.labels == tp
        indices = np.where(mask)[0]
        return np.random.choice(indices, size=min(batch_size, len(indices)), replace=True)

    def get_labels(self):
        return self.labels


def make_args(cfg, data_wrapper, holdout, device):
    """Build a namespace mimicking TrajectoryNet's argparse output."""
    times = data_wrapper.get_unique_times()
    time_scale = times[-1] - times[0]

    args = SimpleNamespace(
        data=data_wrapper,
        dims=cfg["hidden_dims"],
        num_blocks=1,
        layer_type="concatsquash",
        nonlinearity="tanh",
        time_length=1.0,
        train_T=True,
        divergence_fn="approximate",
        solver="dopri5",
        atol=1e-5,
        rtol=1e-5,
        step_size=None,
        test_solver=None,
        test_atol=None,
        test_rtol=None,
        residual=False,
        rademacher=True,
        batch_norm=False,
        bn_lag=0,
        niters=cfg["niters"],
        batch_size=256,
        lr=1e-3,
        weight_decay=0.0,
        spectral_norm=False,
        l1int=0.0,
        l2int=0.0,
        sl2int=0.0,
        dl2int=0.0,
        dtl2int=0.0,
        JFrobint=0.0,
        JdiagFrobint=0.0,
        JoffdiagFrobint=0.0,
        time_scale=time_scale,
        timepoints=times,
        int_tps=[float(t) for t in times],
        leaveout_timepoint=-1,
        use_growth=False,
        training_noise=0.01,
        save=str(BASE_DIR / "logs" / "baselines_eval" / "tnet_tmp"),
        save_freq=cfg["niters"] + 1,
        viz_freq=cfg["niters"] + 1,
        val_freq=cfg["niters"] + 1,
        log_freq=500,
        device=device,
    )
    return args


# ── Training and transport ───────────────────────────────────────────────────


def train_trajectorynet(args, device, pca_dim):
    """Train TrajectoryNet and return the model."""
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, pca_dim, regularization_fns).to(device)
    set_cnf_options(args, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    full_data = torch.from_numpy(args.data.get_data()).float().to(device)

    for itr in range(1, args.niters + 1):
        model.train()
        optimizer.zero_grad()
        loss = _compute_loss(device, args, model, full_data)
        loss.backward()
        optimizer.step()
        if itr % 500 == 0:
            logger.info(f"      Iter {itr}/{args.niters} loss={loss.item():.6f}")

    return model


def _compute_loss(device, args, model, full_data):
    """Simplified loss: integrate each timepoint pair independently."""
    times = args.timepoints
    total_loss = 0.0

    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        integration_times = torch.tensor([t0, t1]).float().to(device)
        idx = args.data.sample_index(args.batch_size, t1)
        x = torch.from_numpy(args.data.get_data()[idx]).float().to(device)
        zero = torch.zeros(x.shape[0], 1, device=device)
        _, delta_logp = model(x, zero, integration_times=integration_times, reverse=False)
        total_loss += delta_logp.mean()

    return total_loss


@torch.no_grad()
def transport_trajectorynet(model, source, t_start, t_end, device, time_scale):
    """Forward-integrate from t_start to t_end."""
    model.eval()
    x0 = torch.from_numpy(source).float().to(device)
    integration_times = torch.tensor([t_start, t_end]).float().to(device)
    z = model(x0, integration_times=integration_times, reverse=True)
    return z.cpu().numpy()


# ── Main ─────────────────────────────────────────────────────────────────────


def run_experiment(exp_name, seed=42, device_str="cuda", save_dir=None):
    cfg = EXPERIMENT_CONFIGS[exp_name]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"=== TrajectoryNet {exp_name} seed={seed} on {device} ===")

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "trajectorynet"
    save_dir = Path(save_dir)
    out_dir = save_dir / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"seed{seed}.json"

    eval_all_times = exp_name in ("eb100_loo", "eb100_l2o")
    all_fold_metrics = []
    all_train_times_sec = []

    t_run_start = time.perf_counter()
    for fold_idx, holdout in enumerate(cfg["folds"]):
        torch.manual_seed(seed)
        np.random.seed(seed)

        data = load_data(cfg["dataset"], cfg["pca_dim"], holdout_times=holdout)
        all_times = data["all_times"]
        train_times = data["train_times"]

        logger.info(
            f"\n--- Fold {fold_idx+1}/{len(cfg['folds'])}: holdout={holdout}, train={train_times} ---"
        )

        data_wrapper = SCDataWrapper(data["marginals"], train_times, cfg["pca_dim"])
        args = make_args(cfg, data_wrapper, holdout, device)

        t_start = time.perf_counter()
        model = train_trajectorynet(args, device, cfg["pca_dim"])
        train_sec = time.perf_counter() - t_start
        all_train_times_sec.append(train_sec)
        logger.info(f"    Training time: {train_sec:.1f}s")

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
        time_scale = args.time_scale

        for t_eval in all_times:
            if t_eval == all_times[0]:
                continue
            if t_eval not in metric_times:
                continue
            gen = transport_trajectorynet(
                model, source, float(all_times[0]), float(t_eval), device, time_scale
            )
            gt = data["marginals"][t_eval]
            gen_metric, gt_metric = gen, gt
            if cfg["primary_metric_space"] == "original" and data["scaler"] is not None:
                gen_metric = data["scaler"].inverse_transform(gen)
                gt_metric = data["scaler"].inverse_transform(gt)
            val = metric_fn(gen_metric, gt_metric)
            fold_results[t_eval] = val
            logger.info(f"    t={t_eval}: {cfg['primary_metric']}={val:.6f}")

        all_fold_metrics.append(fold_results)

    folds_serialized = [
        {
            "holdout": list(h),
            "per_time": {str(t): float(v) for t, v in fr.items()},
            "fold_avg": float(np.mean(list(fr.values()))) if fr else None,
        }
        for h, fr in zip(cfg["folds"], all_fold_metrics)
    ]
    all_values = [v for fold in all_fold_metrics for v in fold.values()]
    grand_avg = float(np.mean(all_values)) if all_values else 0.0
    logger.info(f"  Grand avg {cfg['primary_metric'].upper()}: {grand_avg:.6f}")

    total_sec = time.perf_counter() - t_run_start
    payload = {
        "experiment": exp_name,
        "method": "trajectorynet",
        "seed": seed,
        "primary_metric": cfg["primary_metric"],
        "primary_metric_space": cfg["primary_metric_space"],
        "pca_dim": cfg["pca_dim"],
        "folds": folds_serialized,
        "grand_avg": grand_avg,
        "train_sec_per_fold": [float(s) for s in all_train_times_sec],
        "total_sec": float(total_sec),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    logger.info(f"  Saved metrics to {metrics_path}")
    return all_fold_metrics


def main():
    ap = argparse.ArgumentParser(description="TrajectoryNet evaluation harness")
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save-dir", type=str, default=None)
    a = ap.parse_args()
    save_dir = Path(a.save_dir) if a.save_dir else BASE_DIR / "results" / "trajectorynet"
    run_experiment(a.experiment, seed=a.seed, device_str=a.device, save_dir=save_dir)


if __name__ == "__main__":
    main()
