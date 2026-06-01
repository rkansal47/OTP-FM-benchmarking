"""
DeepRUOT evaluation harness.

Trains DeepRUOT (velocity + growth + score networks) using the train_un1
function, then evaluates by integrating the velocity field from t=0.

Uses OTP-FM data loaders for consistent evaluation. DeepRUOT expects data
as a pandas DataFrame with 'samples' column (time labels) and feature columns.

Usage:
    conda run -n env_deepruot python train_deepruot_eval.py --experiment eb5_l2o --seed 42
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
import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OTP_FM_DIR = BASE_DIR / "OTP-FM"
DEEPRUOT_DIR = Path(__file__).resolve().parents[1] / "baselines" / "DeepRUOT"

sys.path.insert(0, str(DEEPRUOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runtime_patches import shim_tqdm_notebook  # noqa: E402

shim_tqdm_notebook()  # DeepRUOT/train.py uses `from tqdm.notebook import tqdm`
from DeepRUOT.models import FNet, ODEFunc  # noqa: E402
from DeepRUOT.train import train_un1  # noqa: E402
from DeepRUOT.losses import OT_loss1  # noqa: E402

from _traj_utils import save_trajectory_and_checkpoint_torchdiffeq  # noqa: E402


def _import_from(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eb_data = _import_from("eb_data", OTP_FM_DIR / "experiments" / "singlecell" / "data.py")
_cite_data = _import_from("cite_data", OTP_FM_DIR / "experiments" / "citeseq" / "data.py")


EXPERIMENT_CONFIGS = {
    "eb5_l2o": {
        "dataset": "eb",
        "pca_dim": 5,
        "folds": [[1, 3]],
        "primary_metric": "w2",
        "primary_metric_space": "normalized",
        "hidden_dim": 64,
        "n_hiddens": 3,
        "n_batches": 10,
        "n_epochs": 50,
    },
    "eb100_loo": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1], [2], [3], []],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "hidden_dim": 256,
        "n_hiddens": 3,
        "n_batches": 20,
        "n_epochs": 100,
    },
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1, 3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "hidden_dim": 256,
        "n_hiddens": 3,
        "n_batches": 20,
        "n_epochs": 100,
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "hidden_dim": 64,
        "n_hiddens": 3,
        "n_batches": 10,
        "n_epochs": 50,
    },
    "cite50_loo": {
        "dataset": "cite",
        "pca_dim": 50,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "hidden_dim": 256,
        "n_hiddens": 3,
        "n_batches": 20,
        "n_epochs": 100,
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


def compute_w2(gen, gt, max_dim=10):
    import ot as pot_lib

    d = min(gen.shape[1], max_dim)
    g, r = gen[:, :d], gt[:, :d]
    n = min(len(g), len(r))
    g, r = g[:n].astype(np.float64), r[:n].astype(np.float64)
    M = pot_lib.dist(g, r, metric="sqeuclidean")
    a, b = np.ones(n) / n, np.ones(n) / n
    return float(np.sqrt(pot_lib.emd2(a, b, M, numItermax=int(1e7))))


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


def make_df(marginals, train_times, pca_dim):
    """Build a pandas DataFrame in DeepRUOT format from OTP-FM marginals.
    Normalizes time to [0, 1] range for efficient Euler integration."""
    t_min, t_max = float(min(train_times)), float(max(train_times))
    rows = []
    for t in train_times:
        t_norm = (float(t) - t_min) / (t_max - t_min) if t_max > t_min else 0.0
        arr = marginals[t]
        for row in arr:
            rows.append([t_norm] + list(row))
    cols = ["samples"] + [f"x{i}" for i in range(pca_dim)]
    return pd.DataFrame(rows, columns=cols)


# ── Training ─────────────────────────────────────────────────────────────────


def train_deepruot(df, cfg, device, seed):
    """Train DeepRUOT model on the given DataFrame."""
    dim = cfg["pca_dim"]
    model = FNet(dim, cfg["hidden_dim"], cfg["n_hiddens"], activation="Tanh").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    groups = sorted(df["samples"].unique())
    use_cuda = device.type == "cuda"

    n_initial = len(df[df["samples"] == groups[0]])
    relative_mass = [1.0] * (len(groups) + 1)
    for epoch in range(cfg["n_epochs"]):
        train_un1(
            model,
            df,
            groups,
            optimizer,
            n_batches=cfg["n_batches"],
            criterion=OT_loss1(),
            use_cuda=use_cuda,
            sample_size=(256,),
            sample_with_replacement=True,
            local_loss=True,
            global_loss=False,
            hold_one_out=False,
            use_penalty=True,
            lambda_energy=0.01,
            lambda_mass=1.0,
            initial_size=n_initial,
            relative_mass=relative_mass,
            device=device,
            best_model_path=str(BASE_DIR / "logs" / "baselines_eval" / "deepruot_best.pt"),
        )
        if (epoch + 1) % 100 == 0:
            logger.info(f"      Epoch {epoch+1}/{cfg['n_epochs']}")

    return model


# ── Transport ────────────────────────────────────────────────────────────────


@torch.no_grad()
def transport_deepruot(model, source, t_start, t_end, device):
    """Integrate DeepRUOT velocity field from t_start to t_end."""
    model.eval()
    x0 = torch.from_numpy(source).float().to(device)
    time = torch.tensor([t_start, t_end], device=device)
    traj = odeint(ODEFunc(model.v_net), x0, time, method="dopri5", atol=1e-5, rtol=1e-5)
    return traj[-1].cpu().numpy()


# ── Main ─────────────────────────────────────────────────────────────────────


def _serialize_fold(holdout, fold_metrics):
    return {
        "holdout": list(holdout),
        "holdout_label": "_".join(map(str, holdout)) if holdout else "none",
        "per_time": {str(t): float(v) for t, v in fold_metrics.items()},
        "fold_avg": float(np.mean(list(fold_metrics.values()))) if fold_metrics else None,
    }


def run_experiment(exp_name, seed=42, device_str="cuda", save_dir=None):
    cfg = EXPERIMENT_CONFIGS[exp_name]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"=== DeepRUOT {exp_name} seed={seed} on {device} ===")

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "deepruot"
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
        t_min, t_max = float(min(all_times)), float(max(all_times))

        logger.info(
            f"\n--- Fold {fold_idx+1}/{len(cfg['folds'])}: holdout={holdout}, train={train_times} ---"
        )

        df = make_df(data["marginals"], train_times, cfg["pca_dim"])
        logger.info(f"    DataFrame shape: {df.shape}, groups: {sorted(df['samples'].unique())}")

        t_start = time.perf_counter()
        model = train_deepruot(df, cfg, device, seed)
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

        for t_eval in all_times:
            if t_eval == all_times[0]:
                continue
            if t_eval not in metric_times:
                continue
            t_norm_end = (float(t_eval) - t_min) / (t_max - t_min) if t_max > t_min else 1.0
            gen = transport_deepruot(model, source, 0.0, t_norm_end, device)
            gt = data["marginals"][t_eval]
            gen_metric, gt_metric = gen, gt
            if cfg["primary_metric_space"] == "original" and data["scaler"] is not None:
                gen_metric = data["scaler"].inverse_transform(gen)
                gt_metric = data["scaler"].inverse_transform(gt)
            val = metric_fn(gen_metric, gt_metric)
            fold_results[t_eval] = val
            logger.info(f"    t={t_eval}: {cfg['primary_metric']}={val:.6f}")

        all_fold_metrics.append(fold_results)

        # Save checkpoint + fine trajectory (normalized PCA space)
        v_net = model.v_net
        ode_func = ODEFunc(v_net)
        save_trajectory_and_checkpoint_torchdiffeq(
            model=model,
            source_np=source,
            ode_fn=ode_func,
            out_dir=out_dir,
            fold_idx=fold_idx,
            holdout=holdout,
            seed=seed,
            method="deepruot",
            dataset=cfg["dataset"],
            dim=cfg["pca_dim"],
            marginal_times=all_times,
            config={k: v for k, v in cfg.items() if k != "folds"},
            state_dict_pieces={"model": model.state_dict()},
            extras={
                "train_sec": float(train_sec),
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
    all_values = [v for fold in all_fold_metrics for v in fold.values()]
    grand_avg = float(np.mean(all_values)) if all_values else 0.0
    logger.info(f"  Grand avg {cfg['primary_metric'].upper()}: {grand_avg:.6f}")

    total_sec = time.perf_counter() - t_run_start
    payload = {
        "experiment": exp_name,
        "method": "deepruot",
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
    parser = argparse.ArgumentParser(description="DeepRUOT evaluation harness")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else BASE_DIR / "results" / "deepruot"
    run_experiment(args.experiment, seed=args.seed, device_str=args.device, save_dir=save_dir)


if __name__ == "__main__":
    main()
