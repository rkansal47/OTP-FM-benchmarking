"""
VGFM (Velocity-Growth Flow Matching) evaluation harness.

Trains VGFM using its native pretrain phase, then integrates the ODE from t0
to each evaluation time. Evaluates with W1/W2/MMD following OTP-FM protocol.

Uses baselines/VGFM/ code (FNet, pretrain) with OTP-FM data loaders for
preprocessing and holdout handling.

Usage:
    conda run -n env_vgfm python train_vgfm_eval.py --experiment eb5_l2o --seed 42
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
import pandas as pd
import torch
from torchdiffeq import odeint_adjoint as odeint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OTP_FM_DIR = BASE_DIR / "OTP-FM"
VGFM_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "VGFM"
sys.path.insert(0, str(VGFM_ROOT))

from VGFM.models import FNet, ODEFunc2  # noqa: E402
from VGFM.train import pretrain  # noqa: E402


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
        "hidden_dim": 128,
        "n_hiddens": 3,
        "activation": "leakyrelu",
        "n_pretrain_epochs": 2000,
        "batch_size": 256,
        "lr": 1e-3,
        "reg": 0.01,
        "reg_m": 5,
        "norm_cost": True,
    },
    "eb5_l2o": {
        "dataset": "eb",
        "pca_dim": 5,
        "folds": [[1, 3]],
        "primary_metric": "w2",
        "primary_metric_space": "normalized",
        "hidden_dim": 128,
        "n_hiddens": 3,
        "activation": "leakyrelu",
        "n_pretrain_epochs": 2000,
        "batch_size": 256,
        "lr": 1e-3,
        "reg": 0.01,
        "reg_m": 5,
        "norm_cost": True,
    },
    "eb100_loo": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1], [2], [3], []],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "hidden_dim": 256,
        "n_hiddens": 5,
        "activation": "leakyrelu",
        "n_pretrain_epochs": 2000,
        "batch_size": 256,
        "lr": 1e-3,
        "reg": 0.01,
        "reg_m": 5,
        "norm_cost": True,
    },
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1, 3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "hidden_dim": 256,
        "n_hiddens": 5,
        "activation": "leakyrelu",
        "n_pretrain_epochs": 2000,
        "batch_size": 256,
        "lr": 1e-3,
        "reg": 0.01,
        "reg_m": 5,
        "norm_cost": True,
    },
    "cite50_loo": {
        "dataset": "cite",
        "pca_dim": 50,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "hidden_dim": 256,
        "n_hiddens": 5,
        "activation": "leakyrelu",
        "n_pretrain_epochs": 5000,
        "batch_size": 256,
        "lr": 1e-3,
        "reg": 0.01,
        "reg_m": 5,
        "norm_cost": True,
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "hidden_dim": 128,
        "n_hiddens": 3,
        "activation": "leakyrelu",
        "n_pretrain_epochs": 2000,
        "batch_size": 256,
        "lr": 1e-3,
        "reg": 0.01,
        "reg_m": 5,
        "norm_cost": True,
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
    """Load via OTP-FM loaders. Returns normalized marginals and metadata."""
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


def make_vgfm_df(marginals_np, train_times):
    """Build DataFrame in VGFM format: column 'samples' = integer index,
    remaining columns = features. Returns (df, idx_to_real_time)."""
    rows = []
    idx_to_real = {}
    for idx, t in enumerate(train_times):
        idx_to_real[idx] = t
        arr = marginals_np[t]
        df_t = pd.DataFrame(arr, columns=[f"x{i+1}" for i in range(arr.shape[1])])
        df_t.insert(0, "samples", idx)
        rows.append(df_t)
    df = pd.concat(rows, ignore_index=True)
    return df, idx_to_real


# ── Transport via ODE integration ───────────────────────────────────────────


@torch.no_grad()
def transport_vgfm(model, source_np, t_start, t_end, device):
    """Integrate VGFM ODE (velocity + growth) from t_start to t_end."""
    model.eval()
    x0 = torch.from_numpy(source_np).float().to(device)
    lnw0 = torch.zeros(x0.shape[0], device=device)
    initial_state = (x0, lnw0)
    t_span = torch.tensor([float(t_start), float(t_end)], device=device)
    generated, lnw = odeint(
        ODEFunc2(model), initial_state, t_span, method="euler", options=dict(step_size=0.05)
    )
    return generated[-1].cpu().numpy()


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
    logger.info(f"=== VGFM {exp_name} seed={seed} on {device} ===")

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "vgfm"
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
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data = load_data(cfg["dataset"], cfg["pca_dim"], holdout_times=holdout)
        all_times = data["all_times"]
        train_times = data["train_times"]

        logger.info(
            f"\n--- Fold {fold_idx+1}/{len(cfg['folds'])}: holdout={holdout}, "
            f"train={train_times}, all={all_times} ---"
        )

        df, idx_to_real = make_vgfm_df(data["marginals"], train_times)
        logger.info(f"    DataFrame shape: {df.shape}, train indices: {list(idx_to_real.keys())}")

        f_net = FNet(
            in_out_dim=dim,
            hidden_dim=cfg["hidden_dim"],
            n_hiddens=cfg["n_hiddens"],
            activation=cfg["activation"],
        ).to(device)
        n_params = sum(p.numel() for p in f_net.parameters())
        logger.info(f"    Model params: {n_params:,}")

        sample_sizes = df.groupby("samples").size()
        relative_mass = torch.tensor(
            (sample_sizes / sample_sizes.iloc[0]).values, dtype=torch.float32
        )
        optimizer = torch.optim.Adam(f_net.parameters(), lr=cfg["lr"])

        t_start = time.perf_counter()
        f_net, v_losses, g_losses, losses = pretrain(
            f_net,
            df,
            optimizer,
            n_epoch=cfg["n_pretrain_epochs"],
            hold_out=-999,  # sentinel: don't hold anything out during VGFM training
            logger=logging.getLogger("vgfm_pretrain"),
            device=device,
            relative_mass=relative_mass,
            reg=cfg["reg"],
            reg_m=[cfg["reg_m"], np.inf],
            norm_cost=cfg["norm_cost"],
            batch_size=cfg["batch_size"],
        )
        train_sec = time.perf_counter() - t_start
        all_train_times_sec.append(train_sec)
        logger.info(f"    Pretrain time: {train_sec:.1f}s  Final loss: {losses[-1]:.6f}")

        # Evaluate: integrate ODE from t=0 to each eval time
        # The ODE was trained on integer indices [0, 1, ..., n_train-1].
        # Map real eval times to this integer scale using linear interpolation.
        t_real_min = min(all_times)
        t_real_max = max(all_times)
        n_train = len(train_times)

        def real_to_ode_time(t_real):
            """Map real time to the ODE's integer-index time scale."""
            frac = (t_real - t_real_min) / (t_real_max - t_real_min)
            return frac * (n_train - 1)

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
            ode_t = real_to_ode_time(t_eval)
            gen = transport_vgfm(f_net, source, 0.0, ode_t, device)
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

        # Save checkpoint + fine trajectory (normalized PCA space).
        # The ODE was trained on integer indices [0, n_train-1]; integrate over
        # that range and remap t_eval -> [0, 1].
        f_net.eval()
        n_traj = min(2000, source.shape[0])
        x0 = torch.from_numpy(source[:n_traj].astype(np.float32)).to(device)
        lnw0 = torch.zeros(x0.shape[0], device=device)
        ode_t_max = float(n_train - 1)
        t_span_ode = torch.linspace(0.0, ode_t_max, 101, device=device)
        with torch.no_grad():
            generated, _ = odeint(
                ODEFunc2(f_net),
                (x0, lnw0),
                t_span_ode,
                method="euler",
                options=dict(step_size=0.05),
            )
        traj_np = generated.cpu().numpy().transpose(1, 0, 2).astype(np.float32)
        t_eval_norm = (t_span_ode.cpu().numpy() / ode_t_max).astype(np.float32)
        from _traj_utils import save_trajectory_npz, save_torch_checkpoint

        holdout_str = "_".join(map(str, holdout)) if holdout else "none"
        base = f"fold{fold_idx}_holdout{holdout_str}_seed{seed}"
        save_trajectory_npz(
            trajectories=traj_np,
            t_eval=t_eval_norm,
            marginal_times=all_times,
            method="vgfm",
            dataset=cfg["dataset"],
            dim=cfg["pca_dim"],
            output_path=out_dir / "trajectories" / f"{base}_trajectories.npz",
            config={k: v for k, v in cfg.items() if k != "folds"},
        )
        save_torch_checkpoint(
            {"f_net": f_net.state_dict()},
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
        "method": "vgfm",
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
    parser = argparse.ArgumentParser(description="VGFM evaluation harness")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else BASE_DIR / "results" / "vgfm"
    run_experiment(args.experiment, seed=args.seed, device_str=args.device, save_dir=save_dir)


if __name__ == "__main__":
    main()
