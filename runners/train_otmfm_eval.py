"""
OT-MFM (Metric Flow Matching) evaluation harness.

Implements OT-MFM training without the full pytorch-lightning pipeline:
1. Trains geopath network (learns geodesic deformation)
2. Trains flow network (learns velocity conditioned on geopath)
3. Evaluates by integrating the flow network with torchdiffeq

Uses precomputed OT couplings (same as OT-CFM) and the MFM's geodesic
conditional flow matcher.

Usage:
    conda run -n env_mfm python train_otmfm_eval.py --experiment eb5_l2o --seed 42
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"

import matplotlib

matplotlib.use("Agg")
import numpy as np
import ot as pot_lib
import torch
import torchdiffeq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OTP_FM_DIR = BASE_DIR / "OTP-FM"
MFM_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "metric-flow-matching"
sys.path.insert(0, str(MFM_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _runtime_patches import patch_mfm_rbf_eps

patch_mfm_rbf_eps()  # guard against zero sigma in RBFNetwork

from _traj_utils import save_trajectory_and_checkpoint_torchdiffeq
from mfm.flow_matchers.models.mfm import MetricFlowMatcher
from mfm.networks.flow_networks.mlp import VelocityNet
from mfm.networks.geopath_networks.mlp import GeoPathMLP

# Patch torch.load for compat
_orig = torch.load


def _patched(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)


torch.load = _patched


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
        "hidden_dims_flow": [64, 64, 64],
        "hidden_dims_geopath": [64, 64, 64],
        "geopath_epochs": 500,
        "flow_epochs": 500,
        "lr": 1e-3,
        "batch_size": 256,
        "sigma": 0.1,
    },
    "eb100_loo": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1], [2], [3], []],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "hidden_dims_flow": [1024, 1024, 1024],
        "hidden_dims_geopath": [1024, 1024, 1024],
        "geopath_epochs": 200,
        "flow_epochs": 500,
        "lr": 1e-3,
        "batch_size": 256,
        "sigma": 0.1,
    },
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1, 3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "hidden_dims_flow": [1024, 1024, 1024],
        "hidden_dims_geopath": [1024, 1024, 1024],
        "geopath_epochs": 200,
        "flow_epochs": 500,
        "lr": 1e-3,
        "batch_size": 256,
        "sigma": 0.1,
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "hidden_dims_flow": [64, 64, 64],
        "hidden_dims_geopath": [64, 64, 64],
        "geopath_epochs": 500,
        "flow_epochs": 500,
        "lr": 1e-3,
        "batch_size": 256,
        "sigma": 0.1,
    },
}


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_w1(gen, gt):
    n = min(len(gen), len(gt))
    g, r = gen[:n].astype(np.float64), gt[:n].astype(np.float64)
    M = pot_lib.dist(g, r, metric="euclidean")
    a, b = np.ones(n) / n, np.ones(n) / n
    return float(pot_lib.emd2(a, b, M, numItermax=int(1e7)))


def compute_w2(gen, gt, max_dim=10):
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
            ot_coupling=True,
            holdout_times=holdout_times,
        )
    elif dataset == "cite":
        raw = _cite_data.load_citeseq_data(
            data_dir=OTP_FM_DIR / "data",
            pca_dim=pca_dim,
            normalize=True,
            ot_coupling=True,
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
        "ot_alignments": raw["ot_alignments"],
    }


# ── Training ─────────────────────────────────────────────────────────────────


def _make_batched_indices(n_samples, batch_size):
    """Yield shuffled batch index arrays, matching PL DataLoader behaviour."""
    perm = np.random.permutation(n_samples)
    for start in range(0, n_samples - batch_size + 1, batch_size):
        yield perm[start : start + batch_size]


def train_otmfm(
    flow_net, geopath_net, fm, X_by_train_time, tp_norm, train_times, ot_alignments, cfg, device
):
    """Train geopath then flow network using MetricFlowMatcher.

    Iterates over the full dataset per epoch (multiple batches), matching
    the original PL pipeline's CombinedLoader behaviour.
    """
    batch_size = cfg["batch_size"]
    n_train = len(X_by_train_time)
    min_n = min(arr.shape[0] for arr in X_by_train_time)
    n_batches = max(1, min_n // batch_size)
    logger.info(
        f"    Data: {n_train} timepoints, min_n={min_n}, "
        f"batch_size={batch_size}, {n_batches} batches/epoch"
    )

    # Phase 1: Train geopath network
    logger.info("    Phase 1: Training geopath network...")
    geopath_opt = torch.optim.Adam(geopath_net.parameters(), lr=cfg["lr"])
    geopath_net.train()
    for epoch in range(cfg["geopath_epochs"]):
        batch_iters = [
            list(_make_batched_indices(arr.shape[0], batch_size)) for arr in X_by_train_time
        ]
        actual_batches = min(len(bi) for bi in batch_iters)
        for b in range(actual_batches):
            geopath_opt.zero_grad()
            loss_total = 0.0
            for seg in range(n_train - 1):
                idx0 = batch_iters[seg][b]
                t_src, t_tgt = train_times[seg], train_times[seg + 1]
                mapping = ot_alignments[(t_src, t_tgt)]
                idx1 = mapping[idx0]
                x0 = torch.from_numpy(X_by_train_time[seg][idx0]).float().to(device)
                x1 = torch.from_numpy(X_by_train_time[seg + 1][idx1]).float().to(device)
                t_min, t_max = float(tp_norm[seg]), float(tp_norm[seg + 1])
                t, xt, ut = fm.sample_location_and_conditional_flow(
                    x0, x1, t_min, t_max, training_geopath_net=True
                )
                vt = flow_net(t[:, None], xt)
                loss_total += torch.mean((vt - ut) ** 2)
            loss_total.backward()
            geopath_opt.step()
        if (epoch + 1) % 100 == 0:
            logger.info(
                f"      Geopath epoch {epoch+1}/{cfg['geopath_epochs']} loss={loss_total.item():.6f}"
            )

    # Phase 2: Train flow network with frozen geopath
    logger.info("    Phase 2: Training flow network...")
    geopath_net.eval()
    flow_opt = torch.optim.Adam(flow_net.parameters(), lr=cfg["lr"])
    flow_net.train()
    for epoch in range(cfg["flow_epochs"]):
        batch_iters = [
            list(_make_batched_indices(arr.shape[0], batch_size)) for arr in X_by_train_time
        ]
        actual_batches = min(len(bi) for bi in batch_iters)
        for b in range(actual_batches):
            flow_opt.zero_grad()
            loss_total = 0.0
            for seg in range(n_train - 1):
                idx0 = batch_iters[seg][b]
                t_src, t_tgt = train_times[seg], train_times[seg + 1]
                mapping = ot_alignments[(t_src, t_tgt)]
                idx1 = mapping[idx0]
                x0 = torch.from_numpy(X_by_train_time[seg][idx0]).float().to(device)
                x1 = torch.from_numpy(X_by_train_time[seg + 1][idx1]).float().to(device)
                t_min, t_max = float(tp_norm[seg]), float(tp_norm[seg + 1])
                with torch.no_grad():
                    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1, t_min, t_max)
                vt = flow_net(t[:, None], xt)
                loss_total += torch.mean((vt - ut) ** 2)
            loss_total.backward()
            flow_opt.step()
        if (epoch + 1) % 100 == 0:
            logger.info(
                f"      Flow epoch {epoch+1}/{cfg['flow_epochs']} loss={loss_total.item():.6f}"
            )


# ── Transport ────────────────────────────────────────────────────────────────


@torch.no_grad()
def transport(flow_net, source, t_start, t_end, device):
    flow_net.eval()
    x0 = torch.from_numpy(source).float().to(device)

    def ode_fn(t, x):
        t_vec = t.expand(x.shape[0])[:, None]
        return flow_net(t_vec, x)

    t_span = torch.tensor([t_start, t_end], device=device)
    traj = torchdiffeq.odeint(ode_fn, x0, t_span, method="dopri5", atol=1e-5, rtol=1e-5)
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
    logger.info(f"=== OT-MFM {exp_name} seed={seed} on {device} ===")

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "otmfm"
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

        data = load_data(cfg["dataset"], cfg["pca_dim"], holdout_times=holdout)
        all_times = data["all_times"]
        train_times = data["train_times"]
        t_min, t_max = min(all_times), max(all_times)
        tp_norm = np.array([(t - t_min) / (t_max - t_min) for t in train_times], dtype=np.float32)

        logger.info(
            f"\n--- Fold {fold_idx+1}/{len(cfg['folds'])}: holdout={holdout}, train={train_times} ---"
        )

        flow_net = VelocityNet(dim=dim, hidden_dims=cfg["hidden_dims_flow"], activation="silu").to(
            device
        )
        geopath_net = GeoPathMLP(
            input_dim=dim,
            hidden_dims=cfg["hidden_dims_geopath"],
            time_geopath=True,
            activation="silu",
            batch_norm=False,
        ).to(device)
        fm = MetricFlowMatcher(geopath_net=geopath_net, sigma=cfg["sigma"], alpha=1)

        n_params = sum(p.numel() for p in flow_net.parameters()) + sum(
            p.numel() for p in geopath_net.parameters()
        )
        logger.info(f"    Model params: {n_params:,}")

        X_by_train_time = [data["marginals"][t] for t in train_times]

        t_start = time.perf_counter()
        train_otmfm(
            flow_net,
            geopath_net,
            fm,
            X_by_train_time,
            tp_norm,
            train_times,
            data["ot_alignments"],
            cfg,
            device,
        )
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

        def norm_t(t):
            return (t - t_min) / (t_max - t_min)

        for t_eval in all_times:
            if t_eval == all_times[0]:
                continue
            gen = transport(flow_net, source, norm_t(all_times[0]), norm_t(t_eval), device)
            if t_eval in metric_times:
                gt = data["marginals"][t_eval]
                gen_metric, gt_metric = gen, gt
                if cfg["primary_metric_space"] == "original" and data["scaler"] is not None:
                    gen_metric = data["scaler"].inverse_transform(gen)
                    gt_metric = data["scaler"].inverse_transform(gt)
                val = metric_fn(gen_metric, gt_metric)
                fold_results[t_eval] = val
                logger.info(f"    t={t_eval}: {cfg['primary_metric']}={val:.6f}")

        all_fold_metrics.append(fold_results)

        # Save checkpoint + fine trajectory for downstream PCA plots (normalized space)
        def _ode_fn(t, x, _model=flow_net):
            t_vec = t.expand(x.shape[0])[:, None]
            return _model(t_vec, x)

        save_trajectory_and_checkpoint_torchdiffeq(
            model=flow_net,
            source_np=source,
            ode_fn=_ode_fn,
            out_dir=out_dir,
            fold_idx=fold_idx,
            holdout=holdout,
            seed=seed,
            method="otmfm",
            dataset=cfg["dataset"],
            dim=cfg["pca_dim"],
            marginal_times=all_times,
            config={k: v for k, v in cfg.items() if k != "folds"},
            state_dict_pieces={
                "flow_net": flow_net.state_dict(),
                "geopath_net": geopath_net.state_dict(),
            },
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
    if exp_name in ("eb100_loo", "eb100_l2o"):
        grand_avg = float(np.mean([np.mean(list(f.values())) for f in all_fold_metrics]))
    else:
        all_values = [v for fold in all_fold_metrics for v in fold.values()]
        grand_avg = float(np.mean(all_values))
    logger.info(f"  Grand avg {cfg['primary_metric'].upper()}: {grand_avg:.6f}")

    total_sec = time.perf_counter() - t_run_start
    payload = {
        "experiment": exp_name,
        "method": "otmfm",
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
    parser = argparse.ArgumentParser(description="OT-MFM evaluation harness")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else BASE_DIR / "results" / "otmfm"
    run_experiment(args.experiment, seed=args.seed, device_str=args.device, save_dir=save_dir)


if __name__ == "__main__":
    main()
