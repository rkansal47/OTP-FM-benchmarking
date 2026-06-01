#!/usr/bin/env python
"""
NLSB (Neural Lagrangian Schrödinger Bridge) evaluation.

Trains NLSB on OTP-FM data and evaluates at held-out timepoints.
Uses the cellular Lagrangian (GMM-based Waddington's landscape prior).

Usage:
    conda run -n env_mmfm python -u train_nlsb_eval.py --experiment eb5_l2o --seed 42
    conda run -n env_mmfm python -u train_nlsb_eval.py --experiment eb100_l2o --seed 42
    conda run -n env_mmfm python -u train_nlsb_eval.py --experiment cite5_loo --seed 42
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import ot
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OTP_FM_DIR = BASE_DIR / "OTP-FM"
NLSB_DIR = Path(__file__).resolve().parents[1] / "baselines" / "NLSB"

sys.path.insert(0, str(NLSB_DIR))
os.chdir(str(NLSB_DIR))

from model import SDENet, SDE_MODEL_NAME, LAGRANGIAN_NAME
from dataset import scRNASeq, BalancedBatchSampler


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
        "epochs": 500,
        "batch_size": 1000,
        "lr": 1e-3,
        "hidden_m": 32,
        "diffusion_hidden": 16,
        "n_gmm_components": 8,
    },
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1, 3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "epochs": 500,
        "batch_size": 1000,
        "lr": 1e-3,
        "hidden_m": 32,
        "diffusion_hidden": 16,
        "n_gmm_components": 8,
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "epochs": 500,
        "batch_size": 1000,
        "lr": 1e-3,
        "hidden_m": 32,
        "diffusion_hidden": 16,
        "n_gmm_components": 8,
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────


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


def make_nlsb_npz(marginals, train_times, all_times, pca_dim, tmpdir):
    """Create npz files in NLSB's expected format from OTP-FM marginals."""
    X_list, ts_list = [], []
    time_to_int = {t: i for i, t in enumerate(all_times)}

    for t in all_times:
        arr = marginals[t][:, :pca_dim]
        X_list.append(arr)
        ts_list.append(np.full(len(arr), time_to_int[t], dtype=np.float64))

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    ts = np.concatenate(ts_list, axis=0)

    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    train_path = str(tmpdir / "train.npz")
    np.savez(train_path, X=X, ts=ts)
    val_path = str(tmpdir / "val.npz")
    np.savez(val_path, X=X, ts=ts)

    return train_path, val_path, time_to_int


# ── Training ──────────────────────────────────────────────────────────────────


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_nlsb(cfg, train_path, val_path, pca_dim, holdout_int_times, all_int_times, device, seed):
    """Train NLSB and return the trained model + scaler from dataset."""
    fix_seed(seed)

    lmt_val = holdout_int_times[0] if len(holdout_int_times) == 1 else -1

    tr_ds = scRNASeq([train_path], pca_dim, use_v=False, LMT=lmt_val)
    va_ds = scRNASeq([val_path], pca_dim, use_v=False, LMT=lmt_val, scaler=tr_ds.get_scaler())

    if len(holdout_int_times) > 1:
        keep_mask_tr = torch.ones(len(tr_ds.X), dtype=torch.bool)
        for ht in holdout_int_times:
            keep_mask_tr &= tr_ds.labels != ht
        tr_ds.X = tr_ds.X[keep_mask_tr]
        tr_ds.labels = tr_ds.labels[keep_mask_tr]
        tr_ds.ncells = tr_ds.X.shape[0]
        tr_ds.t_set = sorted(list(set(tr_ds.labels.numpy())))
        tr_ds._full_data = dict(
            X=torch.cat([torch.from_numpy(tr_ds.y0[:, :pca_dim]).float(), tr_ds.X], dim=0),
            t=torch.cat([torch.zeros(len(tr_ds.y0)), tr_ds.labels], dim=0),
        )

        keep_mask_va = torch.ones(len(va_ds.X), dtype=torch.bool)
        for ht in holdout_int_times:
            keep_mask_va &= va_ds.labels != ht
        va_ds.X = va_ds.X[keep_mask_va]
        va_ds.labels = va_ds.labels[keep_mask_va]
        va_ds.ncells = va_ds.X.shape[0]
        va_ds.t_set = sorted(list(set(va_ds.labels.numpy())))

    t_set = tr_ds.get_label_set()
    train_t_set = t_set[:]
    n_components_list = [cfg["n_gmm_components"]] * len(t_set)

    logger.info(f"  Train timepoints: {train_t_set}")
    logger.info(f"  Holdout int times: {holdout_int_times}")
    logger.info(f"  Samples: {tr_ds.ncells}")

    L = LAGRANGIAN_NAME["cellular"](
        tr_ds.full_data["X"],
        tr_ds.full_data["t"],
        n_components_list=n_components_list,
        lm_u2=0.0,
        lm_U=10.0,
        lm_v=0.0,
        device=device,
    )
    net = SDE_MODEL_NAME["ito"](
        noise_type="diagonal",
        sigma_type="MLP",
        input_dim=pca_dim,
        brownian_size=pca_dim,
        drift_cfg={"nTh": 2, "m": cfg["hidden_m"], "use_t": True},
        diffusion_cfg={
            "hidden_dim": cfg["diffusion_hidden"],
            "num_layers": 2,
            "tanh": True,
            "use_t": True,
        },
        criterion_cfg={"alpha_D": 1.0, "alpha_L": 0.01, "alpha_R": 0.001, "p": 2, "blur": 0.05},
        solver_cfg={"adjoint": False, "dt": 0.01, "method": "euler", "adaptive": False},
        lagrangian=L,
    )
    model = SDENet(net, device)
    model.to(device)

    optimizer = optim.Adam(model.parameters_lr(), lr=cfg["lr"])
    batch_sampler_tr = BalancedBatchSampler(tr_ds, cfg["batch_size"])
    tr_dl = DataLoader(tr_ds, batch_sampler=batch_sampler_tr)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        outputs = []
        for batch_idx, train_batch in enumerate(tr_dl):
            train_batch["base"] = tr_ds.base_sample(cfg["batch_size"])
            optimizer.zero_grad()
            out = model.training_step(train_batch, batch_idx, train_t_set, tr_ds.T0)
            outputs.append(out)
            loss = out["loss"]
            loss.backward()
            del train_batch
            optimizer.step()
            if hasattr(model, "clamp_parameters"):
                model.clamp_parameters()

        train_result = model.training_epoch_end(outputs)
        avg_loss = train_result["avg_loss"]

        if epoch % 50 == 0:
            logger.info(f"  Epoch {epoch}/{cfg['epochs']}: loss={avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                "net": {k: v.clone() for k, v in model.net.state_dict().items()},
            }

    if best_state is not None:
        model.net.load_state_dict(best_state["net"])

    return model, tr_ds.get_scaler()


# ── Transport & Metrics ───────────────────────────────────────────────────────


@torch.no_grad()
def transport_nlsb(model, source_np, t_start, t_end, device, num_repeat=5):
    """Transport samples from t_start to t_end using the SDE model."""
    model.eval()
    x0 = torch.from_numpy(source_np).float().to(device)
    int_time = [t_start, t_end]
    traj = model.sample_with_uncertainty(x0, int_time, num_repeat)
    pred = traj[:, -1, :, :].mean(dim=1)
    return pred.cpu().numpy()


def compute_w1(gen, ref):
    n = min(len(gen), len(ref))
    M = ot.dist(gen[:n], ref[:n])
    return ot.emd2(np.ones(n) / n, np.ones(n) / n, M) ** 0.5


def compute_w2(gen, ref, max_dim=10):
    d = min(gen.shape[1], max_dim)
    g, r = gen[:, :d], ref[:, :d]
    n = min(len(g), len(r))
    M = ot.dist(g[:n], r[:n])
    return ot.emd2(np.ones(n) / n, np.ones(n) / n, M) ** 0.5


def compute_mmd(gen, ref, gamma=None):
    from scipy.spatial.distance import cdist

    if gamma is None:
        dists = cdist(ref, ref, "sqeuclidean")
        gamma = 1.0 / np.median(dists[dists > 0])
    K_rr = np.exp(-gamma * cdist(ref, ref, "sqeuclidean"))
    K_gg = np.exp(-gamma * cdist(gen, gen, "sqeuclidean"))
    K_rg = np.exp(-gamma * cdist(ref, gen, "sqeuclidean"))
    return float(K_rr.mean() + K_gg.mean() - 2 * K_rg.mean())


# ── Experiment runner ─────────────────────────────────────────────────────────


def run_experiment(exp_name, seed=42, device_str="cuda", save_dir=None):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"=== NLSB {exp_name} seed={seed} on {device} ===")

    cfg = EXPERIMENT_CONFIGS[exp_name]
    results = {"experiment": exp_name, "seed": seed, "method": "nlsb", "folds": []}

    t_start_total = time.time()

    for fold_idx, holdout in enumerate(cfg["folds"]):
        logger.info(f"--- Fold {fold_idx}: holdout={holdout} ---")
        t_fold_start = time.time()

        data = load_data(cfg["dataset"], cfg["pca_dim"], holdout)
        all_times = data["all_times"]
        train_times = data["train_times"]
        scaler = data["scaler"]

        time_to_int = {t: i for i, t in enumerate(all_times)}
        holdout_int = [time_to_int[t] for t in holdout]
        all_int = list(range(len(all_times)))

        tmpdir = BASE_DIR / "logs" / "baselines_eval" / f"nlsb_{exp_name}_fold{fold_idx}"
        train_path, val_path, _ = make_nlsb_npz(
            data["marginals"], train_times, all_times, cfg["pca_dim"], tmpdir
        )

        model, nlsb_scaler = train_nlsb(
            cfg, train_path, val_path, cfg["pca_dim"], holdout_int, all_int, device, seed
        )

        ckpt_dir = save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        holdout_str = "_".join(map(str, holdout))
        ckpt_path = ckpt_dir / f"{exp_name}_holdout{holdout_str}_seed{seed}.pt"
        torch.save({"net": model.net.state_dict()}, ckpt_path)
        logger.info(f"  Saved checkpoint to {ckpt_path}")

        fold_result = {"holdout": holdout, "metrics": {}}
        source_time = all_times[0]
        source_int = 0

        for t_eval in holdout:
            t_eval_int = time_to_int[t_eval]

            source_data = data["marginals"][source_time][:, : cfg["pca_dim"]]
            source_scaled = nlsb_scaler.transform(source_data)

            gen_scaled = transport_nlsb(
                model, source_scaled, float(source_int), float(t_eval_int), device
            )
            gen_normalized = nlsb_scaler.inverse_transform(gen_scaled)
            ref_normalized = data["marginals"][t_eval][:, : cfg["pca_dim"]]

            if scaler is not None:
                gen_original = scaler.inverse_transform(gen_normalized)
                ref_original = scaler.inverse_transform(ref_normalized)
            else:
                gen_original = gen_normalized
                ref_original = ref_normalized

            metrics = {}
            if cfg["primary_metric"] == "w1":
                if cfg["primary_metric_space"] == "normalized":
                    metrics["w1"] = float(compute_w1(gen_normalized, ref_normalized))
                else:
                    metrics["w1"] = float(compute_w1(gen_original, ref_original))
            elif cfg["primary_metric"] == "w2":
                if cfg["primary_metric_space"] == "normalized":
                    metrics["w2"] = float(compute_w2(gen_normalized, ref_normalized))
                else:
                    metrics["w2"] = float(compute_w2(gen_original, ref_original))
            elif cfg["primary_metric"] == "mmd":
                metrics["mmd"] = float(compute_mmd(gen_original, ref_original))

            fold_result["metrics"][str(t_eval)] = metrics
            logger.info(f"  t={t_eval}: {metrics}")

        fold_result["train_sec"] = time.time() - t_fold_start
        results["folds"].append(fold_result)

        # Save fine trajectory (normalized PCA space) for downstream PCA plots
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from _traj_utils import save_trajectory_npz

            n_int = len(all_times)
            t_max_int = float(n_int - 1)
            n_traj = min(2000, data["marginals"][source_time].shape[0])
            src_norm = data["marginals"][source_time][:n_traj, : cfg["pca_dim"]]
            src_scaled = nlsb_scaler.transform(src_norm)
            x0 = torch.from_numpy(src_scaled).float().to(device)
            fine_int_time = torch.linspace(0.0, t_max_int, 101, device=device)
            with torch.no_grad():
                traj = model.sample_with_uncertainty(x0, fine_int_time.tolist(), 1)
            traj_mean = traj[:, :, :, :].mean(dim=2).cpu().numpy()
            traj_mean = (
                nlsb_scaler.inverse_transform(traj_mean.reshape(-1, cfg["pca_dim"]))
                .reshape(traj_mean.shape)
                .astype(np.float32)
            )
            t_eval_norm = (fine_int_time.cpu().numpy() / t_max_int).astype(np.float32)
            traj_dir = save_dir / "trajectories"
            base = f"{exp_name}_holdout{holdout_str}_seed{seed}"
            save_trajectory_npz(
                trajectories=traj_mean,
                t_eval=t_eval_norm,
                marginal_times=all_times,
                method="nlsb",
                dataset=cfg["dataset"],
                dim=cfg["pca_dim"],
                output_path=traj_dir / f"{base}_trajectories.npz",
                config={k: v for k, v in cfg.items() if k != "folds"},
            )
            logger.info(f"  Saved trajectory to {traj_dir / (base + '_trajectories.npz')}")
        except Exception as e:
            logger.warning(f"  Failed to save trajectory for fold {fold_idx}: {e}")

    results["total_sec"] = time.time() - t_start_total

    primary = cfg["primary_metric"]
    all_values = []
    for fold in results["folds"]:
        for t_key, m in fold["metrics"].items():
            if primary in m:
                all_values.append(m[primary])
    results["primary_metric"] = primary
    results["primary_value"] = float(np.mean(all_values)) if all_values else None

    logger.info(
        f"=== NLSB {exp_name} DONE: {primary}={results['primary_value']:.4f} "
        f"({results['total_sec']:.0f}s) ==="
    )

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "nlsb" / exp_name
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    a = ap.parse_args()
    save_dir = BASE_DIR / "results" / "nlsb" / a.experiment
    run_experiment(a.experiment, seed=a.seed, device_str=a.device, save_dir=save_dir)
