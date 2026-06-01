"""
iJKOnet evaluation harness.

Trains iJKOnet (inverse-jkonet-time-potential) then evaluates via
forward SDE simulation from t=0. Uses W1/W2/MMD following OTP-FM protocol.

Usage:
    conda run -n env_ijkonet python train_ijkonet_eval.py --experiment eb5_loo --seed 42
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OTP_FM_DIR = BASE_DIR / "OTP-FM"
IJKONET_DIR = Path(__file__).resolve().parents[1] / "baselines" / "iJKOnet"

sys.path.insert(0, str(IJKONET_DIR))
os.chdir(str(IJKONET_DIR))

import jax
import numpy as np
import yaml
from pathlib import Path
from models import EnumMethod, get_model
from utils.dataset.dataset import PopulationEvalDataset
from utils.sde_simulator import get_SDE_predictions
from utils.train.config import resolve_tau_for_solver
from torch.utils.data import DataLoader, Dataset


class FilteredPopulationDataset(Dataset):
    """PopulationDataset that excludes held-out time indices from training."""

    def __init__(self, dataset_name, data_dir, holdout_indices, batch_size):
        import math
        from collections import defaultdict

        base_path = Path(data_dir) / dataset_name
        data = np.load(base_path / "train_data.npy")
        labels = np.load(base_path / "train_sample_labels.npy")

        mask = np.ones(len(labels), dtype=bool)
        for h in holdout_indices:
            mask &= labels != h
        data = data[mask]
        labels = labels[mask]

        self.trajectory = defaultdict(list)
        for value, label in zip(data, labels):
            self.trajectory[label].append(value)
        for label in self.trajectory:
            self.trajectory[label] = np.array(self.trajectory[label])

        self.max_particles = max(p.shape[0] for p in self.trajectory.values())
        if self.max_particles % batch_size != 0:
            self.max_particles = math.ceil(self.max_particles / batch_size) * batch_size
        self.batch_size = batch_size
        self.random_indices = True
        self.data_dim = data.shape[1]
        logger.info(
            f"    FilteredPopulationDataset: {len(self.trajectory)} times "
            f"(excluded {holdout_indices}), {sum(len(v) for v in self.trajectory.values())} samples"
        )

    def __len__(self):
        return self.max_particles

    def __getitem__(self, idx):
        timesteps = sorted(self.trajectory.keys())
        num_timesteps = len(timesteps)
        indices = np.random.choice(self.max_particles, num_timesteps, replace=True)
        sampled = []
        for timestep, ind in zip(timesteps, indices):
            pool = self.trajectory[timestep]
            sampled.append(pool[ind % len(pool)])
        return sampled


def _import_from(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eb_data = _import_from("eb_data", OTP_FM_DIR / "experiments" / "singlecell" / "data.py")
_cite_data = None  # lazy-loaded; requires pandas


def _get_cite_data():
    global _cite_data
    if _cite_data is None:
        _cite_data = _import_from("cite_data", OTP_FM_DIR / "experiments" / "citeseq" / "data.py")
    return _cite_data


EXPERIMENT_CONFIGS = {
    "eb5_loo": {
        "dataset": "eb",
        "pca_dim": 5,
        "ijkonet_data": "RAW_RNA_eb_5",
        "folds": [[1], [2], [3]],
        "primary_metric": "w1",
        "primary_metric_space": "normalized",
        "epochs": 1000,
        "solver": "inverse-jkonet-time-potential",
    },
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "ijkonet_data": "RAW_RNA_eb_100",
        "folds": [[1, 3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "epochs": 1000,
        "solver": "inverse-jkonet-time-potential",
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "ijkonet_data": "RAW_RNA_cite_5",
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "normalized",
        "epochs": 1000,
        "solver": "inverse-jkonet-time-potential",
    },
    "cite50_loo": {
        "dataset": "cite",
        "pca_dim": 50,
        "ijkonet_data": "RAW_RNA_multi_50",
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "epochs": 1000,
        "solver": "inverse-jkonet-time-potential",
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
    import torch

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


# ── Data ─────────────────────────────────────────────────────────────────────


def load_reference_marginals(dataset, pca_dim, holdout_times):
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
        raw = _get_cite_data().load_citeseq_data(
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


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


# ── Main ─────────────────────────────────────────────────────────────────────


def _serialize_fold(holdout, fold_metrics):
    return {
        "holdout": list(holdout),
        "holdout_label": "_".join(map(str, holdout)) if holdout else "none",
        "per_time": {str(t): float(v) for t, v in fold_metrics.items()},
        "fold_avg": float(np.mean(list(fold_metrics.values()))) if fold_metrics else None,
    }


def run_experiment(exp_name, seed=42, save_dir=None):
    cfg = EXPERIMENT_CONFIGS[exp_name]
    logger.info(f"=== iJKOnet {exp_name} seed={seed} ===")

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "ijkonet"
    save_dir = Path(save_dir)
    out_dir = save_dir / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"seed{seed}.json"

    # Load configs (deep merge: base then method overlay)
    base_cfg = yaml.safe_load(open(str(IJKONET_DIR / "configs" / "config-base.yaml")))
    method_cfg = yaml.safe_load(
        open(str(IJKONET_DIR / "configs" / "config-method-inverse-jkonet.yaml"))
    )
    config = base_cfg.copy()
    for k, v in method_cfg.items():
        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
            config[k] = {**config[k], **v}
        else:
            config[k] = v
    config["train"]["epochs"] = cfg["epochs"]
    config.setdefault("K", 5)

    all_fold_metrics = []
    all_train_times_sec = []
    t_run_start = time.perf_counter()

    for fold_idx, holdout in enumerate(cfg["folds"]):
        key = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        data = load_reference_marginals(cfg["dataset"], cfg["pca_dim"], holdout_times=holdout)
        all_times = data["all_times"]

        logger.info(f"\n--- Fold {fold_idx+1}/{len(cfg['folds'])}: holdout={holdout} ---")

        dataset_name = cfg["ijkonet_data"]
        data_dir_str = str(IJKONET_DIR / "data")
        dt = config["train"]["dt"]
        dataset_eval = PopulationEvalDataset(
            key=key,
            dataset_name=dataset_name,
            solver=cfg["solver"],
            wasserstein_metric=config["metrics"]["wasserstein_order"],
            data_dir=data_dir_str,
            label="train_data",
            dt=dt,
        )

        tau = resolve_tau_for_solver(0.01, EnumMethod(cfg["solver"]), int(config["K"]))
        model = get_model(EnumMethod(cfg["solver"]), config, dataset_eval.data_dim, tau)
        state = model.create_state(key)
        batch_size = config["train"]["batch_size"]
        dataset_train = FilteredPopulationDataset(dataset_name, data_dir_str, holdout, batch_size)

        import torch

        torch.manual_seed(seed)
        batch_size = config["train"]["batch_size"]
        loader_train = DataLoader(
            dataset_train,
            batch_size=model.batch_size if batch_size > 0 else len(dataset_train),
            shuffle=True,
            collate_fn=numpy_collate,
        )
        loader_val = DataLoader(
            dataset_eval,
            batch_size=len(dataset_eval),
            shuffle=False,
            collate_fn=numpy_collate,
        )

        # Train
        epochs = cfg["epochs"]
        logger.info(f"    Training iJKOnet ({cfg['solver']}) for {epochs} epochs...")
        train_step = jax.jit(model.train_step)

        t_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            for sample in loader_train:
                state, metrics = train_step(state, sample)
            if epoch % 250 == 0:
                loss_e = metrics.get("loss_energy", float("nan"))
                logger.info(f"      Epoch {epoch}/{epochs} loss_energy={float(loss_e):.6f}")
        train_sec = time.perf_counter() - t_start
        all_train_times_sec.append(train_sec)
        logger.info(f"    Training time: {train_sec:.1f}s")

        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        holdout_str = "_".join(map(str, holdout))
        try:
            import pickle as _pickle

            ckpt_pkl = ckpt_dir / f"fold{fold_idx}_holdout{holdout_str}_seed{seed}.pkl"
            states_to_save = state if isinstance(state, tuple) else (state,)
            payload = {}
            for si, s in enumerate(states_to_save):
                payload[f"state{si}_params"] = jax.tree_util.tree_map(np.asarray, s.params)
                if hasattr(s, "step"):
                    payload[f"state{si}_step"] = int(s.step)
            with open(ckpt_pkl, "wb") as f:
                _pickle.dump(payload, f)
            logger.info(f"    Saved checkpoint to {ckpt_pkl}")
        except Exception as _e:
            logger.warning(f"    Failed to save checkpoint: {_e}")

        # Generate predictions via SDE simulation
        key, key_eval = jax.random.split(key)
        init_pp = next(iter(loader_val))
        potential = model.get_potential(state)
        beta = model.get_beta(state)
        interaction = model.get_interaction(state)

        predictions = get_SDE_predictions(
            cfg["solver"],
            dataset_eval.dt,
            dataset_eval.K,
            1,
            potential,
            beta,
            interaction,
            key_eval,
            init_pp,
        )

        predictions_np = np.array(predictions)
        logger.info(f"    Predictions shape: {predictions_np.shape}, K={dataset_eval.K}")

        # iJKOnet uses per-feature StandardScaler, OTP-FM CITE uses uniform max-std scaling.
        # Compute conversion for CITE datasets.
        if cfg["dataset"] == "cite" and data["scaler"] is not None:
            cite_mod = _get_cite_data()
            raw_cite = cite_mod.load_citeseq_data(
                data_dir=OTP_FM_DIR / "data",
                pca_dim=cfg["pca_dim"],
                normalize=False,
                ot_coupling=False,
            )
            raw_all = np.concatenate(
                [raw_cite["marginals"][t].numpy() for t in sorted(raw_cite["marginals"].keys())]
            )
            jko_mean = raw_all.mean(axis=0, keepdims=True)
            jko_std = raw_all.std(axis=0, keepdims=True)
        else:
            jko_mean = None
            jko_std = None

        metric_fn = METRIC_FNS[cfg["primary_metric"]]
        fold_results = {}

        for t_idx, t_real in enumerate(all_times):
            if t_real == all_times[0]:
                continue
            if t_real not in holdout and holdout:
                continue

            pred = predictions_np[t_idx]
            gt = data["marginals"][t_real]

            if cfg["dataset"] == "cite" and jko_mean is not None:
                pred_orig = pred * jko_std + jko_mean
                if cfg["primary_metric_space"] == "original":
                    pred_metric = pred_orig
                    gt_metric = data["scaler"].inverse_transform(gt)
                else:
                    pred_metric = data["scaler"].transform(pred_orig)
                    gt_metric = gt
            elif cfg["primary_metric_space"] == "original" and data["scaler"] is not None:
                pred_metric = data["scaler"].inverse_transform(pred)
                gt_metric = data["scaler"].inverse_transform(gt)
            else:
                pred_metric, gt_metric = pred, gt

            val = metric_fn(pred_metric, gt_metric)
            fold_results[t_real] = val
            logger.info(f"    t={t_real} (idx={t_idx}): {cfg['primary_metric']}={val:.6f}")

        all_fold_metrics.append(fold_results)

        # Save trajectory in normalized PCA space
        try:
            sys.path.insert(
                0, str(Path(__file__).resolve().parents[1] / "src" / "experiments" / "baselines")
            )
            from _traj_utils import save_trajectory_npz

            traj_steps = predictions_np  # (K+1, n_particles, dim)
            n_steps = traj_steps.shape[0]
            t_eval_norm = np.linspace(0.0, 1.0, n_steps).astype(np.float32)
            traj_for_save = traj_steps.transpose(1, 0, 2).astype(np.float32)
            n_traj = min(2000, traj_for_save.shape[0])
            traj_for_save = traj_for_save[:n_traj]
            if cfg["dataset"] == "cite" and jko_mean is not None:
                traj_for_save = traj_for_save * jko_std + jko_mean
                traj_for_save = (
                    data["scaler"]
                    .transform(traj_for_save.reshape(-1, cfg["pca_dim"]))
                    .reshape(traj_for_save.shape)
                    .astype(np.float32)
                )
            traj_dir = out_dir / "trajectories"
            base = f"fold{fold_idx}_holdout{holdout_str}_seed{seed}"
            save_trajectory_npz(
                trajectories=traj_for_save,
                t_eval=t_eval_norm,
                marginal_times=all_times,
                method="ijkonet",
                dataset=cfg["dataset"],
                dim=cfg["pca_dim"],
                output_path=traj_dir / f"{base}_trajectories.npz",
                config={k: v for k, v in cfg.items() if k != "folds"},
            )
            logger.info(f"    Saved trajectory to {traj_dir / (base + '_trajectories.npz')}")
        except Exception as e:
            logger.warning(f"    Failed to save trajectory: {e}")

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
        "method": "ijkonet",
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
    parser = argparse.ArgumentParser(description="iJKOnet evaluation harness")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_absolute():
            save_dir = BASE_DIR / args.save_dir
    else:
        save_dir = BASE_DIR / "results" / "ijkonet"
    run_experiment(args.experiment, seed=args.seed, save_dir=save_dir)


if __name__ == "__main__":
    main()
