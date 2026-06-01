"""
JKOnet* evaluation harness.

Trains JKOnet* (jkonet-star-time-potential) on the full training data,
then evaluates by forward SDE simulation from t=0 to all evaluation times.
Computes W1/W2/MMD following the OTP-FM protocol.

JKOnet* learns energy functionals (potential + internal + interaction) for
Wasserstein gradient flows. After training, get_SDE_predictions() forward-
simulates particles from the initial distribution.

For holdout evaluation: since JKOnet* learns continuous dynamics via
potentials, we train on all available data then evaluate at specific times.
The model quality is measured by how well it transports the t=0 marginal
to later time points.

Usage:
    conda run -n env_jkonet python train_jkonet_eval.py --experiment eb5_loo --seed 42
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
JKONET_DIR = Path(__file__).resolve().parents[1] / "baselines" / "jkonet-star"

sys.path.insert(0, str(JKONET_DIR))
os.chdir(str(JKONET_DIR))

import jax
import numpy as np
import yaml
from models import EnumMethod, get_model
from dataset import PopulationEvalDataset
from utils.sde_simulator import get_SDE_predictions
from torch.utils.data import DataLoader, Dataset


class FilteredCouplingsDataset(Dataset):
    """Computes OT couplings between consecutive TRAINING marginals, excluding holdout."""

    def __init__(self, dataset_name, holdout_indices):
        import ot as pot_lib
        from sklearn.mixture import GaussianMixture

        data_dir = os.path.join("data", dataset_name)
        all_data = np.load(os.path.join(data_dir, "data.npy"))
        all_labels = np.load(os.path.join(data_dir, "sample_labels.npy"))
        dim = all_data.shape[1]

        unique_times = sorted(set(all_labels))
        train_times = [t for t in unique_times if t not in holdout_indices]
        logger.info(
            f"    Computing couplings for train times {train_times} " f"(holdout={holdout_indices})"
        )

        all_couplings = []
        all_densities = []

        for i in range(len(train_times) - 1):
            t_src, t_tgt = train_times[i], train_times[i + 1]
            src_data = all_data[all_labels == t_src].astype(np.float64)
            tgt_data = all_data[all_labels == t_tgt].astype(np.float64)

            a = np.ones(len(src_data)) / len(src_data)
            b = np.ones(len(tgt_data)) / len(tgt_data)
            M = pot_lib.dist(src_data, tgt_data)
            plan = pot_lib.emd(a, b, M, numItermax=1_000_000)

            min_prob = 1.0 / (10 * max(len(src_data), len(tgt_data)))
            idx_src, idx_tgt = np.where(plan > min_prob)
            weights = plan[idx_src, idx_tgt]

            x_coupled = src_data[idx_src].astype(np.float32)
            y_coupled = tgt_data[idx_tgt].astype(np.float32)
            t_col = np.full(len(weights), float(t_tgt), dtype=np.float32)
            w_col = weights.astype(np.float32)

            coupling_block = np.column_stack([x_coupled, y_coupled, t_col, w_col])
            all_couplings.append(coupling_block)

            n_components = min(10, len(tgt_data) // 10)
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(tgt_data)
            log_dens = gmm.score_samples(y_coupled.astype(np.float64))
            densities_val = np.exp(log_dens).astype(np.float32).reshape(-1, 1)
            densities_grad = np.zeros((len(y_coupled), dim), dtype=np.float32)
            all_densities.append(np.concatenate([densities_val, densities_grad], axis=1))

            logger.info(
                f"      {t_src}->{t_tgt}: {len(idx_src)} couplings from "
                f"{len(src_data)}x{len(tgt_data)} marginals"
            )

        couplings = np.concatenate(all_couplings)
        self.weight = couplings[:, -1]
        self.x = couplings[:, :dim]
        self.y = couplings[:, dim:-2]
        self.time = couplings[:, -2]
        densities = np.concatenate(all_densities)
        self.densities = densities[:, 0]
        self.densities_grads = densities[:, 1:]
        logger.info(f"    Total: {len(self.x)} coupling pairs")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.time[idx],
            self.weight[idx],
            self.densities[idx],
            self.densities_grads[idx],
        )


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
        "jkonet_data": "EB_5D",
        "folds": [[1], [2], [3]],
        "primary_metric": "w1",
        "primary_metric_space": "normalized",
        "epochs": 100,
        "solver": "jkonet-star-time-potential",
    },
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "jkonet_data": "EB_100D",
        "folds": [[1, 3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "epochs": 100,
        "solver": "jkonet-star-time-potential",
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "jkonet_data": "CITE_5D",
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "normalized",
        "epochs": 100,
        "solver": "jkonet-star-time-potential",
    },
    "cite50_loo": {
        "dataset": "cite",
        "pca_dim": 50,
        "jkonet_data": "CITE_50D",
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "epochs": 100,
        "solver": "jkonet-star-time-potential",
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
    """Load ground-truth marginals from OTP-FM loaders for metric computation."""
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
    logger.info(f"=== JKOnet* {exp_name} seed={seed} ===")

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "jkonet"
    save_dir = Path(save_dir)
    out_dir = save_dir / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"seed{seed}.json"

    config = yaml.safe_load(open(str(JKONET_DIR / "config.yaml")))
    jkonet_config = yaml.safe_load(open(str(JKONET_DIR / "config-jkonet-extra.yaml")))
    config.update(jkonet_config)
    config["train"]["epochs"] = cfg["epochs"]

    all_fold_metrics = []
    all_train_times_sec = []
    t_run_start = time.perf_counter()

    for fold_idx, holdout in enumerate(cfg["folds"]):
        key = jax.random.PRNGKey(seed)
        np.random.seed(seed)

        # Load reference marginals
        data = load_reference_marginals(cfg["dataset"], cfg["pca_dim"], holdout_times=holdout)
        all_times = data["all_times"]

        logger.info(f"\n--- Fold {fold_idx+1}/{len(cfg['folds'])}: holdout={holdout} ---")

        dataset_name = cfg["jkonet_data"]
        dataset_eval = PopulationEvalDataset(
            key, dataset_name, cfg["solver"], config["metrics"]["wasserstein_error"], "train_data"
        )
        model = get_model(EnumMethod(cfg["solver"]), config, dataset_eval.data_dim, dataset_eval.dt)
        state = model.create_state(key)
        dataset_train = FilteredCouplingsDataset(dataset_name, holdout)

        import torch

        torch.manual_seed(seed)
        batch_size = config["train"]["batch_size"]
        loader_train = DataLoader(
            dataset_train,
            batch_size=batch_size if batch_size > 0 else len(dataset_train),
            shuffle=True,
            collate_fn=numpy_collate,
        )
        loader_val = DataLoader(
            dataset_eval, batch_size=len(dataset_eval), shuffle=False, collate_fn=numpy_collate
        )

        # Train
        logger.info(f"    Training JKOnet* ({cfg['solver']}) for {cfg['epochs']} epochs...")
        epochs = cfg["epochs"]
        train_step = jax.jit(model.train_step)

        t_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            loss = 0
        for sample in loader_train:
            step_loss, state = train_step(state, sample)
            loss += step_loss
            if epoch % 25 == 0:
                logger.info(f"      Epoch {epoch}/{epochs} loss={loss/len(loader_train):.6f}")
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
            dataset_eval.T,
            1,
            potential,
            beta,
            interaction,
            key_eval,
            init_pp,
        )

        # predictions shape: (T+1, n_particles, dim)  where T is number of steps
        predictions_np = np.array(predictions)
        logger.info(f"    Predictions shape: {predictions_np.shape}, T={dataset_eval.T}")

        # JKOnet uses per-feature StandardScaler (each feature -> unit variance).
        # OTP-FM uses uniform max-std scaling. For CITE datasets these differ.
        # We need to convert JKOnet predictions from JKOnet-normalized to evaluation space.
        if cfg["dataset"] == "cite" and data["scaler"] is not None:
            raw_cite = _cite_data.load_citeseq_data(
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

            pred = predictions_np[t_idx]  # (n_particles, dim) in JKOnet-normalized space
            gt = data["marginals"][t_real]  # in OTP-FM normalized space

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

        # Save trajectory in normalized PCA space (matching plot expectations).
        # JKOnet outputs (T+1, n_particles, dim) at integer time steps.
        try:
            sys.path.insert(
                0, str(Path(__file__).resolve().parents[1] / "src" / "experiments" / "baselines")
            )
            from _traj_utils import save_trajectory_npz

            traj_steps = predictions_np  # (T+1, n_particles, dim)
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
                method="jkonet",
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
        "method": "jkonet",
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
    parser = argparse.ArgumentParser(description="JKOnet* evaluation harness")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_absolute():
            save_dir = BASE_DIR / args.save_dir
    else:
        save_dir = BASE_DIR / "results" / "jkonet"
    run_experiment(args.experiment, seed=args.seed, save_dir=save_dir)


if __name__ == "__main__":
    main()
