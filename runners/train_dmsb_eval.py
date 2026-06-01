#!/usr/bin/env python
"""
DMSB (Multi-Marginal Schrödinger Bridge) evaluation.

Trains DMSB via its sb_alternate_train with reduced stages for missing experiments.
Missing: eb100_l2o (hold out marginals 1 & 3), cite50_loo (hold out 1 or 2).

Usage:
    conda run -n env_dmsb python -u train_dmsb_eval.py --experiment eb100_l2o --seed 42
    conda run -n env_dmsb python -u train_dmsb_eval.py --experiment cite50_loo --seed 42
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
from types import SimpleNamespace

import numpy as np
import ot as pot
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OTP_FM_DIR = BASE_DIR / "OTP-FM"
DMSB_DIR = Path(__file__).resolve().parents[1] / "baselines" / "DMSB"


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
        "folds": [[1], [2], [3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "problem_name": "RNAsc",
        "num_marg": 5,
        "T": 4.0,
        "interval": 400,
        "num_stage": 5,
        "num_itr": 500,
        "samp_bs": 2000,
        "train_bs_x": 256,
    },
    "eb100_l2o": {
        "dataset": "eb",
        "pca_dim": 100,
        "folds": [[1, 3]],
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "problem_name": "RNAsc",
        "num_marg": 5,
        "T": 4.0,
        "interval": 400,
        "num_stage": 5,
        "num_itr": 500,
        "samp_bs": 2000,
        "train_bs_x": 256,
    },
    "cite50_loo": {
        "dataset": "cite",
        "pca_dim": 50,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "problem_name": "CITEseq",
        "num_marg": 4,
        "T": 3.0,
        "interval": 300,
        "num_stage": 5,
        "num_itr": 500,
        "samp_bs": 2000,
        "train_bs_x": 256,
    },
    "cite5_loo": {
        "dataset": "cite",
        "pca_dim": 5,
        "folds": [[1], [2]],
        "primary_metric": "w1",
        "primary_metric_space": "original",
        "problem_name": "CITEseq",
        "num_marg": 4,
        "T": 3.0,
        "interval": 300,
        "num_stage": 5,
        "num_itr": 500,
        "samp_bs": 2000,
        "train_bs_x": 256,
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


# ── DMSB training ────────────────────────────────────────────────────────────


def build_dmsb_opt(cfg, holdout_indices, device_str, seed):
    """Build a namespace mimicking DMSB's argparse output."""
    num_marg = cfg["num_marg"]
    train_indices = [i for i in range(num_marg) if i not in holdout_indices]

    opt = SimpleNamespace(
        seed=seed,
        gpu=0,
        cpu=False,
        device=device_str,
        problem_name=cfg["problem_name"],
        T=cfg["T"],
        t0=0.01,
        interval=cfg["interval"],
        forward_net="toy",
        backward_net="toy",
        use_arange_t=True,
        train_bs_x=cfg["train_bs_x"],
        train_bs_t=cfg["interval"],
        v_sampling="langevin",
        use_corrector=True,
        snr=0.15,
        num_corrector_bdy=1,
        num_corrector_mid=0,
        use_amp=True,
        var=0.4,
        v_scale=0.01,
        reg=0.5,
        RNA_dim=cfg["pca_dim"],
        num_marg=len(train_indices),
        samp_bs=cfg["samp_bs"],
        sde_type="simple",
        num_itr=cfg["num_itr"],
        num_epoch=1,
        num_stage=cfg["num_stage"],
        num_ResNet=2,
        weight_decay=0,
        optimizer="AdamW",
        lr=2e-4,
        lr_f=2e-4,
        lr_b=2e-4,
        lr_gamma=0.999,
        lr_step=1000,
        l2_norm=0.0,
        grad_clip=None,
        noise_type="gaussian",
        data_scale=1,
        LOO=-1,
        prior_x="gaussian",
        mode="train",
        load=None,
        log_tb=False,
        snapshot_freq=0,
        ckpt_freq=0,
        metrics=["MMD"],
        model_configs=None,
        dir="dmsb_eval_tmp",
        data_dim=[cfg["pca_dim"]],
    )
    return opt, train_indices


def train_and_predict_dmsb(cfg, holdout_indices, all_marginals_scaled, device_str, seed):
    """Train DMSB and return generated samples at holdout timepoints."""
    sys.path.insert(0, str(DMSB_DIR))
    orig_dir = os.getcwd()
    os.chdir(str(DMSB_DIR))

    try:
        import data as dmsb_data
        import sde
        import policy
        from runner import Runner, freeze_policy

        opt, train_indices = build_dmsb_opt(cfg, holdout_indices, device_str, seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if not opt.cpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        os.makedirs(os.path.join("results", opt.dir, "forward"), exist_ok=True)
        os.makedirs(os.path.join("results", opt.dir, "backward"), exist_ok=True)
        os.makedirs(os.path.join("checkpoint", opt.dir), exist_ok=True)
        opt.ckpt_path = os.path.join("checkpoint", opt.dir)
        opt.eval_path = os.path.join("results", opt.dir)

        train_marginals = [all_marginals_scaled[i] for i in train_indices]
        dists = [
            dmsb_data.DataSampler(m.astype("float32"), opt.samp_bs, opt.device)
            for m in train_marginals
        ]
        opt.num_dist = len(dists)

        class PatchedRunner(Runner):
            def __init__(self, opt, dists):
                self.start_time = time.time()
                self.ts = torch.linspace(opt.t0, opt.T, opt.interval)
                self.x_dists = dists
                self.x_data = [d.ground_truth for d in dists]
                self.v_dists = {
                    i: opt.v_scale * torch.randn(opt.samp_bs, *opt.data_dim)
                    for i in range(len(dists))
                }
                from metrics import metric_build

                self.metrics = metric_build(opt)
                self.dyn = sde.build(opt, self.x_dists, self.v_dists)
                self.z_f = policy.build(opt, self.dyn, "forward")
                self.z_b = policy.build(opt, self.dyn, "backward")

                from runner import build_optimizer_ema_sched

                self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(
                    opt, self.z_f
                )
                self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(
                    opt, self.z_b
                )

                self.writer = None
                self.it_f = 0
                self.it_b = 0

            def log_tb(self, step, val, name, tag):
                pass

        # Make DataSampler have .data attribute
        for d in dists:
            if not hasattr(d, "data"):
                d.data = d.ground_truth if hasattr(d, "ground_truth") else d.samples

        runner = PatchedRunner(opt, dists)

        logger.info(f"  DMSB training: {opt.num_stage} stages × 6 substages × {opt.num_itr} iters")
        t0 = time.time()
        runner.sb_alternate_train(opt)
        train_time = time.time() - t0
        logger.info(f"  DMSB training done in {train_time:.0f}s")

        # Save checkpoint for re-evaluation without retraining
        ckpt_dir = Path(BASE_DIR) / "results" / "dmsb" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        holdout_str = "_".join(map(str, holdout_indices))
        ckpt_path = ckpt_dir / f"{cfg['problem_name']}_holdout{holdout_str}_seed{seed}.pt"
        torch.save(
            {
                "z_f": runner.z_f.state_dict(),
                "z_b": runner.z_b.state_dict(),
                "opt": vars(opt),
                "train_indices": train_indices,
                "train_time": train_time,
            },
            ckpt_path,
        )
        logger.info(f"  Saved checkpoint to {ckpt_path}")

        # Generate forward trajectory
        self_z_f = freeze_policy(runner.z_f)
        corrector = (
            (lambda x, t: runner.z_f(x, t) + runner.z_b(x, t)) if opt.use_corrector else None
        )
        ms, _, _, _, _ = runner.dyn.sample_traj(
            runner.ts,
            self_z_f,
            save_traj=True,
            corrector=corrector,
            rollout=[0, opt.num_dist - 1],
            resample=False,
            test=True,
        )
        traj = ms.detach().cpu().numpy()

        all_num_marg = cfg["num_marg"]

        predictions = {}
        for hi in holdout_indices:
            frac = hi / (all_num_marg - 1)
            t_idx = int(frac * (opt.interval - 1))
            pred = traj[:, t_idx, : cfg["pca_dim"]]
            predictions[hi] = pred

        # Trajectory in normalized PCA space, shape (n_samples, n_steps, dim)
        full_traj_pca = traj[:, :, : cfg["pca_dim"]].astype(np.float32)
        n_steps = full_traj_pca.shape[1]
        t_eval_norm = np.linspace(0.0, 1.0, n_steps).astype(np.float32)

        return predictions, train_time, full_traj_pca, t_eval_norm, str(ckpt_path)

    finally:
        os.chdir(orig_dir)
        torch.set_default_tensor_type("torch.FloatTensor")


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_w1(gen, ref):
    n = min(len(gen), len(ref), 2000)
    idx_g = np.random.choice(len(gen), n, replace=False)
    idx_r = np.random.choice(len(ref), n, replace=False)
    g, r = gen[idx_g].astype(np.float64), ref[idx_r].astype(np.float64)
    M = pot.dist(g, r, metric="euclidean")
    a, b = np.ones(n) / n, np.ones(n) / n
    return float(pot.emd2(a, b, M, numItermax=int(1e7)))


def compute_mmd(gen, ref, kernel_mul=2.0, kernel_num=5):
    """Multi-kernel MMD matching other baselines."""
    source = torch.from_numpy(gen.astype(np.float32))
    target = torch.from_numpy(ref.astype(np.float32))
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


# ── Main experiment runner ────────────────────────────────────────────────────


def run_experiment(exp_name, seed=42, device_str="cuda", save_dir=None):
    device = device_str if torch.cuda.is_available() else "cpu"
    logger.info(f"=== DMSB {exp_name} seed={seed} on {device} ===")

    cfg = EXPERIMENT_CONFIGS[exp_name]
    results = {"experiment": exp_name, "seed": seed, "method": "dmsb", "folds": []}
    t_start_total = time.time()

    for fold_idx, holdout in enumerate(cfg["folds"]):
        logger.info(f"--- Fold {fold_idx}: holdout={holdout} ---")

        data = load_data(cfg["dataset"], cfg["pca_dim"], holdout)
        all_times = data["all_times"]
        scaler = data["scaler"]

        time_to_idx = {t: i for i, t in enumerate(all_times)}
        holdout_indices = [time_to_idx[t] for t in holdout]

        all_marginals_scaled = []
        for t in all_times:
            m = data["marginals"][t][:, : cfg["pca_dim"]]
            all_marginals_scaled.append(m.astype(np.float32))

        predictions, train_time, full_traj_pca, t_eval_norm, ckpt_path_str = train_and_predict_dmsb(
            cfg, holdout_indices, all_marginals_scaled, device, seed
        )

        fold_result = {
            "holdout": holdout,
            "metrics": {},
            "train_sec": train_time,
            "checkpoint": ckpt_path_str,
        }

        for t_eval in holdout:
            t_idx = time_to_idx[t_eval]
            gen_norm = predictions[t_idx]
            ref_norm = data["marginals"][t_eval][:, : cfg["pca_dim"]]

            if cfg["primary_metric_space"] == "original" and scaler is not None:
                gen_metric = scaler.inverse_transform(gen_norm)
                ref_metric = scaler.inverse_transform(ref_norm)
            else:
                gen_metric = gen_norm
                ref_metric = ref_norm

            metrics = {}
            if cfg["primary_metric"] == "mmd":
                metrics["mmd"] = compute_mmd(gen_metric, ref_metric)
            elif cfg["primary_metric"] == "w1":
                metrics["w1"] = compute_w1(gen_metric, ref_metric)

            fold_result["metrics"][str(t_eval)] = metrics
            logger.info(f"  t={t_eval}: {metrics}")

        results["folds"].append(fold_result)

        # Save fine trajectory in normalized PCA space
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from _traj_utils import save_trajectory_npz

            traj_dir = (
                save_dir if save_dir is not None else BASE_DIR / "results" / "dmsb" / exp_name
            ) / "trajectories"
            holdout_str = "_".join(map(str, holdout))
            base = f"{exp_name}_holdout{holdout_str}_seed{seed}"
            n_traj = min(2000, full_traj_pca.shape[0])
            save_trajectory_npz(
                trajectories=full_traj_pca[:n_traj],
                t_eval=t_eval_norm,
                marginal_times=all_times,
                method="dmsb",
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
        for m in fold["metrics"].values():
            if primary in m:
                all_values.append(m[primary])
    results["primary_metric"] = primary
    results["primary_value"] = float(np.mean(all_values)) if all_values else None

    logger.info(
        f"=== DMSB {exp_name} DONE: {primary}={results['primary_value']:.4f} "
        f"({results['total_sec']:.0f}s) ==="
    )

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "dmsb" / exp_name
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"seed{seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {save_dir / f'seed{seed}.json'}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENT_CONFIGS.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    a = ap.parse_args()
    save_dir = BASE_DIR / "results" / "dmsb" / a.experiment
    run_experiment(a.experiment, seed=a.seed, device_str=a.device, save_dir=save_dir)
