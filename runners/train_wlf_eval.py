#!/usr/bin/env python
"""
WLF-UOT (Wasserstein Lagrangian Flows) evaluation.

Adapts the WLF-UOT training (run_eb.py) to handle all missing experiments:
  eb5_l2o, eb100_loo, eb100_l2o, cite5_loo

Uses the `wlf` conda environment (JAX + ml_collections + flax + scanpy + ot).

Usage:
    conda run -n wlf python -u train_wlf_eval.py --experiment eb5_l2o --seed 42
    conda run -n wlf python -u train_wlf_eval.py --experiment eb100_loo --seed 42
    conda run -n wlf python -u train_wlf_eval.py --experiment eb100_l2o --seed 42
    conda run -n wlf python -u train_wlf_eval.py --experiment cite5_loo --seed 42
"""

import argparse
import functools
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["WANDB_MODE"] = "disabled"

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import ot as pot
from jax import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
WLF_DIR = Path(__file__).resolve().parents[1] / "baselines" / "wl-mechanics"

sys.path.insert(0, str(WLF_DIR))
os.chdir(str(WLF_DIR))

import eval_utils as eutils
import losses
import train_utils as tutils
from models import mlp  # noqa: F401 – registers models
from models import utils as mutils

EXPERIMENT_CONFIGS = {
    "eb5_l2o": {
        "data_name": "embrio",
        "dim": 5,
        "holdouts": [[1, 3]],
        "n_marginals_total": 5,
        "primary_metric": "w2",
        "primary_metric_space": "normalized",
        "n_iters": 100_000,
    },
    "eb100_loo": {
        "data_name": "embrio",
        "dim": 100,
        "holdouts": [[1], [2], [3]],
        "n_marginals_total": 5,
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "n_iters": 100_000,
    },
    "eb100_l2o": {
        "data_name": "embrio",
        "dim": 100,
        "holdouts": [[1, 3]],
        "n_marginals_total": 5,
        "primary_metric": "mmd",
        "primary_metric_space": "original",
        "n_iters": 100_000,
    },
    "cite5_loo": {
        "data_name": "cite",
        "dim": 5,
        "holdouts": [[1], [2]],
        "n_marginals_total": 4,
        "primary_metric": "w1",
        "primary_metric_space": "normalized",
        "n_iters": 100_000,
    },
}


def load_config_base(dim, data_name, n_train_marginals, n_iters):
    """Build a config mimicking configs/embrio/ubot.py with adjustable params."""
    import ml_collections

    config = ml_collections.ConfigDict()

    config.seed = 0
    config.loss = "ubot"
    config.interpolant = "linear"
    config.metric = "w1"
    config.lambd = 0.1

    config.data = data = ml_collections.ConfigDict()
    data.task = "OT"
    data.name = data_name
    data.dim = dim
    data.whiten = True
    data.test_id = None
    data.t_0, data.t_1 = 0.0, 1.0

    config.model_s = model_s = ml_collections.ConfigDict()
    model_s.input_dim = dim
    model_s.name = "mlp_s"
    model_s.ema_rate = 0.999
    model_s.nonlinearity = "swish"
    model_s.nf = 512
    model_s.n_layers = 2
    model_s.skip = False
    model_s.embed_time = True
    model_s.dropout = 0.0

    config.model_q = model_q = ml_collections.ConfigDict()
    model_q.input_dim = dim
    model_q.n_marginals = n_train_marginals
    model_q.name = "mlp_q"
    model_q.ema_rate = 0.999
    model_q.nonlinearity = "swish"
    model_q.nf = 512
    model_q.n_layers = 0
    model_q.skip = False
    model_q.indicator = True
    model_q.dropout = 0.0

    config.optimizer_s = optimizer_s = ml_collections.ConfigDict()
    optimizer_s.name = "adamw"
    optimizer_s.lr = 2e-4
    optimizer_s.beta1 = 0.9
    optimizer_s.eps = 1e-8
    optimizer_s.warmup = 5_000
    optimizer_s.grad_clip = 1.0

    config.optimizer_q = optimizer_q = ml_collections.ConfigDict()
    optimizer_q.name = "adamw"
    optimizer_q.lr = 2e-4
    optimizer_q.beta1 = 0.9
    optimizer_q.eps = 1e-8
    optimizer_q.warmup = 5_000
    optimizer_q.grad_clip = 1.0

    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 512
    train.n_gradient_steps = 10
    train.step_size = 1e-2
    train.n_jitted_steps = 1
    train.n_iters = n_iters
    train.save_every = n_iters + 1
    train.eval_every = n_iters // 5
    train.log_every = 1000

    config.eval = eval_cfg = ml_collections.ConfigDict()
    eval_cfg.batch_size = 128
    eval_cfg.num_samples = 500
    eval_cfg.use_ema = True

    return config


def get_data_multi_holdout(config, holdout_indices):
    """Load data and return train/held-out marginals with proper handling of multi-holdout."""
    import scanpy as sc

    if config.data.name == "embrio":
        adata = sc.read_h5ad("assets/ebdata_v3.h5ad")
        adata.obs["day"] = adata.obs["sample_labels"].cat.codes
    elif config.data.name == "cite":
        adata = sc.read_h5ad("assets/op_cite_inputs_0.h5ad")
    else:
        raise NotImplementedError(f"Unknown data: {config.data.name}")

    times = adata.obs["day"].unique()
    coords = adata.obsm["X_pca"][:, : config.data.dim]

    if config.data.whiten:
        mu = coords.mean(axis=0, keepdims=True)
        sigma = coords.std(axis=0, keepdims=True)
        coords_norm = (coords - mu) / sigma
    else:
        mu = coords.mean(axis=0, keepdims=True)
        sigma = np.max(coords.std(axis=0, keepdims=True))
        coords_norm = (coords - mu) / sigma

    def inv_scaler(_x):
        return _x * sigma + mu

    X_all = [coords_norm[adata.obs["day"] == t] for t in times]
    t_all = np.linspace(0.0, 1.0, len(X_all)).tolist()

    X_held = {i: X_all[i] for i in holdout_indices}
    t_held = {i: t_all[i] for i in holdout_indices}

    X_train = [X_all[i] for i in range(len(X_all)) if i not in holdout_indices]
    t_train = [t_all[i] for i in range(len(t_all)) if i not in holdout_indices]

    # OTP-FM reference scaler for CITE (MaxStdScaler: divide by max per-feature std)
    otpfm_scaler = None
    if config.data.name == "cite":
        max_std = float(coords.std(axis=0).max())
        mean_all = coords.mean(axis=0, keepdims=True)

        def otpfm_scaler(_x):
            return (_x - mean_all) / max_std

    return X_train, t_train, X_held, t_held, inv_scaler, otpfm_scaler


def train_and_evaluate_fold(cfg_exp, holdout_indices, seed):
    """Train WLF-UOT and evaluate on held-out marginals."""
    n_train = cfg_exp["n_marginals_total"] - len(holdout_indices)
    config = load_config_base(cfg_exp["dim"], cfg_exp["data_name"], n_train, cfg_exp["n_iters"])
    config.seed = seed

    X_train, t_train, X_held, t_held, inv_scaler, otpfm_scaler = get_data_multi_holdout(
        config, holdout_indices
    )

    key = random.PRNGKey(seed)
    key, *init_key = random.split(key, 3)

    model_s, initial_params_s = mutils.init_model_s(init_key[0], config.model_s)
    optimizer_s = tutils.get_optimizer(config.optimizer_s)
    opt_state_s = optimizer_s.init(initial_params_s)
    time_sampler, init_sampler_state = tutils.get_time_sampler(config)

    state_s = mutils.State(
        step=1,
        opt_state=opt_state_s,
        model_params=initial_params_s,
        ema_rate=config.model_s.ema_rate,
        params_ema=initial_params_s,
        sampler_state=init_sampler_state,
        key=key,
        wandbid=0,
    )

    model_q, initial_params_q = mutils.init_model_q(init_key[1], config.model_q)
    optimizer_q = tutils.get_optimizer(config.optimizer_q)
    opt_state_q = optimizer_q.init(initial_params_q)

    state_q = mutils.State(
        step=1,
        opt_state=opt_state_q,
        model_params=initial_params_q,
        ema_rate=config.model_q.ema_rate,
        params_ema=initial_params_q,
        sampler_state=init_sampler_state,
        key=key,
        wandbid=0,
    )

    loss_fn = losses.get_loss(config, model_s, model_q, time_sampler, train=True)
    step_fn = tutils.get_step_fn(config, optimizer_s, optimizer_q, loss_fn)
    step_fn = jax.pmap(functools.partial(jax.lax.scan, step_fn), axis_name="batch")

    for i in range(len(X_train)):
        X_train[i] = jnp.array(X_train[i])

    @jax.jit
    def train_iterator(key):
        keys = jax.random.split(key, len(X_train))
        batch_size = config.train.batch_size
        x_batch = jnp.zeros((batch_size, len(X_train), config.data.dim))
        t_batch = jnp.zeros((batch_size, len(X_train), 1))
        for i in range(len(X_train)):
            x_batch = x_batch.at[:, i, :].set(
                jax.random.choice(keys[i], X_train[i], (batch_size,), replace=True)
            )
            t_batch = t_batch.at[:, i, :].set(t_train[i])
        x_batch = x_batch.reshape(
            jax.local_device_count(),
            config.train.n_jitted_steps,
            batch_size // jax.local_device_count(),
            len(X_train),
            config.data.dim,
        )
        t_batch = t_batch.reshape(
            jax.local_device_count(),
            config.train.n_jitted_steps,
            batch_size // jax.local_device_count(),
            len(X_train),
            1,
        )
        return (t_batch, x_batch)

    state_s = flax_utils.replicate(state_s)
    state_q = flax_utils.replicate(state_q)

    n_iters = config.train.n_iters
    t_start = time.time()

    for step in range(1, n_iters + 1, config.train.n_jitted_steps):
        key, batch_key = random.split(key)
        batch = train_iterator(batch_key)
        key, *next_key = random.split(key, num=jax.local_device_count() + 1)
        next_key = jnp.asarray(next_key)
        (_, state_s, state_q), (total_loss, metrics) = step_fn((next_key, state_s, state_q), batch)

        if step % config.train.log_every == 0:
            loss_val = flax.jax_utils.unreplicate(total_loss).mean().item()
            logger.info(f"    step {step}/{n_iters} | loss {loss_val:.4f}")

    train_time = time.time() - t_start
    logger.info(f"    Training done in {train_time:.0f}s")

    ckpt_dir = Path(BASE_DIR) / "results" / "wlf_uot" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    holdout_str = "_".join(map(str, holdout_indices))
    _ds = "eb" if cfg_exp.get("data_name") == "embrio" else cfg_exp.get("data_name", "unknown")
    _dim = cfg_exp.get("dim") or cfg_exp.get("pca_dim")
    ckpt_path = ckpt_dir / f"{_ds}_{_dim}d_holdout{holdout_str}_seed{seed}.npz"
    np.savez(
        ckpt_path,
        params_s=flax.serialization.to_bytes(flax_utils.unreplicate(state_s).model_params),
        params_q=flax.serialization.to_bytes(flax_utils.unreplicate(state_q).model_params),
    )
    logger.info(f"    Saved checkpoint to {ckpt_path}")

    ode_generator, _ = eutils.get_generator(model_s, config)
    ode_generator = jax.jit(ode_generator)

    # Always generate from t=0 (first training marginal) for fair comparison
    X_src = X_train[0]
    t_src = t_train[0]

    results = {}
    for hi in holdout_indices:
        X_tgt = jnp.array(X_held[hi])
        t_tgt = t_held[hi]

        key, eval_key = random.split(key)
        (ode_solution, weights), _ = ode_generator(
            eval_key,
            flax_utils.unreplicate(state_s),
            (X_src, t_src, X_tgt, t_tgt),
        )

        gen_orig = np.array(inv_scaler(ode_solution))
        ref_orig = np.array(inv_scaler(X_tgt))

        # For normalized metrics: use OTP-FM's scaler for CITE, WLF's own for EB
        if otpfm_scaler is not None:
            gen_norm = np.array(otpfm_scaler(gen_orig))
            ref_norm = np.array(otpfm_scaler(ref_orig))
        else:
            gen_norm = np.array(ode_solution)
            ref_norm = np.array(X_tgt)

        w = np.array(weights) if weights is not None else None
        if w is not None:
            w = w / w.sum()
            ids = np.random.choice(len(gen_norm), min(2000, len(ref_norm)), p=w)
            gen_norm_resampled = gen_norm[ids]
            gen_orig_resampled = gen_orig[ids]
        else:
            gen_norm_resampled = gen_norm
            gen_orig_resampled = gen_orig

        n = min(len(gen_norm_resampled), len(ref_norm), 2000)
        results[hi] = {
            "gen_norm": gen_norm_resampled[:n],
            "ref_norm": ref_norm[:n],
            "gen_orig": gen_orig_resampled[:n],
            "ref_orig": ref_orig[:n],
        }

    # Save fine trajectory in normalized PCA space for downstream PCA plots.
    # Use a per-timepoint generator that saves at all marginal times.
    try:
        sys.path.insert(0, str(BASE_DIR / "runners"))
        import diffrax
        from _traj_utils import save_trajectory_npz

        # Ensure all marginal times (train + held) are present, sorted.
        all_t = sorted(list(t_train) + [t_held[hi] for hi in holdout_indices])
        n_steps_traj = 101
        fine_ts = jnp.linspace(float(all_t[0]), float(all_t[-1]), n_steps_traj)
        fine_t_min, fine_t_max = float(all_t[0]), float(all_t[-1])
        t_eval_norm = ((np.asarray(fine_ts) - fine_t_min) / (fine_t_max - fine_t_min)).astype(
            np.float32
        )

        # Replicate the vector field used in eval_utils.get_generator; rf -> vf, else grad_vf.
        loss_kind = getattr(config, "loss", "rf")
        s_fn = mutils.get_model_fn(
            model_s,
            (
                flax_utils.unreplicate(state_s).params_ema
                if config.eval.use_ema
                else flax_utils.unreplicate(state_s).model_params
            ),
            train=False,
        )
        if loss_kind == "rf":

            def vf_fn(t, y, args):
                return s_fn(t * jnp.ones((y.shape[0], 1)), y)

        else:

            def vf_fn(t, y, args):
                dsdx = jax.grad(
                    lambda _t, _x: s_fn(_t * jnp.ones((_x.shape[0], 1)), _x).sum(), argnums=1
                )
                return dsdx(t, y)

        x0 = X_train[0]
        n_traj = min(2000, x0.shape[0])
        x0 = x0[:n_traj]
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(vf_fn),
            solver=diffrax.Dopri5(),
            t0=fine_t_min,
            t1=fine_t_max,
            dt0=1e-3,
            y0=x0,
            saveat=diffrax.SaveAt(ts=fine_ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
            max_steps=200_000,
        )
        traj_steps = np.asarray(sol.ys)  # (n_steps, n_samples, dim)
        traj_for_save = traj_steps.transpose(1, 0, 2).astype(np.float32)
        traj_for_save_orig = np.asarray(
            inv_scaler(traj_for_save.reshape(-1, traj_for_save.shape[-1]))
        ).reshape(traj_for_save.shape)
        if otpfm_scaler is not None:
            traj_for_save_norm = (
                np.asarray(
                    otpfm_scaler(traj_for_save_orig.reshape(-1, traj_for_save_orig.shape[-1]))
                )
                .reshape(traj_for_save_orig.shape)
                .astype(np.float32)
            )
        else:
            traj_for_save_norm = (
                traj_for_save  # already in WLF-normalized space ≈ PCA normalized for EB
            )

        traj_dir = Path(BASE_DIR) / "results" / "wlf_uot" / "trajectories"
        _ds = "eb" if cfg_exp.get("data_name") == "embrio" else cfg_exp.get("data_name", "unknown")
        _dim = cfg_exp.get("dim") or cfg_exp.get("pca_dim")
        base = f"{_ds}_{_dim}d_holdout{holdout_str}_seed{seed}"
        save_trajectory_npz(
            trajectories=traj_for_save_norm,
            t_eval=t_eval_norm,
            marginal_times=cfg_exp.get("marginal_times", list(range(cfg_exp["n_marginals_total"]))),
            method="wlf_uot",
            dataset=_ds,
            dim=_dim,
            output_path=traj_dir / f"{base}_trajectories.npz",
            config={k: v for k, v in cfg_exp.items()},
        )
        logger.info(f"    Saved trajectory to {traj_dir / (base + '_trajectories.npz')}")
    except Exception as e:
        logger.warning(f"    Failed to save trajectory: {e}")

    return results, train_time


def compute_w1(gen, ref):
    n = min(len(gen), len(ref))
    M = pot.dist(gen[:n], ref[:n])
    return float(pot.emd2(np.ones(n) / n, np.ones(n) / n, M) ** 0.5)


def compute_w2(gen, ref, max_dim=10):
    d = min(gen.shape[1], max_dim)
    n = min(len(gen), len(ref))
    M = pot.dist(gen[:n, :d], ref[:n, :d])
    return float(pot.emd2(np.ones(n) / n, np.ones(n) / n, M) ** 0.5)


def compute_mmd(gen, ref, gamma=None):
    from scipy.spatial.distance import cdist

    n = min(len(gen), len(ref), 2000)
    g, r = gen[:n], ref[:n]
    if gamma is None:
        dists = cdist(r, r, "sqeuclidean")
        gamma = 1.0 / np.median(dists[dists > 0])
    K_rr = np.exp(-gamma * cdist(r, r, "sqeuclidean"))
    K_gg = np.exp(-gamma * cdist(g, g, "sqeuclidean"))
    K_rg = np.exp(-gamma * cdist(r, g, "sqeuclidean"))
    return float(K_rr.mean() + K_gg.mean() - 2 * K_rg.mean())


METRIC_FNS = {"w1": compute_w1, "w2": compute_w2, "mmd": compute_mmd}


def run_experiment(exp_name, seed=42, save_dir=None):
    logger.info(f"=== WLF-UOT {exp_name} seed={seed} ===")
    cfg = EXPERIMENT_CONFIGS[exp_name]
    results = {"experiment": exp_name, "seed": seed, "method": "wlf_uot", "folds": []}
    t_total_start = time.time()

    metric_fn = METRIC_FNS[cfg["primary_metric"]]
    use_orig = cfg["primary_metric_space"] == "original"

    for fold_idx, holdout in enumerate(cfg["holdouts"]):
        logger.info(f"--- Fold {fold_idx}: holdout={holdout} ---")

        fold_data, train_time = train_and_evaluate_fold(cfg, holdout, seed)

        fold_result = {"holdout": holdout, "metrics": {}, "train_sec": train_time}
        for hi in holdout:
            d = fold_data[hi]
            gen = d["gen_orig"] if use_orig else d["gen_norm"]
            ref = d["ref_orig"] if use_orig else d["ref_norm"]
            val = metric_fn(gen, ref)
            fold_result["metrics"][str(hi)] = {cfg["primary_metric"]: float(val)}
            logger.info(f"  holdout_idx={hi}: {cfg['primary_metric']}={val:.4f}")

        results["folds"].append(fold_result)

    results["total_sec"] = time.time() - t_total_start

    all_values = [
        m[cfg["primary_metric"]] for fold in results["folds"] for m in fold["metrics"].values()
    ]
    results["primary_metric"] = cfg["primary_metric"]
    results["primary_metric_space"] = cfg["primary_metric_space"]
    results["primary_value"] = float(np.mean(all_values)) if all_values else None

    logger.info(
        f"=== WLF-UOT {exp_name} DONE: {cfg['primary_metric']}="
        f"{results['primary_value']:.4f} ({results['total_sec']:.0f}s) ==="
    )

    if save_dir is None:
        save_dir = BASE_DIR / "results" / "wlf_uot" / exp_name
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
    a = ap.parse_args()
    save_dir = BASE_DIR / "results" / "wlf_uot" / a.experiment
    run_experiment(a.experiment, seed=a.seed, save_dir=save_dir)
