"""
Training timing script for JKOnet* (Terpin et al., "Learning Diffusion at Lightspeed", NeurIPS 2024).

JKOnet* learns energy functionals for Wasserstein gradient flows using the JKO scheme.
It supports both neural network and closed-form linear parametrizations.

The paper uses `jkonet-star-time-potential` for single-cell RNA experiments (100 epochs).
The linear variants (`jkonet-star-linear-potential`) solve in closed form in 1 epoch.

Data must be preprocessed first (couplings + GMM densities) — see prepare_data().

Usage:
    conda run -n env_jkonet python train_jkonet_timing.py --dataset eb5 --solver jkonet-star-time-potential --epochs 100
    conda run -n env_jkonet python train_jkonet_timing.py --dataset eb100 --solver jkonet-star-time-potential --epochs 100
    conda run -n env_jkonet python train_jkonet_timing.py --dataset cite50 --solver jkonet-star-time-potential --epochs 100

    # Linear (closed-form) variant — single epoch:
    conda run -n env_jkonet python train_jkonet_timing.py --dataset eb5 --solver jkonet-star-linear-potential --epochs 1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

JKONET_DIR = Path(__file__).resolve().parents[1] / "baselines" / "jkonet-star"

DATASET_MAP = {
    "eb5": "EB_5D",
    "eb100": "EB_100D",
    "cite50": "CITE_50D",
}


def prepare_data(dataset_key: str, force: bool = False) -> str:
    """
    Prepare data.npy + sample_labels.npy from raw sources,
    then run JKOnet*'s data_generator.py for couplings/GMM/splits.

    Returns the JKOnet* dataset name (folder under jkonet-star/data/).
    """
    from sklearn.preprocessing import StandardScaler

    jkonet_name = DATASET_MAP[dataset_key]
    data_dir = JKONET_DIR / "data" / jkonet_name

    if (data_dir / "train_data.npy").exists() and not force:
        print(f"[prep] Data already exists at {data_dir}, skipping.")
        return jkonet_name

    data_dir.mkdir(parents=True, exist_ok=True)
    base_dir = JKONET_DIR.parents[1] / "OTP-FM"  # canonical OTP-FM data dir lives here

    if dataset_key in ("eb5", "eb100"):
        eb_path = base_dir / "data" / "eb_velocity_v5.npz"
        raw = np.load(eb_path, allow_pickle=True)
        pcs = raw["pcs"]
        labels = raw["sample_labels"]
        scaler = StandardScaler()
        scaler.fit(pcs)
        pcs = scaler.transform(pcs)
        dim = 5 if dataset_key == "eb5" else 100
        data = pcs[:, :dim]
    elif dataset_key == "cite50":
        import pandas as pd

        cite_path = base_dir / "OTP-FM" / "data" / "cite_pca50.csv"
        df = pd.read_csv(cite_path)
        labels = df["samples"].values
        feats = df[[c for c in df.columns if c != "samples"]].values
        scaler = StandardScaler()
        scaler.fit(feats)
        data = scaler.transform(feats)
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    np.save(data_dir / "data.npy", data)
    np.save(data_dir / "sample_labels.npy", labels)
    print(f"[prep] Saved raw data: {data.shape} to {data_dir}")

    import subprocess

    cmd = [
        sys.executable,
        str(JKONET_DIR / "data_generator.py"),
        "--load-from-file",
        jkonet_name,
        "--test-ratio",
        "0.4",
        "--split-population",
        "--n-gmm-components",
        "10",
    ]
    print(f"[prep] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(JKONET_DIR), check=True)
    return jkonet_name


def run_training(
    jkonet_name: str,
    solver: str,
    epochs: int,
    batch_size: int,
    seed: int,
    eval_freq: int,
) -> dict:
    """
    Run JKOnet* training and return timing results.
    Imports are done inside to ensure correct cwd-relative paths.
    """
    os.chdir(str(JKONET_DIR))
    sys.path.insert(0, str(JKONET_DIR))

    import jax
    import yaml
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from models import EnumMethod, get_model
    from dataset import PopulationEvalDataset

    # Resolve solver enum
    solver_enum = EnumMethod(solver)

    key = jax.random.PRNGKey(seed)

    config = yaml.safe_load(open("config.yaml"))
    jkonet_config = yaml.safe_load(open("config-jkonet-extra.yaml"))
    config.update(jkonet_config)

    config["train"]["epochs"] = epochs
    config["train"]["batch_size"] = batch_size
    config["train"]["eval_freq"] = eval_freq
    config["train"]["save_locally"] = False

    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    dataset_eval = PopulationEvalDataset(
        key,
        jkonet_name,
        solver,
        config["metrics"]["wasserstein_error"],
        "test_data",
    )

    model = get_model(solver_enum, config, dataset_eval.data_dim, dataset_eval.dt)
    state = model.create_state(key)
    dataset_train = model.load_dataset(jkonet_name)

    import torch

    torch.manual_seed(seed)
    bs = batch_size if batch_size > 0 else len(dataset_train)
    loader_train = DataLoader(
        dataset_train,
        batch_size=bs,
        shuffle=True,
        collate_fn=numpy_collate,
    )

    actual_epochs = config["train"]["epochs"]
    print(f"\nTraining {solver} on {jkonet_name}")
    print(f"  data_dim={dataset_eval.data_dim}, T={dataset_eval.T}")
    print(f"  epochs={actual_epochs}, batch_size={bs}, batches/epoch={len(loader_train)}")

    train_step = model.train_step
    if actual_epochs > 1:
        train_step = jax.jit(model.train_step)

    losses = []
    epoch_times = []
    total_iters = 0

    t_total_start = time.perf_counter()

    progress_bar = tqdm(range(1, actual_epochs + 1))
    for epoch in progress_bar:
        epoch_loss = 0.0
        t_epoch_start = time.perf_counter()

        for sample in loader_train:
            step_loss, state = train_step(state, sample)
            epoch_loss += float(step_loss)
            total_iters += 1

        t_epoch_end = time.perf_counter()
        epoch_time = t_epoch_end - t_epoch_start
        epoch_times.append(epoch_time)
        avg_loss = epoch_loss / len(loader_train)
        losses.append(avg_loss)

        progress_bar.desc = f"Epoch {epoch} | Loss: {avg_loss:.4f} | {epoch_time:.2f}s"

    total_time = time.perf_counter() - t_total_start

    return {
        "solver": solver,
        "dataset": jkonet_name,
        "data_dim": dataset_eval.data_dim,
        "n_timepoints": dataset_eval.T + 1,
        "total_time_s": total_time,
        "total_epochs": actual_epochs,
        "total_iters": total_iters,
        "time_per_epoch_s": total_time / actual_epochs if actual_epochs else 0,
        "time_per_iter_ms": total_time / total_iters * 1000 if total_iters else 0,
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
        "epoch_times": epoch_times,
    }


def main():
    parser = argparse.ArgumentParser(description="JKOnet* training for timing benchmarks")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["eb5", "eb100", "cite50"],
        help="Dataset: eb5 (EB 5D), eb100 (EB 100D), cite50 (CITE-seq 50D)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="jkonet-star-time-potential",
        choices=[
            "jkonet-star",
            "jkonet-star-potential",
            "jkonet-star-potential-internal",
            "jkonet-star-time-potential",
            "jkonet-star-linear",
            "jkonet-star-linear-potential",
            "jkonet-star-linear-potential-internal",
        ],
        help="JKOnet* solver variant (default: jkonet-star-time-potential)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Eval frequency in epochs (set high to skip eval for timing)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-data", action="store_true", help="Force re-preparation of data")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)

    jkonet_name = prepare_data(args.dataset, force=args.prepare_data)

    results = run_training(
        jkonet_name=jkonet_name,
        solver=args.solver,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_freq=args.eval_freq,
    )

    print("\n=== Timing Results ===")
    print(f"Solver:           {results['solver']}")
    print(
        f"Dataset:          {results['dataset']} ({results['data_dim']}D, {results['n_timepoints']} timepoints)"
    )
    print(f"Total time:       {results['total_time_s']:.2f} s")
    print(f"Total epochs:     {results['total_epochs']}")
    print(f"Total iterations: {results['total_iters']}")
    print(f"Time per epoch:   {results['time_per_epoch_s']:.3f} s")
    print(f"Time per iter:    {results['time_per_iter_ms']:.2f} ms")
    print(f"Final loss:       {results['final_loss']:.6f}")

    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / "outputs" / "jkonet_star"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_path = args.output_dir / f"jkonet_star_{args.solver}_{args.dataset}.json"
    save_data = {k: v for k, v in results.items()}
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    main()
