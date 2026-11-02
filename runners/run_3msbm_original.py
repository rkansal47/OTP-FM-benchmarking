#!/usr/bin/env python
"""
Run 3MSBM training using the original codebase.

This script:
1. Patches 3MSBM's data loading to use the canonical OTP-FM EB data
2. Runs the original 3MSBM training with Hydra config
3. Exports trajectories in the standardized format

Usage:
    conda activate env_3msbm
    python run_3msbm_original.py --dim 100 --epochs 40

Requirements:
    - 3MSBM conda environment (see setup_3msbm.sh)
    - OTP-FM EB data at OTP-FM/data/eb_velocity_v5.npz
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Setup paths
RUNNERS_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = RUNNERS_DIR.parent
MSBM_DIR = PROJECT_ROOT / "baselines" / "3MSBM"
DATA_PATH = PROJECT_ROOT / "OTP-FM" / "data" / "eb_velocity_v5.npz"

# Add 3MSBM to path
sys.path.insert(0, str(MSBM_DIR))

# Import 3MSBM modules (after adding to path)
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


class DataLoaderX(DataLoader):
    """DataLoader with background prefetching."""

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def setup_loader(dataset, batch_size):
    """Setup data loader with infinite sampling."""
    g = torch.Generator(device="cpu")
    sampler = torch.utils.data.RandomSampler(
        dataset, replacement=True, num_samples=batch_size, generator=torch.Generator(device="cpu")
    )
    train_loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        sampler=sampler,
        generator=g,
    )
    print(f"Number of samples: {len(dataset)}")

    while True:
        yield from train_loader


class DataSampler:
    """Data sampler matching 3MSBM's interface."""

    def __init__(self, dataset, batch_size, device, ratio=0.15):
        self.num_sample = len(dataset)
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=ratio)
        self.dataloader = setup_loader(dataset[train_idx, ...], batch_size)
        self.ground_truth = dataset
        self.test_sample = dataset[val_idx, ...]
        self.batch_size = batch_size
        self.device = device

    def sample(self):
        data = next(self.dataloader)
        return data.to(self.device)


def load_eb_data(dim: int = 100, normalize: bool = True):
    """Load EB data from the canonical OTP-FM data dir."""
    print(f"Loading EB data from {DATA_PATH}")
    data = np.load(DATA_PATH)
    pcs = data["pcs"]
    labels = data["sample_labels"]

    if normalize:
        # Standardize each dimension
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        pcs = scaler.fit_transform(pcs)

    pcs = pcs[:, :dim].astype("float32")

    print(f"Loaded data: {pcs.shape}, labels: {np.unique(labels)}")
    return pcs, labels


def RNAsc_builder_patched(opt, pcs, labels):
    """Patched version of RNAsc_builder that uses our data."""
    datas = pcs[:, : opt.data_dim].astype("float32")
    timestamps = labels
    tokens = np.arange(0, opt.T)
    datasets = [datas[np.where(token == timestamps)] for token in tokens]
    dists = [DataSampler(dataset, opt.microbatch, opt.device) for dataset in datasets]
    dists_val = copy.copy(dists)
    return dists, dists_val


def patch_data_loading(pcs, labels):
    """Monkey-patch 3MSBM's data loading."""
    from dataset import RNA_seq, get_dataset

    # Store data in module for access
    RNA_seq._eb_pcs = pcs
    RNA_seq._eb_labels = labels

    def patched_builder(opt):
        return RNAsc_builder_patched(opt, RNA_seq._eb_pcs, RNA_seq._eb_labels)

    RNA_seq.RNAsc_builder = patched_builder

    # Patch get_dist
    original_get_dist = get_dataset.get_dist

    def patched_get_dist(opt):
        if opt.name in ["EB", "EB5", "RNAsc"]:
            return patched_builder(opt)
        return original_get_dist(opt)

    get_dataset.get_dist = patched_get_dist

    print("Data loading patched successfully")


def generate_trajectories(model, pcs, labels, n_samples=2000, device="cuda"):
    """Generate trajectories from trained model."""
    model.eval()

    # Get source samples from t=0
    t0_mask = labels == 0
    source_data = pcs[t0_mask].astype("float32")

    if len(source_data) > n_samples:
        idx = np.random.choice(len(source_data), n_samples, replace=False)
        source_data = source_data[idx]

    x = torch.from_numpy(source_data).float().to(device)
    v = torch.zeros_like(x)  # Initialize velocity to zero

    print(f"Generating trajectories from {x.shape[0]} samples...")

    with torch.no_grad():
        output = model.sample(x, v, direction="fwd")
        ms = output["ms"]  # (n_samples, n_steps, 2*dim)

    # Extract positions (first half of state)
    dim = x.shape[1]
    trajectories = ms[:, :, :dim].cpu().numpy()

    # Create time array
    n_steps = trajectories.shape[1]
    t_eval = np.linspace(0, 1, n_steps)

    return trajectories, t_eval


def save_trajectories(trajectories, t_eval, output_path, config):
    """Save trajectories in standardized format."""
    np.savez(
        output_path,
        trajectories=trajectories,
        t_eval=t_eval,
        marginal_times=np.array([0, 1, 2, 3, 4]),
        method="3msbm",
        dataset="eb",
        dim=config["dim"],
        config=config,
    )
    print(f"Saved trajectories to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run original 3MSBM on EB data")
    parser.add_argument("--dim", type=int, default=100, help="PCA dimension")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise sigma")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--n-samples", type=int, default=2000, help="Samples for export")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--train-times",
        nargs="+",
        type=int,
        default=[0, 2, 4],
        help="Training times (use 0,2,4 for 3 marginals)",
    )
    parser.add_argument("--nfe", type=int, default=801, help="Number of function evaluations")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "trajectories"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    pcs, labels = load_eb_data(dim=args.dim, normalize=args.normalize)

    # Patch data loading
    patch_data_loading(pcs, labels)

    # Import training modules after patching
    import multimarg_runner
    import pytorch_lightning as pl
    from dataset.get_dataset import get_dist
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    # Create config
    cfg = OmegaConf.create(
        {
            "name": "EB",
            "data_dim": args.dim,
            "n_train": 5120,
            "microbatch": args.batch_size,
            "ema": 0.999,
            "lr": args.lr,
            "l2_norm": 0,
            "T": len(args.train_times),
            "exp_f": 2000,
            "sigma": args.sigma,
            "device": device,
            "nfe": args.nfe,
        }
    )

    print("\n=== Training Configuration ===")
    print(f"Dimension: {args.dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Sigma: {args.sigma}")
    print(f"Train times: {args.train_times}")
    print(f"NFE: {args.nfe}")
    print()

    # Create model
    print("Creating 3MSBM model...")
    dists, dists_val = get_dist(cfg)
    model = multimarg_runner.MomMultiMargSBM(cfg, dists, dists_val)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir / "checkpoints_3msbm",
            filename="epoch-{epoch:03d}",
            save_top_k=-1,
            save_last=True,
            every_n_epochs=10,
        ),
        LearningRateMonitor(),
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        callbacks=callbacks,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=-1,
        check_val_every_n_epoch=1,
    )

    # Train
    print("\n=== Starting Training ===")
    trainer.fit(model)

    # Save checkpoint
    checkpoint_path = args.output_dir / "3msbm_original_final.pt"
    torch.save(
        {
            "fwd_net": model.fwd_net.state_dict(),
            "bwd_net": model.bwd_net.state_dict(),
            "fwd_ema": model.fwd_ema.state_dict(),
            "bwd_ema": model.bwd_ema.state_dict(),
            "config": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    # Generate and export trajectories
    print("\n=== Generating Trajectories ===")
    trajectories, t_eval = generate_trajectories(
        model,
        pcs,
        labels,
        n_samples=args.n_samples,
        device=device,
    )

    config = {
        "epochs": args.epochs,
        "dim": args.dim,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sigma": args.sigma,
        "normalize": args.normalize,
        "train_times": args.train_times,
        "nfe": args.nfe,
        "implementation": "original",
    }

    output_path = args.output_dir / f"3msbm_eb_dim{args.dim}.npz"
    save_trajectories(trajectories, t_eval, output_path, config)

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
