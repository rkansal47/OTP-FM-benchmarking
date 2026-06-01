"""
Helper for saving trajectories + checkpoints from baseline eval scripts in
the MMFM-compatible .npz format used by `plot_method_comparison_pca`.

Format:
    trajectories: (n_samples, n_steps, dim) float32
    t_eval:       (n_steps,) float32 in [0, 1]
    marginal_times: (n_marginals,) int64
    method, dataset, dim, config: metadata
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_trajectory_npz(
    trajectories,  # (n_samples, n_steps, dim) ndarray or torch.Tensor
    t_eval,  # (n_steps,) ndarray or torch.Tensor in [0, 1]
    marginal_times,  # list[int]
    method: str,
    dataset: str,
    dim: int,
    output_path: Path,
    config: dict | None = None,
):
    """Save trajectory in MMFM-compatible .npz format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch

        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.detach().cpu().numpy()
        if isinstance(t_eval, torch.Tensor):
            t_eval = t_eval.detach().cpu().numpy()
    except ImportError:
        pass

    trajectories = np.asarray(trajectories, dtype=np.float32)
    t_eval = np.asarray(t_eval, dtype=np.float32)
    assert (
        trajectories.ndim == 3
    ), f"trajectories must be (n_samples, n_steps, dim); got {trajectories.shape}"
    assert (
        trajectories.shape[1] == t_eval.shape[0]
    ), f"shape mismatch: trajectories={trajectories.shape}, t_eval={t_eval.shape}"

    np.savez(
        output_path,
        trajectories=trajectories,
        t_eval=t_eval,
        marginal_times=np.asarray(marginal_times, dtype=np.int64),
        method=np.array(str(method)),
        dataset=np.array(str(dataset)),
        dim=np.int64(dim),
        config=np.array(config or {}, dtype=object),
    )


def integrate_torchdiffeq(
    model,
    source_np,
    ode_fn,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_steps: int = 101,
    method: str = "dopri5",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    device: str = "cuda",
    max_samples: int | None = 2000,
):
    """Integrate `ode_fn(t, x) -> dx/dt` over a fine time grid.

    Returns:
        traj_n_samples_n_steps_dim: ndarray (n_samples, n_steps, dim)
        t_eval: ndarray (n_steps,)
    """
    import torch
    import torchdiffeq

    model.eval()
    src = np.asarray(source_np, dtype=np.float32)
    if max_samples is not None and src.shape[0] > max_samples:
        src = src[:max_samples]
    x0 = torch.from_numpy(src).to(device)
    t_span = torch.linspace(float(t_start), float(t_end), n_steps, device=device)
    with torch.no_grad():
        traj = torchdiffeq.odeint(ode_fn, x0, t_span, method=method, atol=atol, rtol=rtol)
    traj = traj.permute(1, 0, 2).cpu().numpy()
    return traj, t_span.cpu().numpy()


def save_torch_checkpoint(state_dict_pieces: dict, output_path: Path, **extras):
    """Save a torch checkpoint with extra metadata."""
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state_dict_pieces)
    payload.update(extras)
    torch.save(payload, output_path)


def save_trajectory_and_checkpoint_torchdiffeq(
    model,
    source_np,
    ode_fn,
    out_dir: Path,
    fold_idx: int,
    holdout,
    seed: int,
    method: str,
    dataset: str,
    dim: int,
    marginal_times,
    config: dict | None = None,
    state_dict_pieces: dict | None = None,
    extras: dict | None = None,
    t_start: float = 0.0,
    t_end: float = 1.0,
    n_steps: int = 101,
    integrate_method: str = "dopri5",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    device: str = "cuda",
    max_samples: int = 2000,
):
    """One-shot helper: integrate, save trajectory.npz, save checkpoint.pt.

    Files written under ``out_dir``:
      - ``checkpoints/fold{N}_holdout{...}_seed{seed}.pt`` (if state_dict_pieces given)
      - ``trajectories/fold{N}_holdout{...}_seed{seed}_trajectories.npz``
    """
    out_dir = Path(out_dir)
    holdout_str = "_".join(map(str, holdout)) if holdout else "none"
    base = f"fold{fold_idx}_holdout{holdout_str}_seed{seed}"

    if state_dict_pieces is not None:
        save_torch_checkpoint(
            state_dict_pieces,
            out_dir / "checkpoints" / f"{base}.pt",
            **(extras or {}),
        )

    traj, t_eval = integrate_torchdiffeq(
        model,
        source_np,
        ode_fn,
        t_start=t_start,
        t_end=t_end,
        n_steps=n_steps,
        method=integrate_method,
        atol=atol,
        rtol=rtol,
        device=device,
        max_samples=max_samples,
    )
    save_trajectory_npz(
        trajectories=traj,
        t_eval=t_eval,
        marginal_times=marginal_times,
        method=method,
        dataset=dataset,
        dim=dim,
        output_path=out_dir / "trajectories" / f"{base}_trajectories.npz",
        config=config,
    )
    return traj, t_eval
