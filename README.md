# OTP-FM-benchmarking

Documenting code for benchmarking previous methods in [Kansal et. al., *Multimarginal flow matching with optimal transport potentials*, ICML 2026](https://arxiv.org/abs/2606.05327).
This repository 13 baseline trajectory-inference methods as git submodules + evaluation protocols on the Embryoid Body (EB) and CITE single-cell RNA sequencing datasets.
We welcome any feedback from authors or others for improving the evaluation.

## Repository layout

```text
OTP-FM-benchmarking/
├── OTP-FM/                             # submodule: model + data loaders
│   methods.md                          # per-method writeup
├── baselines/                          # 13 method submodules (pinned commits)
│   ├── 3MSBM/                          # panostheo98/3MSBM @ 6568a5e
│   ├── DMSB/                           # rkansal47/DMSB  @ 656273e (CITEseq + CPU)
│   ├── DeepRUOT/                       # zhenyiizhang/DeepRUOT @ e9b06bc
│   ├── MIOFlow/                        # KrishnaswamyLab/MIOFlow @ ec4b8ba
│   ├── MMFM/                           # Genentech/MMFM @ a6f0cbf
│   ├── NLSB/                           # take-koshizuka/NLSB @ dc44f1f
│   ├── TrajectoryNet/                  # KrishnaswamyLab/TrajectoryNet @ 810c89b
│   ├── VGFM/                           # DongyiWang-66/VGFM @ 2b429a3
│   ├── conditional-flow-matching/      # atong01/.../torchcfm @ 75835b2
│   ├── iJKOnet/                        # rkansal47/iJKOnet @ ebc0730 (ottMLP)
│   ├── jkonet-star/                    # antonioterpin/jkonet-star @ 1741c53
│   ├── metric-flow-matching/           # kksniak/metric-flow-matching @ e44e03e
│   └── wl-mechanics/                   # rkansal47/wl-mechanics @ 99133de
├── runners/                            # train_<method>_eval.py and train_<method>_timing.py
│   ├── _runtime_patches.py             # monkey-patches for DeepRUOT/MIOFlow/OT-MFM
│   ├── _traj_utils.py                  # shared trajectory I/O
│   └── train_*.py                      # 20 wrapper scripts
├── scripts/                            # setup_*.sh + run_all_timing.sh
├── configs/nlsb/                       # NLSB EB100D / CITE50D configs we authored
└── pyproject.toml                      # pixi/uv-installable harness env
```

Per-method details (citations, patches, result-cell provenance, exact reproduce CLIs) in [docs/methods.md](docs/methods.md).

## Quick start

### 1. Clone with submodules
```bash
git clone --recurse-submodules https://github.com/rkansal47/OTP-FM-benchmarking.git
cd OTP-FM-benchmarking
# (or if you already cloned without submodules:)
# git submodule update --init --recursive
```

### 2. Install the harness env (pixi)
```bash
pixi install
pixi shell    # numpy / torch / pot / torchdiffeq / sklearn / ml-collections
```

The harness env only carries the small set of dependencies needed by OTP-FM's data loaders and our evaluation helpers. Each baseline lives in its own conda env (`env_<method>`) created by the corresponding [`scripts/setup_<method>.sh`](scripts/) (Linux/CUDA only).

### 3. Put EB / CITE data in place
```bash
ls OTP-FM-benchmarking/data/
# expected:
# eb_velocity_v5.npz   # EB dataset (Tong et al. 2020)
# cite_pca50.csv       # CITE-seq dataset (NeurIPS 2022 challenge)
# cite_pca50.npz       # same as above, NPZ form used by some baselines
```

## Fully-worked example: MIOFlow on EB 5D L1O

```bash
# (a) Make sure submodules are checked out and pinned
git submodule update --init --recursive

# (b) Create the MIOFlow conda env (recipe reproduces the env used for the paper runs)
bash scripts/setup_mioflow.sh

# (c) Run the eval (one seed). Trajectories + checkpoint are written under
#     results/mioflow/eb5_loo/seed42/.
conda run -n env_mioflow python runners/train_mioflow_eval.py \
    --experiment eb5_loo \
    --seed 42 \
    --mode direct
```

The runner picks up the `_runtime_patches.shim` for `mioflow.mioflow.odeint`, so the ODE integration uses `method='rk4', step_size=0.1` — the configuration used for the paper runs.

For other methods, run the corresponding `runners/train_<method>_eval.py`; see [methods.md](methods.md) for the exact CLI per (method, experiment) pair.

## Wall-clock timing benchmark

```bash
# All 10 methods x 3 datasets (EB 5D, EB 100D, CITE 50D), 5-minute timeout each.
bash scripts/run_all_timing.sh

# Or one combination at a time:
bash scripts/run_all_timing.sh mioflow eb100
```

Results are appended to `scripts/timing_results.csv` (one row per (method, dataset)). Each row reports total wall-clock, iterations completed, time per iter, and the extrapolated full-training time used in the the paper.


## Citation

```bibtex
@inproceedings{kansal2026multimarginal,
    title={Multimarginal flow matching with optimal transport potentials},
    author={Raghav Kansal and David Crair and Nghia Nguyen and Scott Pope and Bradley Parry},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    eprint={2606.05327},
    url={https://arxiv.org/abs/2606.05327},
}
```
