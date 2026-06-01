# Baseline methods

## How to read each entry

- **Upstream / submodule**: which git repo we pin at which commit (URL + 7-char SHA).
- **Citation**: reference and short description
- **Patches**: short list of any modifications applied on top of the upstream commit. Implementation lives either in a `rkansal47/<method>` fork (committed) or in [`runners/_runtime_patches.py`](../runners/_runtime_patches.py) (applied at wrapper-import time).
- **Results Provenance**: provenance of each result in Table 3 of our paper.
- **Reproduce**: exact CLI used. Replace `<env>` with the per-method conda env (see [scripts/setup_*.sh](../scripts/)) and seed as needed.

---

## TrajectoryNet
- **Citation**: Tong et al. [*TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics*](https://proceedings.mlr.press/v119/tong20a.html) (ICML 2020) — continuous normalizing flow with dynamic-OT regularization for trajectory inference.
- **Upstream / submodule**: [`KrishnaswamyLab/TrajectoryNet @ 810c89b`](https://github.com/KrishnaswamyLab/TrajectoryNet) (`baselines/TrajectoryNet`).
- **Patches**: none.
- **Results Provenance**: we use the authors' EB 5D configuration. EB 5D L1O is taken from the paper; EB 100D and CITE experiments were not attempted as full evaluation runs because of the prohibitively slow training time, and for the same reason their **timing** values for EB 100D, CITE 5D, and CITE 50D are extrapolated from limited-iteration runs using configurations adapted from the EB 5D config.
- **Reproduce (timing)**:
  ```bash
  bash scripts/run_all_timing.sh trajectorynet eb5
  ```

## NLSB
- **Citation**: Koshizuka and Sato. [*Neural Lagrangian Schrödinger Bridge: Diffusion Modeling for Population Dynamics*](https://openreview.net/forum?id=d3QNWD_pcFv) (ICLR 2023) — Lagrangian-regularized neural SDE for population dynamics.
- **Upstream / submodule**: [`take-koshizuka/NLSB @ dc44f1f`](https://github.com/take-koshizuka/NLSB) (`baselines/NLSB`).
- **Patches**: no source changes; new EB 100D and CITE 50D configs ship under [`configs/nlsb/`](../configs/nlsb/) (adapted from the EB 5D config).
- **Results Provenance**: we use the authors' EB 5D configuration. EB 5D L1O is from the paper (Koshizuka and Sato); EB 100D L0O and L1O are from Chen et al. (NeurIPS 2023); we ran CITE 5D L1O ourselves with a configuration adapted from EB 5D. EB 100D L2O and CITE 50D could not be completed because of the prohibitively slow training time; their timings are extrapolated from ten-minute runs using configurations adapted from the EB 5D config.
- **Reproduce (eval)**:
  ```bash
  conda run -n env_nlsb python runners/train_nlsb_eval.py --experiment cite5_loo --seed 0
  ```

## MIOFlow
- **Citation**: Huguet et al. [*Manifold Interpolating Optimal-Transport Flows for Trajectory Inference*](https://openreview.net/forum?id=ahAEhOtVif) (NeurIPS 2022) — Geodesic Autoencoder (GAGA) + neural ODE trained against marginal-Wasserstein loss rather than the maximum-likelihood loss of CNFs. This avoids the trace-of-Jacobian computation at every integration step.
- **Upstream / submodule**: [`KrishnaswamyLab/MIOFlow @ ec4b8ba`](https://github.com/KrishnaswamyLab/MIOFlow) (`baselines/MIOFlow`).
- **Patches**: applied as runtime monkey-patch via `runners/_runtime_patches.py::patch_mioflow_odeint`: forces `torchdiffeq.odeint` calls inside `mioflow.mioflow.MIOFlow.run_inference` and `train_mioflow` to use `method='rk4', options=dict(step_size=0.1)` (the upstream default `dopri5` step controller was the source of non-convergence in our paper runs).
- **Results Provenance**: the repository does not provide scRNA-specific configurations matching our benchmark, so we adapt the 5D demo configuration for all four datasets. CITE 50D L1O is from the paper (Huguet et al.); EB 100D L0O and L1O are from Chen et al. (NeurIPS 2023); we ran EB 5D L1O, EB 100D L2O, and CITE 5D L1O ourselves with GAGA encoding/decoding bypassed (i.e. trained in PCA space) — see the `--mode direct` flag of [`runners/train_mioflow_eval.py`](../runners/train_mioflow_eval.py).
- **Reproduce (eval)**:
  ```bash
  conda run -n env_mioflow python runners/train_mioflow_eval.py --experiment eb5_loo --seed 42 --mode direct
  ```

## DMSB
- **Citation**: Chen et al. [*Deep Momentum Multi-Marginal Schrödinger Bridge*](https://openreview.net/forum?id=ykvvv0gc4R) (NeurIPS 2023) — neural SDE that lifts multi-marginal Schrödinger bridges to phase space.
- **Upstream / submodule**: [`rkansal47/DMSB @ 656273e`](https://github.com/rkansal47/DMSB) (fork; branch `otp-fm-benchmarking`), based on `TianrongChen/DMSB @ fc2c9e9` (`baselines/DMSB`).
- **Patches** (committed in the fork):
  - New `CITEseq` problem type (`data.py`, `options.py`, `runner.py`, `sde.py`, `util.py`, `configs/default_CITE_config.py`).
  - CPU fallback when CUDA is unavailable.
  - `torch.cuda.amp` → `torch.amp` API update.
  - `sample_size = min(1000, pred_traj.shape[0])` guard in `metrics.py`.
- **Results Provenance**: we use the authors' EB 100D and CITE 50D configurations. EB 5D L1O is from the paper (Chen et al.); CITE 5D L1O is from Kapusniak et al. (NeurIPS 2024); we ran EB 100D L0O, L1O, L2O, and CITE 50D L1O ourselves. Chen et al. report EB 100D L0O and L1O in the z-score *normalized* space whereas Table 2 reports unnormalized MMD, so we retrain DMSB to obtain unnormalized values for these cells. EB 5D and CITE 5D timings use configurations adapted from the EB 100D and CITE 50D configs, respectively.
- **Reproduce (eval)**:
  ```bash
  conda run -n env_dmsb python runners/train_dmsb_eval.py --experiment eb100_l2o --seed 0
  ```

## DeepRUOT
- **Citation**: Zhang et al. [*Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport*](https://openreview.net/forum?id=gQlxd3Mtru) (ICLR 2025) — neural SDE jointly learning velocity, growth, and score networks under a regularized unbalanced OT objective.
- **Upstream / submodule**: [`zhenyiizhang/DeepRUOT @ e9b06bc`](https://github.com/zhenyiizhang/DeepRUOT) (`baselines/DeepRUOT`).
- **Patches**: two patches applied at different stages:
  - **Runtime**: `runners/_runtime_patches.py::shim_tqdm_notebook` shims `tqdm.notebook` to the regular `tqdm` so DeepRUOT's `from tqdm.notebook import tqdm` works from a plain CLI without ipywidgets.
  - **Setup-time**: [`scripts/setup_deepruot.sh`](../scripts/setup_deepruot.sh) sed-patches `baselines/DeepRUOT/setup.py` to drop the `pkg_resources.parse_version` setuptools-version assertion (removed in modern setuptools) before running `pip install -e`.
- **Results Provenance**: we adapt the original four-phase EB 5D training schedule for all datasets. EB 5D L1O, CITE 5D L1O, and CITE 50D L1O are from the paper; we ran the EB 100D settings ourselves. Timing values are obtained by extrapolating the per-phase iteration timing measured within 10 minutes per dataset.
- **Reproduce (eval)**:
  ```bash
  bash scripts/setup_deepruot.sh
  conda run -n env_deepruot python runners/train_deepruot_eval.py --experiment eb100_l2o --seed 42
  ```

## WLF-UOT
- **Citation**: Neklyudov et al. [*A Computational Framework for Solving Wasserstein Lagrangian Flows*](https://proceedings.mlr.press/v235/neklyudov24a.html) (ICML 2024) — Wasserstein Lagrangian Flows (unbalanced OT variant). Solves the variational trajectory-inference problem directly.
- **Upstream / submodule**: [`rkansal47/wl-mechanics @ 99133de`](https://github.com/rkansal47/wl-mechanics) (fork on `main`), based on `necludov/wl-mechanics` HEAD (`8c4d525`) (`baselines/wl-mechanics`).
- **Patches** (committed in the fork):
  - `datasets.py`: eagerly remove the held-out marginal from `X_train` before training so the test-id observations never leak into training data.
  - `train_utils.py`: deprecated `jax.tree_map` → `jax.tree.map` (JAX >= 0.4.25).
  - `configs/{cite50,embrio,embrio100}/ubot.py`: add `config.metric='w1'`, `config.lambd=0.1` as defaults so the held-out W1 evaluation reported in the paper is the configured default. `configs/embrio100/ubot.py` is a new EB 100D config (cite50-style 512×3 MLP architecture, dim=100) used for the EB 100D L0O, L1O, and L2O runs.
  - `train_and_extract_couplings.py`: standalone trainer that fits WLF-UOT+ on EB 5D and extracts couplings between consecutive marginals from the learnt potential (auxiliary tool, not in the main eval loop).
- **Results Provenance**: we use the authors' EB 5D and CITE 50D unbalanced-OT configurations. EB 5D L1O is from the paper (Neklyudov et al. 2024); CITE 5D L1O and CITE 50D L1O are from Kapusniak et al. (NeurIPS 2024); we ran the EB 100D settings ourselves with configurations adapted from those provided. For consistency with the other baselines, evaluation samples are generated starting from $t=0$. Neklyudov et al. (2024) also report values that incorporate potentials based on the *held-out* marginals; we do not include those for fairness.
- **Reproduce (eval)**:
  ```bash
  conda run -n wlf python runners/train_wlf_eval.py --experiment eb5_loo --seed 0
  ```

## JKOnet*
- **Citations**:
  - Bunne et al. [*Proximal Optimal Transport Modeling of Population Dynamics*](https://arxiv.org/abs/2106.06345) (AISTATS 2022) — JKOnet, diffusion as energy-minimizing trajectories in Wasserstein space.
  - Terpin et al. [*Learning diffusion at lightspeed*](https://openreview.net/forum?id=y10avdRFNK) (NeurIPS 2024) — JKOnet*, follow-up that improves training efficiency and performance.
- **Upstream / submodule**: [`antonioterpin/jkonet-star @ 1741c53`](https://github.com/antonioterpin/jkonet-star) (`baselines/jkonet-star`).
- **Patches**: none.
- **Results Provenance**: we use the authors' EB 5D configuration; the other datasets are run with configurations adapted from EB 5D. EB 100D L0O and L1O values are from Persiianov et al. (ICLR 2026); we ran EB 5D L1O, EB 100D L2O, CITE 5D L1O, and CITE 50D L1O ourselves.
- **Reproduce (eval)**:
  ```bash
  conda run -n env_jkonet python runners/train_jkonet_eval.py --experiment cite50_loo --seed 0
  ```

## iJKOnet
- **Citation**: Persiianov et al. [*Learning of Population Dynamics: Inverse Optimization Meets JKO Scheme*](https://openreview.net/forum?id=tVJIKd6CLF) (ICLR 2026) — adversarial inverse-JKO scheme for the JKO optimization problem.
- **Upstream / submodule**: [`rkansal47/iJKOnet @ ebc0730`](https://github.com/rkansal47/iJKOnet) (fork; branch `otp-fm-benchmarking`), based on `MuXauJl11110/iJKOnet @ 873bc0a` (`baselines/iJKOnet`).
- **Patches** (committed in the fork):
  - `models/inverse_jko.py`: replace `from ott.neural.networks.potentials import MLP as ottMLP` (removed in newer OTT) with a local Flax reimplementation. Add `isinstance(_, str)` guards on `act_fn` / `init_fn` resolution.
  - `train.py`: fallback `train_step = model.train_step` when `epochs <= 1` or `--debug` (small ergonomic fix).
  - New `scripts/prepare_data.py` (EB/CITE data in `PopulationDataset` layout) and `scripts/train_timing.sh`.
- **Results Provenance**: we use the authors' EB 5D and CITE 50D configurations. EB 100D L0O and L1O values are from Persiianov et al. (ICLR 2026); we ran EB 5D L1O, EB 100D L2O, CITE 5D L1O, and CITE 50D L1O ourselves. EB 100D and CITE 5D timings use configurations adapted from EB 5D and CITE 50D, respectively.
- **Reproduce (eval)**:
  ```bash
  conda run -n env_ijkonet python runners/train_ijkonet_eval.py --experiment eb5_loo --seed 0
  ```

## 3MSBM
- **Citation**: Theodoropoulos et al. [*Momentum Multi-Marginal Schrödinger Bridge Matching*](https://openreview.net/forum?id=C7BIQRM57T) (NeurIPS 2026) — momentum multi-marginal Schrödinger bridge in phase space with score targets from dynamic programming.
- **Upstream / submodule**: [`panostheo98/3MSBM @ 6568a5e`](https://github.com/panostheo98/3MSBM) (`baselines/3MSBM`).
- **Patches**: none.
- **Results Provenance**: only the EB 100D L2O value is reported, taken directly from Theodoropoulos et al. (NeurIPS 2026). Our own training run on EB 100D L2O following the authors' EB-specific configuration did not converge; we therefore use the paper-reported value and we additionally include the author-provided trajectories in the qualitative comparison figure.
- **Reproduce (training)**:
  ```bash
  bash scripts/setup_3msbm.sh
  conda run -n env_3msbm python runners/run_3msbm_original.py --dim 100 --epochs 40
  ```

## I-CFM, OT-CFM, [SF]²M
- **Citations**:
  - Tong et al. [*Improving and generalizing flow-based generative models with minibatch optimal transport*](https://openreview.net/forum?id=CD9Snc73AW) (TMLR 2024) — I-CFM and OT-CFM.
  - Tong et al. [*Simulation-Free Schrödinger Bridges via Score and Flow Matching*](https://arxiv.org/abs/2307.03672) (AISTATS 2023) — [SF]²M.

  I-CFM and OT-CFM train flow-matching velocity models on independently sampled or OT-aligned consecutive marginal pairs respectively; [SF]²M uses score matching to learn a Schrödinger bridge between marginals.
- **Upstream / submodule**: [`atong01/conditional-flow-matching @ 75835b2`](https://github.com/atong01/conditional-flow-matching) (`baselines/conditional-flow-matching`).
- **Patches**: none.
- **Results Provenance**: we use hyperparameters adapted from the single-cell tutorials. EB 5D L1O is from Tong et al. (TMLR 2024) / Tong et al. (AISTATS 2023); CITE 5D L1O and CITE 50D L1O are from Kapusniak et al. (NeurIPS 2024); we ran the EB 100D experiments ourselves. Trajectories are stitched together piecewise between consecutive training marginals. For OT-CFM we precompute a single full-dataset EMD coupling per consecutive time pair rather than re-solving per minibatch, for fair comparison with OTP-FM.
- **Reproduce (eval, one of {`icfm`, `otcfm`, `sf2m`})**:
  ```bash
  conda run -n env_torchcfm python runners/train_cfm_eval.py --method otcfm --experiment eb100_l2o --seed 42
  ```

## OT-MFM
- **Citation**: Kapusniak et al. [*Metric Flow Matching for Smooth Interpolations on the Data Manifold*](https://openreview.net/forum?id=fE3RqiF4Nx) (NeurIPS 2024) — Metric Flow Matching; CFM trajectories follow the data manifold via a data-induced Riemannian metric.
- **Upstream / submodule**: [`kksniak/metric-flow-matching @ e44e03e`](https://github.com/kksniak/metric-flow-matching) (`baselines/metric-flow-matching`).
- **Patches**: applied as runtime monkey-patch via `runners/_runtime_patches.py::patch_mfm_rbf_eps`: guards against zero sigma in `RBFNetwork` (`sigmas[k, :] = max(sigma_val, 1e-6)`) to avoid degeneracy when a cluster has zero variance.
- **Results Provenance**: we use the authors' EB 5D, CITE 5D, and CITE 50D LAND-metric configurations. EB 5D L1O, CITE 5D L1O, and CITE 50D L1O are from the paper (Kapusniak et al.); we ran the EB 100D experiments ourselves using a configuration adapted from the EB 5D config.
- **Reproduce (eval)**:
  ```bash
  WANDB_MODE=disabled conda run -n env_mfm python runners/train_otmfm_eval.py --experiment eb100_l2o --seed 0
  ```

## VGFM
- **Citation**: Wang et al. [*Joint Velocity-Growth Flow Matching for Single-Cell Dynamics Modeling*](https://openreview.net/forum?id=aXAkNlbnGa) (NeurIPS 2026) — jointly learns velocity and growth fields with a CFM loss and a semi-relaxed-OT-derived growth loss.
- **Upstream / submodule**: [`DongyiWang-66/VGFM @ 2b429a3`](https://github.com/DongyiWang-66/VGFM) (`baselines/VGFM`).
- **Patches**: none.
- **Results Provenance**: we use the authors' EB 5D, CITE 5D, and CITE 50D notebook configurations (both the "warm-up" and training phases). EB 5D L1O, CITE 5D L1O, and CITE 50D L1O are from the paper (Wang et al.); we ran the EB 100D experiments ourselves with a configuration adapted from the EB 5D notebook.
- **Reproduce (eval)**:
  ```bash
  conda run -n env_vgfm python runners/train_vgfm_eval.py --experiment eb100_loo --seed 0
  ```

## MMFM
- **Citation**: Rohbeck et al. [*Modeling Complex System Dynamics with Flow Matching Across Time and Conditions*](https://openreview.net/forum?id=hwnObmOTrV) (ICLR 2025) — multi-marginal flow matching with a cubic-spline interpolation as the conditional velocity regression target.
- **Upstream / submodule**: [`Genentech/MMFM @ a6f0cbf`](https://github.com/Genentech/MMFM) (`baselines/MMFM`).
- **Patches**: none.
- **Results Provenance**: we train MMFM following the provided code, using an identical model architecture to OTP-FM's for each respective configuration, over 5 training seeds on all five evaluation settings in Table 2.
- **Reproduce (eval)**:
  ```bash
  bash scripts/setup_mmfm.sh
  conda run -n env_mmfm python runners/train_mmfm_eval.py --experiment eb5_loo --seed 42
  ```
