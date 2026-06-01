"""Runtime monkey-patches applied by wrapper scripts before they import the
underlying baseline modules. Each patch is documented next to the upstream lines it modifies.

USAGE
-----
At the top of a wrapper (after stdlib imports, BEFORE importing the
corresponding baseline package):

    from _runtime_patches import shim_tqdm_notebook   # DeepRUOT
    shim_tqdm_notebook()
    from DeepRUOT import ...

    from _runtime_patches import patch_mioflow_odeint
    import mioflow.mioflow                            # picks up the import
    patch_mioflow_odeint()

    from _runtime_patches import patch_mfm_rbf_eps
    patch_mfm_rbf_eps()
    from mfm.geo_metrics.rbf import RBFNetwork
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "baselines"


def shim_tqdm_notebook() -> None:
    """Make `from tqdm.notebook import tqdm` resolve to the plain `tqdm.tqdm`.

    Patches `DeepRUOT/DeepRUOT/train.py:7` so DeepRUOT can run from a plain
    CLI (no Jupyter kernel) without `ipywidgets`.

    Idempotent — safe to call multiple times.
    """
    import tqdm as _tqdm

    if "tqdm.notebook" in sys.modules and getattr(
        sys.modules["tqdm.notebook"], "_otpfm_shim", False
    ):
        return

    shim = types.ModuleType("tqdm.notebook")
    shim.tqdm = _tqdm.tqdm
    shim.trange = _tqdm.trange
    shim._otpfm_shim = True
    sys.modules["tqdm.notebook"] = shim


def patch_mioflow_odeint() -> None:
    """Force MIOFlow's `odeint(...)` calls to use RK4 with `step_size=0.1`.

    Patches the two `odeint(...)` call sites in `MIOFlow/mioflow/mioflow.py`
    (`MIOFlow.run_inference` and `train_mioflow`). The MIOFlow module must
    already have been imported (we monkey-patch the module-level reference
    to `odeint` it pulled from `torchdiffeq`).

    Idempotent — safe to call multiple times.
    """
    import mioflow.mioflow as _mf

    if getattr(_mf.odeint, "_otpfm_rk4", False):
        return

    _orig_odeint = _mf.odeint

    def _patched_odeint(*args, **kwargs):
        kwargs.setdefault("method", "rk4")
        kwargs.setdefault("options", dict(step_size=0.1))
        return _orig_odeint(*args, **kwargs)

    _patched_odeint._otpfm_rk4 = True  # type: ignore[attr-defined]
    _mf.odeint = _patched_odeint


def patch_mfm_rbf_eps(eps: float = 1e-6) -> None:
    """Clamp the RBF-bandwidth sigma to be at least `eps` in OT-MFM.

    Patches `mfm/geo_metrics/rbf.py` (RBFNetwork init) so that degenerate
    clusters (single point, zero variance) don't produce a zero sigma and
    blow up the metric.

    Rather than monkey-patch the deep call site, we sed the file in place
    on first call.
    """
    rbf_file = BASELINES_DIR / "metric-flow-matching" / "mfm" / "geo_metrics" / "rbf.py"
    if not rbf_file.exists():
        raise FileNotFoundError(
            f"Cannot apply OT-MFM RBF patch — file not found: {rbf_file}.\n"
            f"Make sure git submodules are initialized: "
            f"`git submodule update --init --recursive`"
        )

    text = rbf_file.read_text()
    if "_OTPFM_RBF_EPS_PATCH" in text:
        return  # already patched

    old = (
        "                sigmas[k, :] = np.sqrt(\n"
        "                    variance.sum() if self.image_data else variance.mean()\n"
        "                )"
    )
    new = (
        f"                sigma_val = np.sqrt(  # _OTPFM_RBF_EPS_PATCH\n"
        f"                    variance.sum() if self.image_data else variance.mean()\n"
        f"                )\n"
        f"                sigmas[k, :] = max(sigma_val, {eps})"
    )
    if old not in text:
        raise RuntimeError(
            "OT-MFM RBF patch could not locate the original `sigmas[k, :] = np.sqrt(...)` "
            "block. The submodule may have moved past the pinned commit "
            "e44e03e. Re-run `git -C baselines/metric-flow-matching checkout e44e03e`."
        )
    rbf_file.write_text(text.replace(old, new))
