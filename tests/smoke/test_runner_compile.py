"""Smoke tests: every runner script must compile (parse + syntax-check).

This is the minimum bar we can enforce in CI without provisioning each
baseline's conda env (which is impractical — 12 different Python + JAX +
torch + lightning combinations). It catches:

- syntax errors / typos in the wrappers
- pathlib import drift
- accidental references to absolute paths from the original development tree

For a real one-iteration end-to-end test per method, see the per-method
conda envs documented in `scripts/setup_<method>.sh`.
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNERS = REPO / "runners"
SCRIPTS = REPO / "scripts"


def _runner_files() -> list[Path]:
    return sorted(RUNNERS.glob("*.py"))


@pytest.mark.parametrize("path", _runner_files(), ids=lambda p: p.name)
def test_runner_compiles(path: Path) -> None:
    """Each runner parses cleanly under the current Python."""
    py_compile.compile(str(path), doraise=True)


def test_shell_scripts_present() -> None:
    """Spot-check the scripts directory hasn't been emptied by an accidental rm."""
    assert (SCRIPTS / "run_all_timing.sh").is_file()
    assert (SCRIPTS / "setup_mmfm.sh").is_file()
    assert (SCRIPTS / "setup_3msbm.sh").is_file()


def test_runtime_patches_imports_clean() -> None:
    """The runtime-patches helper must import without side effects."""
    sys.path.insert(0, str(RUNNERS))
    try:
        import _runtime_patches  # noqa: F401
    finally:
        sys.path.remove(str(RUNNERS))


def test_runtime_patches_tqdm_shim() -> None:
    """`shim_tqdm_notebook` provides the expected attributes and is idempotent."""
    sys.path.insert(0, str(RUNNERS))
    try:
        import _runtime_patches

        _runtime_patches.shim_tqdm_notebook()
        import tqdm.notebook as nb  # type: ignore

        assert hasattr(nb, "tqdm")
        assert hasattr(nb, "trange")
        _runtime_patches.shim_tqdm_notebook()  # idempotent
    finally:
        sys.path.remove(str(RUNNERS))


def test_submodule_directories_present() -> None:
    """All 15 baselines + OTP-FM submodule placeholders must exist as directories.

    They may be empty (when run from a clone without `--recurse-submodules`),
    but the directory itself must exist — otherwise our .gitmodules is stale.
    """
    expected = {
        "OTP-FM",
        "baselines/MMFM",
        "baselines/3MSBM",
        "baselines/VGFM",
        "baselines/wl-mechanics",
        "baselines/DMSB",
        "baselines/DeepRUOT",
        "baselines/MIOFlow",
        "baselines/NLSB",
        "baselines/TrajectoryNet",
        "baselines/conditional-flow-matching",
        "baselines/iJKOnet",
        "baselines/jkonet-star",
        "baselines/metric-flow-matching",
    }
    for rel in expected:
        assert (REPO / rel).exists(), f"expected submodule directory {rel} is missing"


def test_methods_doc_present() -> None:
    assert (REPO / "methods.md").is_file()
