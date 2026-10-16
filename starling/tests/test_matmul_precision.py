"""
Tests that STARLING never changes PyTorch's float32 matmul precision behind the
caller's back.

STARLING's model modules used to call torch.set_float32_matmul_precision("high")
when they were imported. That switched every later float32 matmul in the
caller's process to TF32, including SMACOF reconstruction, so structures rebuilt
from saved distance maps in a fresh process came out different. Model code now
runs inside starling.utilities.float32_matmul_precision, which restores the
caller's settings on exit.
"""

import json
import subprocess
import sys
from typing import Any

import pytest
import torch

from starling.utilities import MODEL_FLOAT32_MATMUL_PRECISION, float32_matmul_precision

# PyTorch >= 2.9 keeps per-backend settings next to the legacy one; older
# versions have neither of these. torch's type stubs do not declare
# torch.backends.mkldnn.matmul, hence the getattr.
MKLDNN_MATMUL: Any = getattr(torch.backends.mkldnn, "matmul", None)
HAS_PER_BACKEND_SETTINGS = hasattr(torch.backends.cuda.matmul, "fp32_precision") and hasattr(
    MKLDNN_MATMUL, "fp32_precision"
)

# Every module that used to switch the process to TF32 on import, plus the
# package and the entry points that pull them in
MODULES_TO_IMPORT = (
    "starling",
    "starling.models.attention",
    "starling.models.blocks",
    "starling.models.diffusion",
    "starling.models.normalization",
    "starling.models.transformer",
    "starling.models.vae",
    "starling.models.vae_components",
    "starling.models.vit",
    "starling.samplers",
    "starling.inference.generation",
    "starling.inference.model_loading",
    "starling.frontend.ensemble_generation",
)

# Run in a fresh interpreter, because by the time a test runs pytest has
# already imported the models. It deliberately imports nothing from STARLING
# before recording the starting state.
IMPORT_CHECK_SCRIPT = """
import importlib
import json
import sys

import torch


def state():
    return {
        "precision": torch.get_float32_matmul_precision(),
        "cuda_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


before = state()
for module_name in sys.argv[1:]:
    importlib.import_module(module_name)
print(json.dumps({"before": before, "after": state()}))
"""

# Short sequence so the generate() test stays quick
TEST_SEQUENCE = "MKVLAAGIDSSSEEKPRKRSGSAD"


def matmul_precision_state() -> dict[str, object]:
    """
    Read every float32 matmul setting a caller could observe.

    Returns
    -------
    dict
        The legacy precision and CUDA allow_tf32 flag, plus the per-backend
        settings when this PyTorch has them.
    """
    state: dict[str, object] = {
        "precision": torch.get_float32_matmul_precision(),
        "cuda_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    if HAS_PER_BACKEND_SETTINGS:
        state["cuda_fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
        state["mkldnn_fp32_precision"] = MKLDNN_MATMUL.fp32_precision
    return state


def reset_to_pytorch_defaults() -> None:
    """Put every float32 matmul setting back to what a fresh PyTorch process has."""
    torch.set_float32_matmul_precision("highest")
    if HAS_PER_BACKEND_SETTINGS:
        # "none" means "inherit the generic setting", which is how they start
        torch.backends.cuda.matmul.fp32_precision = "none"
        MKLDNN_MATMUL.fp32_precision = "none"


@pytest.fixture(autouse=True)
def pytorch_default_matmul_precision():
    """Start each test from PyTorch's defaults and leave the process there afterwards."""
    reset_to_pytorch_defaults()
    yield
    reset_to_pytorch_defaults()


def test_importing_starling_leaves_matmul_precision_unchanged():
    """Importing STARLING's models must not touch the process-wide precision."""
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_CHECK_SCRIPT, *MODULES_TO_IMPORT],
        capture_output=True,
        text=True,
        check=True,
    )
    # the JSON is the last line; anything printed while importing comes before it
    states = json.loads(completed.stdout.strip().splitlines()[-1])
    assert states["after"] == states["before"]


def test_scope_restores_pytorch_defaults_exactly():
    """
    The per-backend settings start as "none", and must not come back as "ieee".

    "none" and "ieee" give the same numerics on their own, but "none" follows
    the generic torch.backends.fp32_precision if the caller later sets it.
    """
    default_state = matmul_precision_state()
    with float32_matmul_precision(MODEL_FLOAT32_MATMUL_PRECISION):
        assert torch.get_float32_matmul_precision() == MODEL_FLOAT32_MATMUL_PRECISION
        assert torch.backends.cuda.matmul.allow_tf32
    assert matmul_precision_state() == default_state


@pytest.mark.parametrize("caller_precision", ["highest", "high", "medium"])
def test_scope_restores_caller_precision(caller_precision):
    """Whatever precision the caller chose is back in place after the block."""
    torch.set_float32_matmul_precision(caller_precision)
    caller_state = matmul_precision_state()
    with float32_matmul_precision(MODEL_FLOAT32_MATMUL_PRECISION):
        assert torch.get_float32_matmul_precision() == MODEL_FLOAT32_MATMUL_PRECISION
    assert matmul_precision_state() == caller_state


def test_scope_restores_precision_after_an_exception():
    """An exception inside the block must not leave the model precision behind."""
    caller_state = matmul_precision_state()
    with pytest.raises(ValueError), float32_matmul_precision(MODEL_FLOAT32_MATMUL_PRECISION):
        raise ValueError("raised inside the block")
    assert matmul_precision_state() == caller_state


@pytest.mark.skipif(not HAS_PER_BACKEND_SETTINGS, reason="needs the per-backend settings of PyTorch >= 2.9")
def test_scope_handles_callers_using_the_per_backend_api():
    """
    A caller who sets TF32 only through the per-backend API gets it back.

    Doing so leaves the legacy and per-backend settings disagreeing, and
    torch.get_float32_matmul_precision() then raises, so the scope cannot
    simply save and restore the legacy value.
    """
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    with pytest.raises(RuntimeError):
        torch.get_float32_matmul_precision()

    with float32_matmul_precision(MODEL_FLOAT32_MATMUL_PRECISION):
        assert torch.get_float32_matmul_precision() == MODEL_FLOAT32_MATMUL_PRECISION

    assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    assert MKLDNN_MATMUL.fp32_precision == "none"


@pytest.mark.slow
@pytest.mark.requires_weights
def test_generate_scopes_model_precision_and_leaves_callers_alone(monkeypatch):
    """
    generate() runs the models at MODEL_FLOAT32_MATMUL_PRECISION and SMACOF at
    SMACOF_FLOAT32_MATMUL_PRECISION, whatever the caller has set, and leaves
    the caller's settings as it found them.
    """
    from starling.frontend.ensemble_generation import generate
    from starling.samplers.base_sampler import BaseSampler
    from starling.structure import coordinates

    precision_seen_during: dict[str, str] = {}
    original_predict_noise = BaseSampler.predict_noise
    original_classical_mds = coordinates._classical_mds

    def recording_predict_noise(self, *args, **kwargs):
        precision_seen_during["sampling"] = torch.get_float32_matmul_precision()
        return original_predict_noise(self, *args, **kwargs)

    def recording_classical_mds(*args, **kwargs):
        precision_seen_during["reconstruction"] = torch.get_float32_matmul_precision()
        return original_classical_mds(*args, **kwargs)

    monkeypatch.setattr(BaseSampler, "predict_noise", recording_predict_noise)
    monkeypatch.setattr(coordinates, "_classical_mds", recording_classical_mds)

    # a caller precision different from both of STARLING's, so each scope shows
    torch.set_float32_matmul_precision("medium")
    caller_state = matmul_precision_state()
    ensemble = generate(
        TEST_SEQUENCE,
        conformations=2,
        return_structures=True,
        relax=False,
        return_single_ensemble=True,
        show_progress_bar=False,
        show_per_step_progress_bar=False,
    )

    assert len(ensemble) == 2
    assert matmul_precision_state() == caller_state
    assert precision_seen_during == {
        "sampling": MODEL_FLOAT32_MATMUL_PRECISION,
        "reconstruction": coordinates.SMACOF_FLOAT32_MATMUL_PRECISION,
    }
