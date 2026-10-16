"""
Regression tests for the CUDA review fixes.

Each test pins down one fix: SMACOF's fixed matmul precision and input checks,
model loading on the CPU, inference-mode encoding, float32 ionic strengths,
seeded and relaxation-independent sampling, protection against reused
CUDA-graph output buffers, chunked VAE decoding, deterministic constraint
guidance, and smaller device-handling fixes. Tests that need CUDA or the
released weights are marked and skipped without them.
"""

import json
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from starling import configs
from starling.data.distributions import DiagonalGaussianDistribution
from starling.inference import generation
from starling.inference.constraints import RgConstraint
from starling.models.diffusion import DiffusionModel
from starling.models.vae import VAE
from starling.samplers.ddim_sampler import DDIMSampler
from starling.samplers.ddpm_sampler import DDPMSampler
from starling.samplers.plms_sampler import PLMS_MIN_STEPS, PLMSSampler
from starling.structure.coordinates import distance_matrix_to_3d_structure_torch_mds
from starling.structure.ensemble import Ensemble
from starling.tests.test_samplers import diffusion, unused_encoder
from starling.utilities import deterministic_cudnn

# Ideal C-alpha virtual bond length used to build test chains, in Angstroms
BOND_LENGTH_A = 3.8

# Sequences for the tests that run the released model; long enough that the
# sampler, SMACOF and relaxation all do real work, short enough to stay quick
FIRST_SEQUENCE = "MKVLAAGIDSSSEEKPRKRSGSAD"
SECOND_SEQUENCE = "GSPEDKRRAQWLLEEGSSKPTRLA"

# A decoder that maps a (B, 1, 24, 24) latent to a (B, 1, 12, 12) distance map
# with a fixed random linear layer, in float64 so that chunked and whole-batch
# results can be compared tightly (float32 accumulation order differs by shape)
TOY_MAP_SIZE = 12


def random_chain_maps(n_frames: int, n_residues: int, seed: int) -> np.ndarray:
    """
    Build noisy distance maps of random-walk chains.

    Parameters
    ----------
    n_frames : int
        Number of maps.
    n_residues : int
        Number of beads per chain.
    seed : int
        Seed for numpy's generator.

    Returns
    -------
    np.ndarray
        float32 maps in Angstroms, shape (n_frames, n_residues, n_residues),
        symmetric with a zero diagonal. The noise makes them non-Euclidean, so
        SMACOF has real work to do.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n_frames, n_residues, 3))
    steps *= BOND_LENGTH_A / np.linalg.norm(steps, axis=-1, keepdims=True)
    coordinates = np.cumsum(steps, axis=1)
    maps = np.linalg.norm(coordinates[:, :, None] - coordinates[:, None], axis=-1)
    noise = rng.normal(scale=1.0, size=maps.shape)
    maps = np.abs(maps + (noise + noise.transpose(0, 2, 1)) / 2)
    for frame in maps:
        np.fill_diagonal(frame, 0.0)
    return maps.astype(np.float32)


def identity_decoder() -> VAE:
    """A stand-in VAE whose decode() returns the latents unchanged."""
    return cast(VAE, SimpleNamespace(decode=lambda latents: latents))


# ---------------------------------------------------------------------------
# SMACOF
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 only exists on CUDA")
def test_smacof_runs_at_full_precision_whatever_the_caller_sets():
    """Reconstruction must not change when the caller turns TF32 on."""
    maps = random_chain_maps(n_frames=24, n_residues=60, seed=1)
    previous = torch.get_float32_matmul_precision()
    try:
        results = {}
        for caller_precision in ("highest", "high"):
            torch.set_float32_matmul_precision(caller_precision)
            results[caller_precision], _ = distance_matrix_to_3d_structure_torch_mds(
                maps, device="cuda", progress_bar=False
            )
            # and the caller's own setting is left as it was
            assert torch.get_float32_matmul_precision() == caller_precision
    finally:
        torch.set_float32_matmul_precision(previous)
    np.testing.assert_array_equal(results["high"], results["highest"])


def test_smacof_rejects_non_finite_maps_and_names_them():
    """A NaN map used to crash CUDA's eigensolver for the whole batch and give NaN coordinates on CPU."""
    maps = random_chain_maps(n_frames=4, n_residues=10, seed=2)
    maps[2, 3, 5] = np.nan
    maps[3, 1, 1] = np.inf
    with pytest.raises(ValueError, match=r"2 of 4 .*frame indices: \[2, 3\]"):
        distance_matrix_to_3d_structure_torch_mds(maps, device="cpu", progress_bar=False)


def test_smacof_picks_a_device_when_none_is_given():
    """The default used to be 'cuda', which failed on CPU-only and MPS machines."""
    maps = random_chain_maps(n_frames=2, n_residues=8, seed=3)
    coordinates, stress = distance_matrix_to_3d_structure_torch_mds(maps, progress_bar=False)
    assert coordinates.shape == (2, 8, 3)
    assert np.isfinite(coordinates).all() and np.isfinite(stress).all()


# ---------------------------------------------------------------------------
# Model loading and encoders
# ---------------------------------------------------------------------------

# Run in a fresh interpreter, so nothing else in the test session has touched CUDA
CPU_LOAD_CHECK_SCRIPT = """
import json

import torch
from starling.inference.generation import model_manager

encoder_model, diffusion_model = model_manager.get_models(device="cpu")
devices = {str(parameter.device) for model in (encoder_model, diffusion_model) for parameter in model.parameters()}
print(json.dumps({
    "cuda_initialized": torch.cuda.is_initialized(),
    "devices": sorted(devices),
    "inner_vae_dropped": diffusion_model.distance_map_encoder is None,
}))
"""


@pytest.mark.slow
@pytest.mark.requires_weights
def test_loading_on_the_cpu_never_touches_the_gpu():
    """The training-only inner VAE used to be restored onto cuda:0 whatever device was asked for."""
    completed = subprocess.run(
        [sys.executable, "-c", CPU_LOAD_CHECK_SCRIPT], capture_output=True, text=True, check=True
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {"cuda_initialized": False, "devices": ["cpu"], "inner_vae_dropped": True}


def test_ensemble_encoder_builds_no_autograd_graph():
    """Without inference mode every batch built a graph through the encoder, several times the memory."""
    grad_enabled_during_encode = []

    def encode(maps):
        grad_enabled_during_encode.append(torch.is_grad_enabled())
        return SimpleNamespace(mode=lambda: maps[:, :, :24, :24].clone())

    manager = SimpleNamespace(get_models=lambda **kwargs: (SimpleNamespace(encode=encode), None))
    latents = generation.ensemble_encoder_backend(np.zeros((3, 10, 10)), "cpu", 2, model_manager=manager)
    assert latents.shape == (3, 24, 24)
    assert grad_enabled_during_encode == [False, False]


def test_sequence_encoder_returns_float32_ionic_strength_and_compact_embeddings():
    """
    A numpy float64 ionic strength used to crash the float32 model, and each
    stored embedding was a view that kept (and saved) the whole padded batch.
    """
    ionic_strength_dtypes = []

    def sequence2labels(sequences, sequence_mask, ionic_strength):
        ionic_strength_dtypes.append(ionic_strength.dtype)
        return torch.randn(sequences.shape[0], sequences.shape[1], 16)

    manager = SimpleNamespace(get_models=lambda **kwargs: (None, SimpleNamespace(sequence2labels=sequence2labels)))
    embeddings = generation.sequence_encoder_backend(
        {"long": "MKVLAAGIDS" * 4, "short": "MKVL"},
        device="cpu",
        batch_size=2,
        ionic_strength=np.float64(150.0),
        aggregate=False,
        model_manager=manager,
    )
    assert ionic_strength_dtypes == [torch.float32]
    assert embeddings is not None
    for embedding in embeddings.values():
        assert embedding.untyped_storage().nbytes() == embedding.numel() * embedding.element_size()


@pytest.mark.parametrize("ionic_strength", [150, 150.0, np.float64(150.0), np.int64(150)])
def test_sampler_ionic_strength_is_always_float32(ionic_strength):
    """Any numeric type must reach the model as float32."""
    sampler = DDIMSampler(diffusion(), unused_encoder(), 30, ionic_strength=ionic_strength)
    assert sampler.ionic_strength.dtype == torch.float32
    assert sampler.ionic_strength.item() == 150.0


# ---------------------------------------------------------------------------
# Seeds and random numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampler_class, arguments",
    [(DDIMSampler, {"n_steps": 30, "ddim_eta": 1.0}), (DDPMSampler, {"n_steps": None})],
)
def test_samplers_draw_all_noise_from_their_generator(sampler_class, arguments):
    """With a generator, the same seed gives the same samples and the global generator is untouched."""

    def sample(seed):
        sampler = sampler_class(
            diffusion(), identity_decoder(), generator=torch.Generator().manual_seed(seed), **arguments
        )
        return sampler.sample(4, "MKVL", show_per_step_progress_bar=False)

    global_state = torch.random.get_rng_state()
    first, repeat, other_seed = sample(7), sample(7), sample(8)
    assert torch.equal(first, repeat)
    assert not torch.equal(first, other_seed)
    assert torch.equal(global_state, torch.random.get_rng_state())


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**64, "7"])
def test_generate_rejects_invalid_seeds(seed):
    """Seeds are checked before any model is loaded."""
    from starling.frontend.ensemble_generation import generate

    with pytest.raises(ValueError, match="seed"):
        generate("MKVL", seed=seed)


def test_cli_forwards_seed(monkeypatch, tmp_path):
    """starling --seed reaches generate()."""
    from starling.scripts import starling_main_cli as cli

    seen = {}
    monkeypatch.setattr(cli, "generate", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["starling", "MKVL", "--seed", "5", "-o", str(tmp_path)])
    cli.main()
    assert seen["seed"] == 5


def _maps_for_two_sequences(**kwargs):
    """Generate a small ensemble for each test sequence and return their distance maps."""
    from starling.frontend.ensemble_generation import generate

    ensembles = generate(
        {"first": FIRST_SEQUENCE, "second": SECOND_SEQUENCE},
        conformations=4,
        batch_size=4,
        show_progress_bar=False,
        show_per_step_progress_bar=False,
        **kwargs,
    )
    return {name: ensemble.distance_maps() for name, ensemble in ensembles.items()}


@pytest.mark.slow
@pytest.mark.requires_weights
def test_relaxation_never_changes_the_distance_maps():
    """
    Relaxation used to draw from torch's global generator, so turning it on
    changed the maps sampled for every later sequence under a fixed seed.
    """
    torch.manual_seed(0)
    without_structures = _maps_for_two_sequences(return_structures=False)
    torch.manual_seed(0)
    with_relaxation = _maps_for_two_sequences(return_structures=True, relax=True)
    for name in without_structures:
        np.testing.assert_array_equal(without_structures[name], with_relaxation[name])


@pytest.mark.slow
@pytest.mark.requires_weights
def test_seeded_generate_is_reproducible_and_leaves_global_random_state_alone():
    """generate(seed=...) repeats exactly and does not advance torch's global generators."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    first = _maps_for_two_sequences(seed=11, return_structures=True, relax=True)
    repeat = _maps_for_two_sequences(seed=11, return_structures=True, relax=True)
    for name in first:
        np.testing.assert_array_equal(first[name], repeat[name])
    assert torch.equal(cpu_state, torch.random.get_rng_state())
    if torch.cuda.is_available():
        assert all(torch.equal(a, b) for a, b in zip(cuda_states, torch.cuda.get_rng_state_all()))


# ---------------------------------------------------------------------------
# Reused output buffers (torch.compile with CUDA graphs) and chunked decoding
# ---------------------------------------------------------------------------


def test_predict_noise_never_returns_the_models_own_output_buffer():
    """
    Under torch.compile(mode="reduce-overhead") the model writes every output
    into the same CUDA-graph buffer. PLMS keeps earlier predictions, which the
    next call silently overwrote.
    """
    output_buffer = torch.zeros(2, 1, 24, 24)

    def model_reusing_its_output_buffer(latents, timesteps, context, attention_mask):
        output_buffer.copy_(latents * 0.5)
        return output_buffer

    stub = diffusion()
    stub.model = model_reusing_its_output_buffer  # type: ignore[misc]
    sampler = PLMSSampler(stub, unused_encoder(), PLMS_MIN_STEPS)
    first = sampler.predict_noise(torch.ones(2, 1, 24, 24), 500, torch.zeros(1), torch.zeros(1))
    sampler.predict_noise(torch.full((2, 1, 24, 24), 3.0), 480, torch.zeros(1), torch.zeros(1))
    assert torch.equal(first, torch.full_like(first, 0.5))


def test_chunked_decoding_matches_whole_batch_even_with_a_reused_output_buffer(monkeypatch):
    """Decoding in chunks must give the whole-batch result, including from a CUDA-graph decoder."""
    output_buffers: dict[torch.Size, torch.Tensor] = {}

    def decode_reusing_its_output_buffer(latents):
        buffer = output_buffers.setdefault(latents.shape, torch.empty(latents.shape))
        buffer.copy_(latents * 3.0)
        return buffer

    decoder = cast(VAE, SimpleNamespace(decode=decode_reusing_its_output_buffer))
    sampler = DDIMSampler(diffusion(), decoder, 30)
    latents = torch.randn(10, 1, 24, 24)
    expected = latents / sampler.latent_space_scaling_factor * 3.0
    monkeypatch.setattr(configs, "VAE_DECODE_CHUNK_SIZE", 3)
    torch.testing.assert_close(sampler.decode(latents), expected, rtol=0, atol=0)


class ToyDecoder(torch.nn.Module):
    """Linear latent-to-map decoder for exact, cheap constraint gradients."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(4)
        self.weights = torch.nn.Parameter(
            torch.randn(24 * 24, TOY_MAP_SIZE**2, generator=generator, dtype=torch.float64) * 0.1
        )
        self.cudnn_deterministic_seen: list[bool] = []

    @property
    def device(self) -> torch.device:
        return self.weights.device

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.cudnn_deterministic_seen.append(torch.backends.cudnn.deterministic)
        maps = torch.nn.functional.softplus(latents.flatten(start_dim=1) @ self.weights) * 10.0
        return maps.reshape(-1, 1, TOY_MAP_SIZE, TOY_MAP_SIZE)


def test_constraint_guidance_does_not_depend_on_chunking_and_is_deterministic(monkeypatch):
    """Chunked guidance computes the same per-conformer losses and gradients, under deterministic cuDNN."""
    decoder = ToyDecoder()
    constraint = RgConstraint(target=20.0, verbose=False)
    constraint.initialize(cast(VAE, decoder), torch.tensor(1.5), 1000, TOY_MAP_SIZE)
    latents = torch.randn(7, 1, 24, 24, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    caller_setting = torch.backends.cudnn.deterministic

    results = {}
    for chunk_size in (1, 3, 10**6):
        monkeypatch.setattr(configs, "VAE_DECODE_CHUNK_SIZE", chunk_size)
        results[chunk_size] = constraint._per_conformer_loss_and_gradient(latents, 400, 1000)

    loss_whole, gradient_whole = results[10**6]
    for chunk_size in (1, 3):
        loss, gradient = results[chunk_size]
        torch.testing.assert_close(loss, loss_whole, rtol=1e-12, atol=0)
        torch.testing.assert_close(gradient, gradient_whole, rtol=1e-9, atol=1e-15)
    assert all(decoder.cudnn_deterministic_seen)
    assert torch.backends.cudnn.deterministic == caller_setting


def test_deterministic_cudnn_restores_the_callers_setting():
    """The setting is only changed inside the block, even when it raises."""
    caller_setting = torch.backends.cudnn.deterministic
    with pytest.raises(ValueError), deterministic_cudnn():
        assert torch.backends.cudnn.deterministic
        raise ValueError("raised inside the block")
    assert torch.backends.cudnn.deterministic == caller_setting


# ---------------------------------------------------------------------------
# Smaller device-handling fixes
# ---------------------------------------------------------------------------


def test_relax_ensemble_builds_structures_on_the_requested_device(monkeypatch):
    """device and batch_size used to be ignored when relax_ensemble built the MDS structures."""
    from starling.minimizer.relax import relax_ensemble

    class StopAfterBuild(Exception):
        pass

    build_arguments = {}

    def record_build(self, **kwargs):
        build_arguments.update(kwargs)
        raise StopAfterBuild

    monkeypatch.setattr(Ensemble, "build_ensemble_trajectory", record_build)
    ensemble = Ensemble(random_chain_maps(n_frames=2, n_residues=12, seed=6), "ACDEFGHIKLMN")
    with pytest.raises(StopAfterBuild):
        relax_ensemble(ensemble, device="cpu", batch_size=7, progress_bar=False)
    assert build_arguments == {"progress_bar": False, "device": "cpu", "batch_size": 7}


def test_deterministic_posterior_kl_and_nll_match_the_means_device_shape_and_dtype():
    """In deterministic mode they used to return a one-element CPU tensor."""
    parameters = torch.randn(3, 2, 24, 24, dtype=torch.float64)
    posterior = DiagonalGaussianDistribution(parameters, deterministic=True)
    for value in (posterior.kl(), posterior.nll(torch.zeros(3, 1, 24, 24, dtype=torch.float64))):
        assert value.shape == (3,)
        assert value.dtype == torch.float64
        assert value.device == posterior.mean.device
        assert torch.all(value == 0)


def test_training_latents_are_encoded_in_float32_under_autocast():
    """bf16 mixed precision used to produce bf16 latents (and bf16 noise) for diffusion training."""
    convolution = torch.nn.Conv2d(1, 1, kernel_size=16, stride=16)
    encoder = SimpleNamespace(encode=lambda maps: SimpleNamespace(mode=lambda: convolution(maps)))
    model = SimpleNamespace(distance_map_encoder=encoder)
    maps = torch.rand(2, 1, 384, 384)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert convolution(maps).dtype == torch.bfloat16
        latents = DiffusionModel._encode_distance_maps(cast(DiffusionModel, model), maps)
    assert latents.dtype == torch.float32


def test_index_training_sample_is_reproducible_and_leaves_random_alone():
    """sample_vectors used to reseed the global random module and trim with an unseeded permutation."""
    from starling.search.builder import IndexBuilder

    shards = {"a": torch.arange(200.0).reshape(50, 4), "b": torch.arange(200.0, 400.0).reshape(50, 4)}
    builder = SimpleNamespace(
        files=["a", "b"],
        counts=[50, 50],
        total=100,
        _load_features=lambda path: shards[path],
        _log=lambda message: None,
    )
    global_state = random.getstate()
    first = IndexBuilder.sample_vectors(builder, sample_size=21, seed=3)
    repeat = IndexBuilder.sample_vectors(builder, sample_size=21, seed=3)
    assert torch.equal(first, repeat)
    assert random.getstate() == global_state


def test_save_index_raises_conversion_errors_and_skips_cpu_indexes(monkeypatch, tmp_path):
    """A failed GPU-to-CPU copy used to be swallowed, and CPU indexes were deep-copied."""
    faiss = pytest.importorskip("faiss")
    from starling.search import builder as builder_module

    calls = []

    def failing_index_gpu_to_cpu(index):
        calls.append(index)
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(builder_module.faiss, "index_gpu_to_cpu", failing_index_gpu_to_cpu, raising=False)
    index = faiss.IndexFlatL2(4)
    quiet_builder = SimpleNamespace(verbose=False, _log=lambda message: None)

    with pytest.raises(RuntimeError, match="conversion failed"):
        builder_module.IndexBuilder.save_index(quiet_builder, index, str(tmp_path / "gpu.index"))

    calls.clear()
    builder_module.IndexBuilder.save_index(quiet_builder, index, str(tmp_path / "cpu.index"), on_gpu=False)
    assert calls == []
    assert (tmp_path / "cpu.index").exists()
