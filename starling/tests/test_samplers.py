"""Sampler numerical contracts without model checkpoints."""

from types import SimpleNamespace
from typing import cast
import warnings

import numpy as np
import pytest
import torch

from starling.data.schedulers import cosine_beta_schedule, sigmoid_beta_schedule
from starling.models.diffusion import DiffusionModel
from starling.models.vae import VAE
from starling.samplers.ddim_sampler import DDIMSampler


def test_default_sigmoid_schedule_is_finite_and_invalid_interval_is_rejected():
    betas = sigmoid_beta_schedule(1000)
    assert torch.isfinite(betas).all()
    assert torch.all((betas >= 0) & (betas < 1))

    with pytest.raises(ValueError, match="end > start"):
        sigmoid_beta_schedule(10, start=3, end=3)


def unused_encoder() -> VAE:
    return cast(VAE, None)


def test_inference_loader_disables_training_only_random_conditioning(monkeypatch, tmp_path):
    from starling.inference import model_loading
    from starling.models import diffusion, transformer, vae as vae_module, vit
    from starling.models.transformer import SequenceEncoder

    # Only checkpoint IO and large architecture construction are substituted.
    # Evaluation mode and the actual sequence-conditioning forward remain real.
    sequence_encoder = SequenceEncoder(1, 8, 2)
    model = torch.nn.Module()
    model.add_module("sequence_encoder", sequence_encoder)
    vae = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(8, 8))
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.touch()
    monkeypatch.setattr(transformer, "SequenceEncoder", lambda *args: sequence_encoder)
    monkeypatch.setattr(vit, "ViT", lambda *args: torch.nn.Identity())
    monkeypatch.setattr(
        diffusion.DiffusionModel,
        "load_from_checkpoint",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(vae_module.VAE, "load_from_checkpoint", lambda *args, **kwargs: vae)

    encoder, diffusion_model = model_loading.ModelManager().load_models(str(checkpoint), str(checkpoint), "cpu")
    assert all(not module.training for root in (encoder, diffusion_model) for module in root.modules())
    tokens = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    rng_state = torch.random.get_rng_state()
    sequence_encoder(tokens, mask, torch.tensor([[150]]))
    assert torch.equal(rng_state, torch.random.get_rng_state())


def diffusion():
    alpha = torch.cumprod(1 - cosine_beta_schedule(1000), dim=0)
    return cast(
        DiffusionModel,
        SimpleNamespace(
            device=torch.device("cpu"),
            num_timesteps=1000,
            alphas_cumprod=alpha,
            latent_space_scaling_factor=2.0,
            sequence2labels=lambda tokens, mask, salt: tokens,
            model=lambda x, t, c, mask: 0.1 * x + t[:, None, None, None] * 0.0001,
        ),
    )


def test_ddim_30_preserves_reference_schedule_and_update():
    model = diffusion()
    sampler = DDIMSampler(model, unused_encoder(), 30)
    timesteps = np.arange(1, 1000, 33)
    assert np.array_equal(sampler.ddim_time_steps, timesteps)
    torch.manual_seed(7)
    x, eps = torch.randn(2, 1, 3, 3), torch.randn(2, 1, 3, 3)
    for index in range(len(timesteps)):
        alpha = model.alphas_cumprod[timesteps[index]]
        prev = model.alphas_cumprod[timesteps[index - 1] if index else 0]
        clean = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
        expected = prev.sqrt() * clean + (1 - prev).sqrt() * eps
        actual, _ = sampler.get_x_prev_and_pred_x0(eps, index, x, 1.0, False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ddim_preserves_safe_quadratic_schedule_including_repeated_steps():
    sampler = DDIMSampler(diffusion(), unused_encoder(), 100, ddim_discretize="quad")
    legacy = (np.linspace(0, np.sqrt(800), 100) ** 2).astype(int) + 1
    assert np.array_equal(sampler.ddim_time_steps, legacy)


@pytest.mark.parametrize("steps", [12, 500, 501, 900, 1000])
def test_ddim_limits_clean_prediction_amplification(steps):
    sampler = DDIMSampler(diffusion(), unused_encoder(), steps)
    assert sampler.ddim_alpha_sqrt.min() >= 0.01
    assert np.all(np.diff(sampler.ddim_time_steps) > 0)


@pytest.mark.parametrize("steps", [0, -1, 1001, 2.5, True])
def test_ddim_rejects_invalid_step_counts(steps):
    with pytest.raises(ValueError, match="n_steps"):
        DDIMSampler(diffusion(), unused_encoder(), steps)


def test_dpmpp_matches_independent_scalar_midpoint_reference():
    from starling.samplers.dpmpp_sampler import DPMppSampler

    model = diffusion()
    sampler = DPMppSampler(model, cast(VAE, SimpleNamespace(decode=lambda x: x)), 12)
    torch.manual_seed(17)
    expected = torch.randn(2, 1, 24, 24).numpy().astype(np.float64)
    abar = model.alphas_cumprod.numpy().astype(np.float64)
    alpha, sigma = np.sqrt(abar), np.sqrt(1 - abar)
    lam = np.log(alpha / sigma)
    previous = previous_time = None
    times = sampler.timesteps
    assert len(times) == 13
    for i, (source, target) in enumerate(zip(times, times[1:])):
        noise = 0.1 * expected + source * 0.0001
        clean = (expected - sigma[source] * noise) / alpha[source]
        h = lam[target] - lam[source]
        estimate = clean
        if previous is not None and i != len(times) - 2:
            estimate = clean + (clean - previous) * h / (2 * (lam[source] - lam[previous_time]))
        expected = sigma[target] / sigma[source] * expected - alpha[target] * np.expm1(-h) * estimate
        previous, previous_time = clean, source
    torch.manual_seed(17)
    actual = sampler.sample(2, "AAAA", show_per_step_progress_bar=False)
    np.testing.assert_allclose(actual.numpy(), expected / 2, rtol=2e-5, atol=2e-5)
    torch.manual_seed(17)
    assert torch.equal(actual, sampler.sample(2, "AAAA", show_per_step_progress_bar=False))


@pytest.mark.parametrize("steps", [1, 12, 20, 30, 500, 1000])
def test_dpmpp_schedule_is_strict_and_finite(steps):
    from starling.samplers.dpmpp_sampler import DPMppSampler

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sampler = DPMppSampler(diffusion(), cast(VAE, SimpleNamespace(decode=lambda x: x)), steps)
    assert np.all(np.diff(sampler.timesteps) < 0)
    assert sampler.timesteps[-1] == 1
    assert torch.isfinite(sampler.lambda_t[sampler.timesteps]).all()
    assert sampler.lambda_t[sampler.timesteps[0]] >= -5.1
    assert torch.isfinite(sampler.sample(1, "AAAA", show_per_step_progress_bar=False)).all()


def test_dpmpp_rejects_constraints_before_sampling():
    from starling.samplers.dpmpp_sampler import DPMppSampler

    sampler = DPMppSampler(diffusion(), unused_encoder(), 12)
    with pytest.raises(ValueError, match="constraints"):
        sampler.sample(1, "AAAA", constraint=object())


@pytest.mark.parametrize("steps", [0, -1, 1001, 2.5, True])
def test_dpmpp_rejects_invalid_steps(steps):
    from starling.samplers.dpmpp_sampler import DPMppSampler

    with pytest.raises(ValueError, match="n_steps"):
        DPMppSampler(diffusion(), unused_encoder(), steps)


def test_dpmpp_generation_uses_shared_output_path(monkeypatch):
    from starling import generate
    from starling.inference import generation

    model = diffusion()
    decoder = SimpleNamespace(decode=lambda x: torch.nn.functional.interpolate(x.abs(), size=(384, 384)))
    monkeypatch.setattr(generation.model_manager, "get_models", lambda **kwargs: (decoder, model))
    result = generate(
        "AAAA",
        sampler="dpmpp",
        steps=12,
        conformations=3,
        batch_size=2,
        device="cpu",
        show_progress_bar=False,
        show_per_step_progress_bar=False,
        return_single_ensemble=True,
    )
    maps = result.distance_maps()
    assert maps.shape == (3, 4, 4)
    assert np.isfinite(maps).all()
    assert np.array_equal(maps, maps.transpose(0, 2, 1))


def test_cli_accepts_dpmpp(monkeypatch, tmp_path):
    import sys
    from starling.scripts import starling_main_cli as cli

    passed = {}
    monkeypatch.setattr(cli, "generate", lambda **kwargs: passed.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "starling",
            "AAAA",
            "--sampler",
            "dpmpp",
            "--steps",
            "12",
            "-o",
            str(tmp_path),
        ],
    )
    cli.main()
    assert passed["sampler"] == "dpmpp"
    assert passed["steps"] == 12


def test_ddim_guidance_reduces_target_loss_with_the_same_initial_noise():
    from starling.inference.constraints import RgConstraint
    from starling.utilities import symmetrize_tensor_distance_maps

    decoder = cast(VAE, SimpleNamespace(device=torch.device("cpu"), decode=lambda x: x))
    sampler = DDIMSampler(diffusion(), decoder, 12)
    constraint = RgConstraint(target=0.0, verbose=False)
    torch.manual_seed(17)
    unconstrained = sampler.sample(2, "AAAA", show_per_step_progress_bar=False)
    torch.manual_seed(17)
    constrained = sampler.sample(2, "AAAA", constraint=constraint, show_per_step_progress_bar=False)
    _, before = constraint.compute_loss(symmetrize_tensor_distance_maps(unconstrained))
    _, after = constraint.compute_loss(symmetrize_tensor_distance_maps(constrained))
    assert after < before


@pytest.mark.parametrize("eta", [-0.1, 1.1, float("nan")])
def test_ddim_rejects_invalid_noise_levels(eta):
    with pytest.raises(ValueError, match="ddim_eta"):
        DDIMSampler(diffusion(), unused_encoder(), 30, ddim_eta=eta)


def test_dpmpp_rejects_nonmonotonic_model_schedule():
    from starling.samplers.dpmpp_sampler import DPMppSampler

    model = diffusion()
    model.alphas_cumprod[500] = model.alphas_cumprod[400]
    with pytest.raises(ValueError, match="decreasing"):
        DPMppSampler(model, unused_encoder(), 12)


def test_public_dpmpp_rejects_constraints_before_loading_models(monkeypatch):
    from starling import generate
    from starling.inference import generation

    def unexpected_load(**kwargs):
        pytest.fail()

    monkeypatch.setattr(generation.model_manager, "get_models", unexpected_load)
    with pytest.raises(ValueError, match="constraints"):
        generate("AAAA", sampler="dpmpp", constraint=object(), device="cpu")
