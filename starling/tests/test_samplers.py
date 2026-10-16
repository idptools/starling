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
from starling.samplers import SAMPLERS, base_sampler
from starling.samplers.ddim_sampler import DDIM_FEW_STEPS_WARNING, DDIMSampler
from starling.samplers.plms_sampler import PLMS_MIN_STEPS, PLMSSampler
from starling.samplers.sampler_utilities import MAX_START_SIGNAL_SCALE, MINIMUM_SIGNAL_SCALE


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
    betas = cosine_beta_schedule(1000)
    alpha = torch.cumprod(1 - betas, dim=0)
    alpha_prev = torch.cat([torch.ones(1), alpha[:-1]])
    return cast(
        DiffusionModel,
        SimpleNamespace(
            device=torch.device("cpu"),
            num_timesteps=1000,
            alphas_cumprod=alpha,
            betas=betas,
            sqrt_recip_alphas=torch.sqrt(1 / (1 - betas)),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1 - alpha),
            posterior_variance=betas * (1 - alpha_prev) / (1 - alpha),
            latent_space_scaling_factor=2.0,
            sequence2labels=lambda tokens, mask, salt: tokens,
            model=lambda x, t, c, mask: 0.1 * x + t[:, None, None, None] * 0.0001,
        ),
    )


def test_ddim_30_preserves_reference_schedule_and_update():
    model = diffusion()
    sampler = DDIMSampler(model, unused_encoder(), 30)
    timesteps = np.arange(1, 1000, 33)[::-1]
    assert np.array_equal(sampler.timesteps, timesteps)
    torch.manual_seed(7)
    x = torch.randn(2, 1, 3, 3)
    context, mask = sampler.encode_sequence("AAAA")
    for index, timestep in enumerate(timesteps):
        eps = model.model(x, torch.full((2,), timestep), context, mask)
        alpha = model.alphas_cumprod[timestep]
        prev = model.alphas_cumprod[timesteps[index + 1] if index + 1 < len(timesteps) else 0]
        clean = (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
        expected = prev.sqrt() * clean + (1 - prev).sqrt() * eps
        actual = sampler.denoising_step(x, index, context, mask, [])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def noisiest_safe_timestep(model) -> int:
    """
    The noisiest timestep DDIM and PLMS may start from.

    Parameters
    ----------
    model : DiffusionModel
        Diffusion model (or stand-in) with an alphas_cumprod schedule.

    Returns
    -------
    int
        The last timestep with sqrt(alpha_bar) >= MINIMUM_SIGNAL_SCALE.
    """
    return int(torch.nonzero(model.alphas_cumprod.sqrt() >= MINIMUM_SIGNAL_SCALE).flatten()[-1])


def test_ddim_preserves_safe_quadratic_schedule_including_repeated_steps():
    # The legacy quadratic grid tops out at timestep 801, where the latents
    # still hold 30% signal, so it is stretched to run quadratically from
    # timestep 1 to the noisiest safe timestep (992), keeping its 100 steps and
    # its repeated steps at the low end
    model = diffusion()
    sampler = DDIMSampler(model, unused_encoder(), 100, ddim_discretize="quad")
    assert noisiest_safe_timestep(model) == 992
    stretched = np.rint(np.linspace(np.sqrt(992 - 1), 0, 100) ** 2 + 1).astype(int)
    assert np.array_equal(sampler.timesteps, stretched)
    assert sampler.timesteps[0] == 992 and sampler.timesteps[-1] == 1
    assert len(sampler.timesteps) == 100
    assert len(set(sampler.timesteps)) < 100


@pytest.mark.filterwarnings("ignore:DDIM with .* steps narrows the ensemble")
@pytest.mark.parametrize("steps", [1, 2, 5, 10, 20, 50])
def test_few_step_uniform_grid_is_stretched_to_start_in_pure_noise(steps):
    """Few-step grids used to start far below the noise level of the initial latents (10 steps: timestep 901)."""
    model = diffusion()
    sampler = DDIMSampler(model, unused_encoder(), steps)
    legacy = np.arange(0, 999, 1000 // steps) + 1

    # same number of steps, evenly spaced from the noisiest safe timestep down
    # to timestep 1 (a single step is just the noisiest safe timestep)
    stretched = np.rint(np.linspace(noisiest_safe_timestep(model), 1, len(legacy))).astype(int)
    assert np.array_equal(sampler.timesteps, stretched)
    assert sampler.next_timesteps == [*sampler.timesteps[1:], 0]


@pytest.mark.parametrize("sampler_class", [DDIMSampler, PLMSSampler])
def test_30_step_grid_is_not_stretched(sampler_class):
    """The default 30-step grid already starts in pure noise (sqrt(alpha_bar) ~ 0.012), so it is not stretched."""
    sampler = sampler_class(diffusion(), unused_encoder(), 30)
    assert np.array_equal(sampler.timesteps, np.arange(1, 1000, 33)[::-1])
    assert sampler.next_timesteps == [*sampler.timesteps[1:], 0]


@pytest.mark.filterwarnings("ignore:DDIM with .* steps narrows the ensemble")
@pytest.mark.parametrize("discretization", ["uniform", "quad"])
@pytest.mark.parametrize(
    ("sampler_class", "steps"),
    [
        (sampler_class, steps)
        for sampler_class in (DDIMSampler, PLMSSampler)
        for steps in (1, 2, 3, 5, 10, 20, 29, 30, 31, 50, 100, 200, 500, 1000)
        # PLMS refuses fewer than PLMS_MIN_STEPS steps
        if sampler_class is DDIMSampler or steps >= PLMS_MIN_STEPS
    ],
)
def test_every_ddim_grid_starts_in_pure_noise(sampler_class, steps, discretization):
    """Whatever the step count and spacing, the first step starts where the latents hold almost no signal."""
    sampler = sampler_class(diffusion(), unused_encoder(), steps, ddim_discretize=discretization)
    signal = sampler.alphas_cumprod[sampler.timesteps].sqrt()
    assert signal[0] <= MAX_START_SIGNAL_SCALE
    assert signal.min() >= MINIMUM_SIGNAL_SCALE
    assert np.all(np.diff(sampler.timesteps) <= 0)


@pytest.mark.parametrize("steps", [12, 500, 501, 900, 1000])
def test_ddim_limits_clean_prediction_amplification(steps):
    sampler = DDIMSampler(diffusion(), unused_encoder(), steps)
    assert sampler.alphas_cumprod[sampler.timesteps].sqrt().min() >= 0.01
    assert np.all(np.diff(sampler.timesteps) < 0)


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
    times = [*sampler.timesteps, sampler.next_timesteps[-1]]
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
    times = [*sampler.timesteps, sampler.next_timesteps[-1]]
    assert np.all(np.diff(times) < 0)
    assert times[-1] == 1
    assert torch.isfinite(sampler.lambda_t[times]).all()
    assert sampler.lambda_t[times[0]] >= -5.1
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


class RecordingConstraint:
    """Constraint stand-in that records how a sampler drives it and changes nothing."""

    def __init__(self):
        self.n_steps = None
        self.timesteps = []

    def initialize(self, encoder_model, latent_space_scaling_factor, n_steps, sequence_length):
        self.n_steps = n_steps

    def apply(self, latents, timestep, logger=None):
        self.timesteps.append(timestep)
        return latents


def identity_decoder() -> VAE:
    return cast(VAE, SimpleNamespace(device=torch.device("cpu"), decode=lambda x: x))


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_every_sampler_is_built_and_sampled_the_same_way(name):
    sampler = SAMPLERS[name](diffusion(), identity_decoder(), 30, 150)
    torch.manual_seed(0)
    latents = sampler.sample(3, "AAAA", show_per_step_progress_bar=False)
    assert latents.shape == (3, 1, 24, 24)
    assert torch.isfinite(latents).all()
    with pytest.raises(ValueError, match="num_conformations"):
        sampler.sample(0, "AAAA")


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_each_step_evaluates_the_network_where_it_starts(name):
    model = diffusion()
    network = model.model
    evaluated_at = []
    model.model = lambda x, t, c, mask: evaluated_at.append(int(t[0])) or network(x, t, c, mask)

    sampler = SAMPLERS[name](model, identity_decoder(), 30)
    sampler.sample(1, "AAAA", show_per_step_progress_bar=False)

    # PLMS's first step makes one extra evaluation, at the next grid timestep
    if name == "plms":
        assert evaluated_at.pop(1) == sampler.timesteps[1]
    assert evaluated_at == sampler.timesteps
    assert len(evaluated_at) == sampler.n_steps


@pytest.mark.parametrize("name", ["ddim", "ddpm", "plms"])
def test_constraints_are_scheduled_on_training_timesteps(name):
    # Constraints weight their guidance by timestep / n_steps, so every sampler
    # must tell them the number of training timesteps, not its own step count
    # (PLMS used to pass its step count, which switched guidance off until the
    # last step).
    sampler = SAMPLERS[name](diffusion(), identity_decoder(), 30)
    constraint = RecordingConstraint()
    sampler.sample(1, "AAAA", show_per_step_progress_bar=False, constraint=constraint)
    assert constraint.n_steps == 1000
    assert constraint.timesteps == [t for t in sampler.timesteps if t != 0]


class RecordingProgressBar:
    """tqdm stand-in that records the progress bars a sampler opens."""

    opened: list[dict] = []

    def __init__(self, *args, **kwargs):
        RecordingProgressBar.opened.append(kwargs)

    def update(self, n=1):
        pass

    def set_postfix_str(self, text):
        pass

    def refresh(self):
        pass

    def close(self):
        pass


@pytest.mark.parametrize(
    ("show_per_step_progress_bar", "constraint_verbose", "expected_bars"),
    [
        (True, True, [("Applying constraints", 1), ("DDIM steps", 2)]),
        (True, False, [("DDIM steps", 1)]),
        (False, True, []),
    ],
)
def test_constraint_progress_bar_follows_verbosity_and_has_its_own_row(
    monkeypatch, show_per_step_progress_bar, constraint_verbose, expected_bars
):
    """The constraint bar used to be shown always, on the same row as the per-step bar."""
    from starling.inference import constraints
    from starling.inference.constraints import RgConstraint

    RecordingProgressBar.opened = []
    monkeypatch.setattr(constraints, "tqdm", RecordingProgressBar)
    monkeypatch.setattr(base_sampler, "tqdm", RecordingProgressBar)

    sampler = DDIMSampler(diffusion(), identity_decoder(), 12)
    constraint = RgConstraint(target=0.0, verbose=constraint_verbose)
    sampler.sample(1, "AAAA", constraint=constraint, show_per_step_progress_bar=show_per_step_progress_bar)

    # the per-step bar is always made, but disabled when progress bars are off
    shown = [bar for bar in RecordingProgressBar.opened if not bar.get("disable", False)]
    assert [(bar["desc"].split(" (")[0], bar["position"]) for bar in shown] == expected_bars


@pytest.mark.parametrize("steps", [1, 10, PLMS_MIN_STEPS - 1])
def test_plms_rejects_too_few_steps(steps):
    # PLMS's multistep combination blows up the spread with few large steps
    with pytest.raises(ValueError, match=f"at least {PLMS_MIN_STEPS} steps"):
        PLMSSampler(diffusion(), unused_encoder(), steps)


def test_plms_accepts_the_minimum_number_of_steps():
    sampler = PLMSSampler(diffusion(), unused_encoder(), PLMS_MIN_STEPS)
    assert sampler.n_steps >= PLMS_MIN_STEPS


@pytest.mark.parametrize("steps", [1, DDIM_FEW_STEPS_WARNING - 1])
def test_ddim_warns_with_few_steps(steps):
    with pytest.warns(UserWarning, match="narrows the ensemble"):
        DDIMSampler(diffusion(), unused_encoder(), steps)


def test_ddim_does_not_warn_from_the_threshold_up():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DDIMSampler(diffusion(), unused_encoder(), DDIM_FEW_STEPS_WARNING)
        DDIMSampler(diffusion(), unused_encoder(), 30)


def test_public_plms_rejects_too_few_steps_before_loading_models(monkeypatch):
    from starling import generate
    from starling.inference import generation

    def unexpected_load(**kwargs):
        pytest.fail("models were loaded before the step count was checked")

    monkeypatch.setattr(generation.model_manager, "get_models", unexpected_load)
    with pytest.raises(ValueError, match=f"at least {PLMS_MIN_STEPS} steps"):
        generate("AAAA", sampler="plms", steps=PLMS_MIN_STEPS - 1, device="cpu")
