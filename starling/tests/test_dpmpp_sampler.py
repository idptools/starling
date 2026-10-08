"""
Tests for the DPM-Solver++(2M) sampler (starling.samplers.dpmpp_sampler).

These drive the sampler with a stub diffusion model that uses STARLING's real
cosine noise schedule but replaces the network, so they are fast and never
need model weights. The stub can act as the exact noise predictor for data
concentrated at a single point x0, for which the probability-flow ODE has the
closed-form solution x_t = alpha_t x0 + sigma_t eps; a correct solver must
land on it.
"""

from typing import cast

import pytest
import torch

from starling.data.schedulers import cosine_beta_schedule
from starling.models.diffusion import DiffusionModel
from starling.models.vae import VAE
from starling.samplers.dpmpp_sampler import DPMppSampler

SEQ = "MKTAYIAKQRQ"
N_TIMESTEPS = 1000
LATENT_SHAPE = (1, 24, 24)


class StubDiffusion:
    """
    Stand-in for DiffusionModel exposing only what DPMppSampler uses.

    With x0 given, model() returns the exact noise for data sitting at x0;
    otherwise it returns zeros. Every timestep it is called at is recorded.
    """

    def __init__(self, x0=None):
        self.num_timesteps = N_TIMESTEPS
        self.alphas_cumprod = torch.cumprod(1.0 - cosine_beta_schedule(N_TIMESTEPS), dim=0)
        self.device = torch.device("cpu")
        self.latent_space_scaling_factor = torch.tensor(1.0)
        self.x0 = x0
        self.called_at = []

    def sequence2labels(self, tokens, mask, ionic_strength):
        return torch.zeros(1, tokens.shape[1], 8)

    def model(self, x, timesteps, context, mask):
        t = int(timesteps[0])
        self.called_at.append(t)
        if self.x0 is None:
            return torch.zeros_like(x)
        alpha_bar = self.alphas_cumprod[t].float()
        return (x - alpha_bar.sqrt() * self.x0) / (1.0 - alpha_bar).sqrt()


class IdentityEncoder:
    """Decoder stand-in that returns latents unchanged."""

    def decode(self, x):
        return x


def make_sampler(model: StubDiffusion, n_steps: int) -> DPMppSampler:
    return DPMppSampler(cast(DiffusionModel, model), cast(VAE, IdentityEncoder()), n_steps)


def test_timesteps_decrease_to_one_and_each_is_evaluated_once():
    stub = StubDiffusion()
    sampler = make_sampler(stub, 12)

    assert sampler.n_steps == 12
    assert len(sampler.timesteps) == 12
    assert sampler.next_timesteps == [*sampler.timesteps[1:], 1]
    assert all(a > b for a, b in zip(sampler.timesteps, sampler.timesteps[1:]))

    sampler.sample(3, SEQ, show_per_step_progress_bar=False)
    # one network evaluation per step, at the timestep each step starts from
    assert stub.called_at == sampler.timesteps


def test_output_shape_and_reproducibility():
    sampler = make_sampler(StubDiffusion(), 12)

    torch.manual_seed(0)
    first = sampler.sample(4, SEQ, show_per_step_progress_bar=False)
    torch.manual_seed(0)
    again = sampler.sample(4, SEQ, show_per_step_progress_bar=False)

    assert first.shape == (4, *LATENT_SHAPE)
    assert torch.equal(first, again)


def test_solves_the_probability_flow_ode_exactly_for_point_data():
    torch.manual_seed(1)
    x0 = torch.randn(1, *LATENT_SHAPE)
    stub = StubDiffusion(x0=x0)
    sampler = make_sampler(stub, 12)

    torch.manual_seed(2)
    result = sampler.sample(5, SEQ, show_per_step_progress_bar=False)

    # replay the sampler's initial noise and evaluate the closed-form solution
    torch.manual_seed(2)
    x_start = torch.randn(5, *LATENT_SHAPE)
    start, end = sampler.timesteps[0], sampler.next_timesteps[-1]
    alpha_start, sigma_start = sampler.alpha[start], sampler.sigma[start]
    alpha_end, sigma_end = sampler.alpha[end], sampler.sigma[end]
    eps = (x_start - alpha_start * x0) / sigma_start
    expected = alpha_end * x0 + sigma_end * eps

    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


def test_rejects_constraints_and_bad_conformation_counts():
    sampler = make_sampler(StubDiffusion(), 12)
    with pytest.raises(ValueError, match="does not support constraints"):
        sampler.sample(2, SEQ, constraint=object())
    with pytest.raises(ValueError, match="num_conformations"):
        sampler.sample(0, SEQ)


def test_warns_when_requested_steps_collide():
    # far more steps than distinct integer timesteps in the usable range
    with pytest.warns(UserWarning, match="distinct steps"):
        sampler = make_sampler(StubDiffusion(), N_TIMESTEPS)
    assert 1 <= sampler.n_steps < N_TIMESTEPS
