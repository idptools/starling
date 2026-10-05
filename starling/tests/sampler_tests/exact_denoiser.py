"""
Drive STARLING's samplers with the exact noise predictor for a known distribution.

If the trained network is replaced by the exact noise predictor
eps*(x_t, t) = E[eps | x_t] for a data distribution we know analytically, an
unbiased sampler has to return samples from that distribution. Any shift in
the mean, the spread or the mode weights is then bias introduced by the
sampler itself (its time discretization, where its schedule starts and stops,
or how it takes its final step), with no contribution from the network.

We use an elementwise Gaussian mixture in latent space. For each component k
(weight w_k, mean m_k, std s_k), x_t = alpha_t x_0 + sigma_t eps is Gaussian
with mean alpha_t m_k and variance v_k = alpha_t^2 s_k^2 + sigma_t^2, so

    E[x_0 | x_t, k] = m_k + (alpha_t s_k^2 / v_k) (x_t - alpha_t m_k)
    E[x_0 | x_t]    = sum_k r_k(x_t) E[x_0 | x_t, k]
    eps*(x_t, t)    = (x_t - alpha_t E[x_0 | x_t]) / sigma_t

where r_k(x_t) is component k's posterior responsibility, and alpha_t =
sqrt(abar_t), sigma_t = sqrt(1 - abar_t) come from STARLING's cosine noise
schedule (the schedule the released checkpoints were trained with).

Every latent element is independent, so one batch of conformations gives
n_conformations * 24 * 24 independent samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Final

import numpy as np
import torch

from starling.data.schedulers import cosine_beta_schedule
from starling.samplers.ddim_sampler import DDIMSampler
from starling.samplers.ddpm_sampler import DDPMSampler
from starling.samplers.dpmpp_sampler import DPMppSampler
from starling.samplers.plms_sampler import PLMSSampler

# the released checkpoints use 1000 timesteps of the cosine schedule
N_TIMESTEPS: Final[int] = 1000

# any valid sequence; the exact predictor ignores the conditioning
SEQUENCE: Final[str] = "MKTAYIAKQRQ"

# the samplers each STARLING sampler name maps to
SAMPLER_CLASSES: Final[dict[str, type]] = {
    "ddpm": DDPMSampler,
    "ddim": DDIMSampler,
    "plms": PLMSSampler,
    "dpmpp": DPMppSampler,
}


# the encoder's decoder is the identity, so samplers return latents directly
IDENTITY_DECODER: Final[SimpleNamespace] = SimpleNamespace(decode=lambda latents: latents)


@dataclass(frozen=True)
class GaussianMixture:
    """
    An elementwise Gaussian mixture in (scaled) latent space.

    Attributes
    ----------
    weights : tuple of float
        Component weights, summing to 1.
    means : tuple of float
        Component means.
    stds : tuple of float
        Component standard deviations.
    """

    weights: tuple[float, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]

    def __post_init__(self) -> None:
        assert len(self.weights) == len(self.means) == len(self.stds), "one value per component"
        assert abs(sum(self.weights) - 1.0) < 1e-12, "weights must sum to 1"
        assert all(std > 0 for std in self.stds), "stds must be positive"

    @classmethod
    def gaussian(cls, mean: float, std: float) -> GaussianMixture:
        """A single Gaussian component."""
        return cls((1.0,), (mean,), (std,))

    @property
    def mean(self) -> float:
        """Mean of the whole mixture."""
        return float(np.dot(self.weights, self.means))

    @property
    def std(self) -> float:
        """Standard deviation of the whole mixture."""
        second_moment = np.dot(self.weights, np.square(self.stds) + np.square(self.means))
        return float(np.sqrt(second_moment - self.mean**2))

    def noise_prediction(self, x: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        """
        Exact E[eps | x_t] for this distribution at noise level alpha_bar.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latents x_t, any shape.
        alpha_bar : torch.Tensor
            Scalar cumulative signal fraction abar_t.

        Returns
        -------
        torch.Tensor
            Exact noise prediction, same shape and dtype as x. Computed in
            float64 so the predictor's own rounding is negligible.
        """
        weights = torch.tensor(self.weights, dtype=torch.float64)
        means = torch.tensor(self.means, dtype=torch.float64)
        stds = torch.tensor(self.stds, dtype=torch.float64)
        alpha_bar64 = alpha_bar.to(torch.float64)
        alpha, sigma = alpha_bar64.sqrt(), (1.0 - alpha_bar64).sqrt()

        x_t = x.to(torch.float64)[..., None]
        variance = alpha**2 * stds**2 + sigma**2
        log_responsibility = (
            torch.log(weights) - 0.5 * torch.log(variance) - (x_t - alpha * means) ** 2 / (2.0 * variance)
        )
        responsibility = torch.softmax(log_responsibility, dim=-1)
        component_means = means + (alpha * stds**2 / variance) * (x_t - alpha * means)
        clean = (responsibility * component_means).sum(dim=-1)

        return ((x.to(torch.float64) - alpha * clean) / sigma).to(x.dtype)


def exact_diffusion_model(mixture: GaussianMixture, scaling_factor: float = 1.0) -> SimpleNamespace:
    """
    A stand-in for STARLING's DiffusionModel whose network is exact.

    Provides every attribute the four samplers read: the cosine schedule and
    the quantities derived from it, the latent scaling factor, a sequence
    encoder that returns its tokens unchanged, and the noise predictor.

    Parameters
    ----------
    mixture : GaussianMixture
        Distribution of the scaled latents the samplers should reproduce.
    scaling_factor : float, optional
        STARLING's latent_space_scaling_factor. Samplers divide their final
        latents by it before decoding. Default 1.

    Returns
    -------
    types.SimpleNamespace
        The stand-in diffusion model.
    """
    betas = cosine_beta_schedule(N_TIMESTEPS).to(torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

    def model(x: torch.Tensor, timesteps: torch.Tensor, context: object, mask: object) -> torch.Tensor:
        # every sampler evaluates one timestep for the whole batch
        return mixture.noise_prediction(x, alphas_cumprod[int(timesteps[0])])

    return SimpleNamespace(
        device=torch.device("cpu"),
        num_timesteps=N_TIMESTEPS,
        betas=betas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
        posterior_variance=betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        latent_space_scaling_factor=torch.tensor(scaling_factor),
        sequence2labels=lambda tokens, mask, ionic_strength: tokens,
        model=model,
    )


def draw_samples(
    sampler: str,
    steps: int,
    mixture: GaussianMixture,
    n_conformations: int,
    seed: int = 0,
    scaling_factor: float = 1.0,
) -> np.ndarray:
    """
    Run one STARLING sampler with the exact predictor and return its output.

    Parameters
    ----------
    sampler : str
        'ddpm', 'ddim', 'plms' or 'dpmpp'.
    steps : int
        Sampler steps. Ignored for 'ddpm', which always uses every timestep.
    mixture : GaussianMixture
        Distribution the sampler should reproduce.
    n_conformations : int
        Number of 24x24 latents to sample.
    seed : int, optional
        Seed for torch's global generator, which every sampler draws its
        noise from. Default 0.
    scaling_factor : float, optional
        Latent scaling factor (see exact_diffusion_model()). Default 1.

    Returns
    -------
    np.ndarray
        Every sampled (decoded) latent element as a flat float64 array, i.e.
        samples of mixture divided by scaling_factor.
    """
    model = exact_diffusion_model(mixture, scaling_factor)
    cls = SAMPLER_CLASSES[sampler]
    instance = cls(model, IDENTITY_DECODER) if sampler == "ddpm" else cls(model, IDENTITY_DECODER, steps)

    torch.manual_seed(seed)
    output = instance.sample(n_conformations, SEQUENCE, show_per_step_progress_bar=False)
    return output.to(torch.float64).flatten().numpy()
