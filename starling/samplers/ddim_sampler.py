"""
DDIM sampling: STARLING's default sampler.

References
----------
[1] Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit
    models. ICLR 2021. arXiv:2010.02502.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Final

import torch

from starling import configs
from starling.samplers.base_sampler import BaseSampler
from starling.samplers.sampler_utilities import (
    UNIFORM_DISCRETIZATION,
    check_step_count,
    ddim_timesteps,
    ddim_update,
)

if TYPE_CHECKING:
    from starling.models.diffusion import DiffusionModel
    from starling.models.vae import VAE


# Below this many steps DDIM's deterministic steps are so large that they pull
# samples towards their mean: with an exact noise predictor the spread of
# N(1, 0.3^2) came out at 0.05-0.57 of its true value with 2-5 steps and ~0.8
# with 10 (see starling/tests/sampler_tests/test_exact_denoiser_bias.py)
DDIM_FEW_STEPS_WARNING: Final[int] = 10


class DDIMSampler(BaseSampler):
    """
    Sampler that skips most timesteps by stepping along a non-Markovian process.

    DDIM [1] estimates the clean latent at each step and moves straight to a
    much less noisy timestep, so it needs ~30 network evaluations instead of
    DDPM's 1000. With eta = 0 (the default) every step is deterministic and
    the only randomness is the initial noise.
    """

    name = "DDIM"

    def __init__(
        self,
        ddpm_model: DiffusionModel,
        encoder_model: VAE,
        n_steps: int,
        ionic_strength: float = configs.DEFAULT_IONIC_STRENGTH,
        ddim_discretize: str = UNIFORM_DISCRETIZATION,
        ddim_eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        """
        Set up a DDIM sampler.

        Parameters
        ----------
        ddpm_model : DiffusionModel
            The trained diffusion model.
        encoder_model : VAE
            The trained VAE, used to decode latents into distance maps.
        n_steps : int
            Requested number of steps; see sampler_utilities.ddim_timesteps()
            for how this maps onto the timestep grid (30 gives 31 steps).
        ionic_strength : float, optional
            Ionic strength in mM. Default configs.DEFAULT_IONIC_STRENGTH.
        ddim_discretize : str, optional
            Spacing of the timestep grid, 'uniform' (default) or 'quad'.
        ddim_eta : float, optional
            Interpolates between deterministic DDIM (0, default) and a process
            with DDPM's noise level (1); eta in Eq. 16 of [1].
        generator : torch.Generator or None, optional
            Generator on the model's device that every random number the
            sampler draws comes from. Default None, which draws from torch's
            global generator.

        Raises
        ------
        ValueError
            If n_steps is not an integer between 1 and the number of training
            timesteps, or ddim_eta is outside [0, 1].
        """
        super().__init__(ddpm_model, encoder_model, ionic_strength, generator)
        check_step_count(n_steps, self.num_timesteps)
        if n_steps < DDIM_FEW_STEPS_WARNING:
            warnings.warn(
                f"DDIM with {n_steps} steps narrows the ensemble: such large deterministic steps pull "
                f"samples towards their mean. Use at least {DDIM_FEW_STEPS_WARNING} steps (30 is the "
                "default), or DPM++ for fewer steps.",
                stacklevel=2,
            )
        if not 0.0 <= ddim_eta <= 1.0:
            raise ValueError("ddim_eta must be between 0 and 1")

        self.eta = ddim_eta
        self.timesteps, self.next_timesteps = ddim_timesteps(self.alphas_cumprod, n_steps, ddim_discretize)

        # size of the fresh noise each step adds (sigma_t, Eq. 16 of [1])
        alpha_bar = self.alphas_cumprod[self.timesteps]
        next_alpha_bar = self.alphas_cumprod[self.next_timesteps]
        # kept as Python floats, so checking whether a step adds noise does
        # not force a GPU-to-CPU synchronization every step
        self.noise_scales: list[float] = (
            ddim_eta * ((1 - next_alpha_bar) / (1 - alpha_bar) * (1 - alpha_bar / next_alpha_bar)).sqrt()
        ).tolist()

    def denoising_step(
        self,
        latents: torch.Tensor,
        step_index: int,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
        history: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Take one DDIM step (Eq. 12 of [1]).

        Parameters
        ----------
        latents : torch.Tensor
            Latents at timestep self.timesteps[step_index], shape (batch, 1, 24, 24).
        step_index : int
            Which step this is.
        context : torch.Tensor
            Sequence context.
        attention_mask : torch.Tensor
            Attention mask for the context.
        history : list of torch.Tensor
            Unused; DDIM steps depend only on the current latents.

        Returns
        -------
        torch.Tensor
            Latents at timestep self.next_timesteps[step_index].
        """
        timestep = self.timesteps[step_index]
        predicted_noise = self.predict_noise(latents, timestep, context, attention_mask)

        return ddim_update(
            latents,
            predicted_noise,
            self.alphas_cumprod[timestep],
            self.alphas_cumprod[self.next_timesteps[step_index]],
            self.noise_scales[step_index],
            generator=self.generator,
        )
