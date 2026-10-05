"""
Ancestral DDPM sampling: the reverse process STARLING's model was trained for.

References
----------
[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic
    models. NeurIPS 2020. arXiv:2006.11239.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from starling import configs
from starling.samplers.base_sampler import BaseSampler

if TYPE_CHECKING:
    from starling.models.diffusion import DiffusionModel
    from starling.models.vae import VAE


class DDPMSampler(BaseSampler):
    """
    Stochastic sampler that steps through every training timestep.

    Each step removes the predicted noise and adds back fresh noise with the
    variance of the true posterior (Algorithm 2 of [1]). This is the slowest
    sampler, one network evaluation per training timestep, and the reference
    the faster samplers approximate.
    """

    name = "DDPM"

    def __init__(
        self,
        ddpm_model: DiffusionModel,
        encoder_model: VAE,
        n_steps: int | None = None,
        ionic_strength: float = configs.DEFAULT_IONIC_STRENGTH,
    ) -> None:
        """
        Set up a DDPM sampler.

        Parameters
        ----------
        ddpm_model : DiffusionModel
            The trained diffusion model.
        encoder_model : VAE
            The trained VAE, used to decode latents into distance maps.
        n_steps : int or None, optional
            Not used: DDPM always takes every training timestep. Accepted so
            that every sampler can be constructed the same way.
        ionic_strength : float, optional
            Ionic strength in mM. Default configs.DEFAULT_IONIC_STRENGTH.
        """
        super().__init__(ddpm_model, encoder_model, ionic_strength)

        # every timestep, noisiest first
        self.timesteps = list(range(self.num_timesteps - 1, -1, -1))

        # coefficients of the reverse step, as registered by the diffusion model
        self.betas = ddpm_model.betas
        self.sqrt_recip_alphas = ddpm_model.sqrt_recip_alphas
        self.sqrt_one_minus_alphas_cumprod = ddpm_model.sqrt_one_minus_alphas_cumprod
        self.posterior_variance = ddpm_model.posterior_variance

    def denoising_step(
        self,
        latents: torch.Tensor,
        step_index: int,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
        history: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Sample x_{t-1} given x_t (Algorithm 2, line 4 of [1]).

        Parameters
        ----------
        latents : torch.Tensor
            Latents x_t, shape (batch, 1, 24, 24).
        step_index : int
            Which step this is; the step starts from self.timesteps[step_index].
        context : torch.Tensor
            Sequence context.
        attention_mask : torch.Tensor
            Attention mask for the context.
        history : list of torch.Tensor
            Unused; DDPM steps depend only on the current latents.

        Returns
        -------
        torch.Tensor
            Latents x_{t-1}. The step from t = 0 returns the posterior mean
            without adding noise.
        """
        timestep = self.timesteps[step_index]
        predicted_noise = self.predict_noise(latents, timestep, context, attention_mask)

        posterior_mean = self.sqrt_recip_alphas[timestep] * (
            latents - self.betas[timestep] * predicted_noise / self.sqrt_one_minus_alphas_cumprod[timestep]
        )
        if timestep == 0:
            return posterior_mean

        return posterior_mean + self.posterior_variance[timestep].sqrt() * torch.randn_like(latents)
