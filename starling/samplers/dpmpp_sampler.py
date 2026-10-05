"""
DPM-Solver++(2M) sampling: a second-order solver for the probability-flow ODE.

References
----------
[1] Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022).
    DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic
    models. arXiv:2211.01095.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from starling import configs
from starling.samplers.base_sampler import BaseSampler
from starling.samplers.sampler_utilities import check_step_count

if TYPE_CHECKING:
    from starling.models.diffusion import DiffusionModel
    from starling.models.vae import VAE

# Noisiest half-log-SNR the solver starts from. This is the cutoff the
# upstream DPM-Solver code uses for the cosine schedule; below it, converting
# a noise prediction into a clean estimate becomes unstable.
MINIMUM_LOG_SNR: Final[float] = -5.1

# The trajectory ends at timestep 1, the cleanest timestep on DDIM's grid
FINAL_TIMESTEP: Final[int] = 1

SCHEDULE_ERROR: Final[str] = "DPM++ requires a finite, decreasing log-SNR schedule"


def dpmpp_timesteps(lambda_t: torch.Tensor, n_steps: int) -> list[int]:
    """
    Space a trajectory's timesteps uniformly in half-log-SNR.

    Parameters
    ----------
    lambda_t : torch.Tensor
        Half-log-SNR, log(alpha_t / sigma_t), of every training timestep,
        shape (num_timesteps,).
    n_steps : int
        Requested number of steps.

    Returns
    -------
    list of int
        Every timestep the trajectory visits, noisiest first and ending at
        FINAL_TIMESTEP. Rounding to integer timesteps can make two targets
        collide, so this can hold fewer than n_steps + 1 timesteps.

    Raises
    ------
    ValueError
        If the usable part of the schedule is not finite and strictly
        decreasing in log-SNR.
    """
    log_snr = lambda_t.cpu().numpy()
    usable = np.flatnonzero(np.isfinite(log_snr) & (log_snr >= MINIMUM_LOG_SNR))
    if usable.size < 2 or usable[-1] <= FINAL_TIMESTEP:
        raise ValueError(SCHEDULE_ERROR)

    start_timestep = int(usable[-1])
    usable_log_snr = log_snr[FINAL_TIMESTEP : start_timestep + 1]
    if not np.all(np.isfinite(usable_log_snr)) or not np.all(np.diff(usable_log_snr) < 0):
        raise ValueError(SCHEDULE_ERROR)

    # interpolate the timestep at each evenly spaced log-SNR (np.interp needs
    # increasing x, so both arrays are reversed)
    target_log_snr = np.linspace(log_snr[start_timestep], log_snr[FINAL_TIMESTEP], n_steps + 1)
    candidate_timesteps = np.arange(FINAL_TIMESTEP, start_timestep + 1)
    timesteps = np.rint(np.interp(target_log_snr, usable_log_snr[::-1], candidate_timesteps[::-1])).astype(int)

    return [int(timestep) for timestep in np.unique(timesteps)[::-1]]


class DPMppSampler(BaseSampler):
    """
    Second-order deterministic multistep solver with a first-order final step.

    Each step solves the probability-flow ODE exactly for a clean-latent
    estimate that is extrapolated linearly from this step's and the previous
    step's estimates (Algorithm 2 of [1]). The first and last steps are first
    order. Timesteps are evenly spaced in half-log-SNR and rounded to integer
    timesteps; collisions reduce the number of steps, with a warning.
    Constraints are not supported.
    """

    name = "DPM++"
    supports_constraints = False

    def __init__(
        self,
        ddpm_model: DiffusionModel,
        encoder_model: VAE,
        n_steps: int,
        ionic_strength: float = configs.DEFAULT_IONIC_STRENGTH,
    ) -> None:
        """
        Set up a DPM++ sampler.

        Parameters
        ----------
        ddpm_model : DiffusionModel
            The trained diffusion model.
        encoder_model : VAE
            The trained VAE, used to decode latents into distance maps.
        n_steps : int
            Number of steps (network evaluations).
        ionic_strength : float, optional
            Ionic strength in mM. Default configs.DEFAULT_IONIC_STRENGTH.

        Raises
        ------
        ValueError
            If n_steps is not an integer between 1 and the number of training
            timesteps, or the schedule is unusable.
        """
        super().__init__(ddpm_model, encoder_model, ionic_strength)
        check_step_count(n_steps, self.num_timesteps)

        # signal scale, noise scale and half-log-SNR of every timestep
        self.alpha = self.alphas_cumprod.sqrt()
        self.sigma = (1 - self.alphas_cumprod).sqrt()
        self.lambda_t = self.alpha.log() - self.sigma.log()

        trajectory = dpmpp_timesteps(self.lambda_t, n_steps)
        self.timesteps = trajectory[:-1]
        self.next_timesteps = trajectory[1:]

        if self.n_steps != n_steps:
            warnings.warn(
                f"DPM++ requested {n_steps} steps; using {self.n_steps} distinct steps at this model's timestep resolution",
                stacklevel=2,
            )

    def denoising_step(
        self,
        latents: torch.Tensor,
        step_index: int,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
        history: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Take one DPM-Solver++(2M) step.

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
            Holds the previous step's clean-latent estimate, if there was a
            previous step. This step replaces it with its own.

        Returns
        -------
        torch.Tensor
            Latents at timestep self.next_timesteps[step_index].
        """
        timestep = self.timesteps[step_index]
        next_timestep = self.next_timesteps[step_index]

        predicted_noise = self.predict_noise(latents, timestep, context, attention_mask)
        clean_estimate = (latents - self.sigma[timestep] * predicted_noise) / self.alpha[timestep]
        log_snr_step = self.lambda_t[next_timestep] - self.lambda_t[timestep]

        # second order: extrapolate the clean estimate using the previous step,
        # except on the final step (lower-order final step, as in [1])
        is_final_step = step_index == self.n_steps - 1
        extrapolated_estimate = clean_estimate
        if history and not is_final_step:
            previous_timestep = self.timesteps[step_index - 1]
            step_ratio = (self.lambda_t[timestep] - self.lambda_t[previous_timestep]) / log_snr_step
            extrapolated_estimate = clean_estimate + (clean_estimate - history[-1]) / (2 * step_ratio)

        history[:] = [clean_estimate]

        return (
            self.sigma[next_timestep] / self.sigma[timestep] * latents
            - self.alpha[next_timestep] * torch.expm1(-log_snr_step) * extrapolated_estimate
        )
