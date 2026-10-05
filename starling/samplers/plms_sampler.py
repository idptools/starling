"""
PLMS sampling: DDIM steps driven by a multistep combination of noise predictions.

References
----------
[1] Liu, L., Ren, Y., Lin, Z., & Zhao, Z. (2022). Pseudo numerical methods
    for diffusion models on manifolds. ICLR 2022. arXiv:2202.09778.
"""

from __future__ import annotations

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

# PLMS combines the current noise prediction with up to three earlier ones
# (fourth-order Adams-Bashforth)
MAX_PREVIOUS_PREDICTIONS: Final[int] = 3


def pseudo_linear_multistep_noise(
    predicted_noise: torch.Tensor,
    previous_predictions: list[torch.Tensor],
) -> torch.Tensor:
    """
    Combine the current and earlier noise predictions with Adams-Bashforth weights.

    With one, two or three earlier predictions this is the second-, third- or
    fourth-order Adams-Bashforth combination used by PLMS [1].

    Parameters
    ----------
    predicted_noise : torch.Tensor
        Noise predicted at the current step.
    previous_predictions : list of torch.Tensor
        Noise predicted at earlier steps, oldest first. Must not be empty;
        only the last three are used.

    Returns
    -------
    torch.Tensor
        The combined noise prediction, same shape as predicted_noise.
    """
    assert previous_predictions, "the first PLMS step has no earlier predictions to combine"

    if len(previous_predictions) == 1:
        return (3 * predicted_noise - previous_predictions[-1]) / 2

    if len(previous_predictions) == 2:
        return (23 * predicted_noise - 16 * previous_predictions[-1] + 5 * previous_predictions[-2]) / 12

    return (
        55 * predicted_noise
        - 59 * previous_predictions[-1]
        + 37 * previous_predictions[-2]
        - 9 * previous_predictions[-3]
    ) / 24


class PLMSSampler(BaseSampler):
    """
    Deterministic multistep sampler on DDIM's timestep grid.

    PLMS [1] takes the same deterministic DDIM steps, but feeds them an
    Adams-Bashforth combination of the current and up to three earlier noise
    predictions, which makes each step higher order. The first step has no
    history, so it averages the noise predicted at the start and at a
    provisional end of the step (a pseudo improved Euler step), which costs
    one extra network evaluation.
    """

    name = "PLMS"

    def __init__(
        self,
        ddpm_model: DiffusionModel,
        encoder_model: VAE,
        n_steps: int,
        ionic_strength: float = configs.DEFAULT_IONIC_STRENGTH,
        ddim_discretize: str = UNIFORM_DISCRETIZATION,
    ) -> None:
        """
        Set up a PLMS sampler.

        Parameters
        ----------
        ddpm_model : DiffusionModel
            The trained diffusion model.
        encoder_model : VAE
            The trained VAE, used to decode latents into distance maps.
        n_steps : int
            Requested number of steps; the grid is DDIM's (30 gives 31 steps).
        ionic_strength : float, optional
            Ionic strength in mM. Default configs.DEFAULT_IONIC_STRENGTH.
        ddim_discretize : str, optional
            Spacing of the timestep grid, 'uniform' (default) or 'quad'.

        Raises
        ------
        ValueError
            If n_steps is not an integer between 1 and the number of training
            timesteps.
        """
        super().__init__(ddpm_model, encoder_model, ionic_strength)
        check_step_count(n_steps, self.num_timesteps)
        self.timesteps, self.next_timesteps = ddim_timesteps(self.alphas_cumprod, n_steps, ddim_discretize)

    def denoising_step(
        self,
        latents: torch.Tensor,
        step_index: int,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
        history: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Take one PLMS step.

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
            Noise predicted at the start of earlier steps, oldest first. This
            step appends its own prediction.

        Returns
        -------
        torch.Tensor
            Latents at timestep self.next_timesteps[step_index].
        """
        timestep = self.timesteps[step_index]
        alpha_bar = self.alphas_cumprod[timestep]
        next_alpha_bar = self.alphas_cumprod[self.next_timesteps[step_index]]
        predicted_noise = self.predict_noise(latents, timestep, context, attention_mask)

        if history:
            combined_noise = pseudo_linear_multistep_noise(predicted_noise, history)
        else:
            # pseudo improved Euler: also predict the noise where a plain DDIM
            # step would land (the next grid timestep), and average the two
            provisional_latents = ddim_update(latents, predicted_noise, alpha_bar, next_alpha_bar)
            provisional_timestep = self.timesteps[min(step_index + 1, self.n_steps - 1)]
            provisional_noise = self.predict_noise(provisional_latents, provisional_timestep, context, attention_mask)
            combined_noise = (predicted_noise + provisional_noise) / 2

        history.append(predicted_noise)
        del history[:-MAX_PREVIOUS_PREDICTIONS]

        return ddim_update(latents, combined_noise, alpha_bar, next_alpha_bar)
