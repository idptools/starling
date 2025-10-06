"""SAMPLING ONLY."""

from functools import partial

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from starling.data.tokenizer import StarlingTokenizer
from starling.inference.constraints import (
    ConstraintLogger,
    DistanceConstraint,
    HelicityConstraint,
    RgConstraint,
)

# def noise_like(shape, device, repeat=False):
#     repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
#         shape[0], *((1,) * (len(shape) - 1))
#     )
#     noise = lambda: torch.randn(shape, device=device)
#     return repeat_noise() if repeat else noise()


def dynamic_thresholding_fn(
    x0: torch.Tensor, p: float = 0.995, max_val: float = 1.0
) -> torch.Tensor:
    """
    Applies dynamic thresholding to VAE latent predictions.
    Args:
        x0: Tensor of shape (B, C, H, W) — predicted x_0 in latent space
        p: Quantile threshold, e.g., 0.995
        max_val: Absolute minimum threshold cap to avoid over-compression
    Returns:
        Clamped and rescaled tensor.
    """
    # Compute scale per sample
    s = torch.quantile(x0.abs().reshape(x0.shape[0], -1), p, dim=1, keepdim=True)
    s = torch.maximum(s, torch.tensor(max_val, device=x0.device))
    while s.ndim < x0.ndim:
        s = s.unsqueeze(-1)  # expand to match x0 shape
    x0 = torch.clamp(x0, -s, s) / s
    return x0


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    return sigmas, alphas, alphas_prev


class PLMSSampler(object):
    def __init__(
        self,
        ddpm_model,
        encoder_model,
        n_steps,
        salt=150,
        ddim_discretize="uniform",
        schedule="linear",
        **kwargs,
    ):
        super().__init__()
        self.ddpm_model = ddpm_model
        self.encoder_model = encoder_model
        self.ddpm_num_timesteps = ddpm_model.num_timesteps
        self.n_steps = n_steps
        self.schedule = schedule
        self.device = ddpm_model.device
        self.salt = torch.tensor([salt], device=self.device).unsqueeze(0)

        self.tokenizer = StarlingTokenizer()

        # Makes the sampler deterministic (I think)
        ddim_eta = 0

        # Ways to discretize the generative process
        if ddim_discretize == "uniform":
            c = self.ddpm_num_timesteps // n_steps
            self.ddim_time_steps = (
                np.asarray(list(range(0, self.ddpm_num_timesteps - 1, c))) + 1
            )
        elif ddim_discretize == "quad":
            self.ddim_time_steps = (
                (np.linspace(0, np.sqrt(self.ddpm_num_timesteps * 0.8), n_steps)) ** 2
            ).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            # ddim sampling parameters
            self.ddim_sigmas, self.ddim_alphas, self.ddim_alphas_prev = (
                make_ddim_sampling_parameters(
                    alphacums=self.ddpm_model.alphas_cumprod.cpu(),
                    ddim_timesteps=self.ddim_time_steps,
                    eta=ddim_eta,
                )
            )

            self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas) ** 0.5

            self.sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
                (1 - self.ddpm_model.alphas_cumprod_prev)
                / (1 - self.ddpm_model.alphas_cumprod)
                * (
                    1
                    - self.ddpm_model.alphas_cumprod
                    / self.ddpm_model.alphas_cumprod_prev
                )
            )

    def generate_labels(self, labels: str) -> torch.Tensor:
        """
        Generate labels to condition the generative process on.

        Parameters
        ----------
        labels : str
            A sequence to generate labels from.

        Returns
        -------
        torch.Tensor
            The labels to condition the generative process on.
        """

        labels = torch.tensor(self.tokenizer.encode(labels), device=self.device)
        labels = rearrange(labels, "f -> 1 f")
        attention_mask = torch.ones_like(labels, device=self.device, dtype=torch.bool)

        labels = self.ddpm_model.sequence2labels(labels, attention_mask, self.salt)

        return labels, attention_mask

    @torch.no_grad()
    def sample(
        self,
        num_conformations: int,
        labels: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        show_per_step_progress_bar: bool = True,
        batch_count: int = 1,
        max_batch_count: int = 1,
        constraint=None,
    ) -> torch.Tensor:
        """
        Sample the generative process using the DDIM model.

        Parameters
        ----------
        num_conformations : int
            Number of conformations to generate.

        labels : torch.Tensor
            The labels to condition the generative process on.

        repeat_noise : bool, optional
            _description_, by default False

        temperature : float, optional
            _description_, by default 1.0

        show_per_step_progress_bar : bool, optional
            whether to show progress bar per step.

        batch_count : int, optional
            The batch count for the progress bar, by default 1

        max_batch_count : int, optional
            The maximum batch count for the progress bar, by default 1

        Returns
        -------
        torch.Tensor
            The generated distance maps.
        """

        sequence_length = len(labels)

        # Initialize the latents with noise
        x = torch.randn(
            [num_conformations, 1, 24, 24],
            device=self.device,
        )

        time_steps = np.flip(self.ddim_time_steps)

        # Get the labels to condition the generative process on
        labels, attention_mask = self.generate_labels(
            labels,
        )

        # initialize progress bar if we want to show it
        if show_per_step_progress_bar:
            pbar_inner = tqdm(
                total=len(time_steps),
                position=1,
                leave=False,
                desc=f"PLMS steps (batch {batch_count} of {max_batch_count})",
            )

        if constraint is not None:
            constraint_logger = ConstraintLogger(
                n_steps=self.n_steps,
                verbose=True,
            )
            constraint_logger.setup()

            constraint.initialize(
                self.encoder_model,
                self.latent_space_scaling_factor,
                self.n_steps,
                sequence_length,
            )

        old_eps = []

        # Denoise the initial latent
        for i, step in enumerate(time_steps):
            index = len(time_steps) - i - 1

            # Batch the timesteps
            ts = x.new_full((num_conformations,), step, dtype=torch.long)

            ts_next = torch.full(
                (num_conformations,),
                time_steps[min(i + 1, len(time_steps) - 1)],
                device=self.device,
                dtype=torch.long,
            )

            # Sample the generative process
            outs = self.p_sample_plms(
                x=x,
                c=labels,
                t=ts,
                attention_mask=attention_mask,
                index=index,
                old_eps=old_eps,
                t_next=ts_next,
                temperature=temperature,
            )

            x, _, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)

            # Apply custom constraint
            if constraint is not None and step != 0:
                x = constraint.apply(x, step, logger=constraint_logger)

            # update progress bar if we are showing it
            if show_per_step_progress_bar:
                pbar_inner.update(1)

        if constraint is not None:
            constraint_logger.close()

        # if we have progress bar, close after finishing the steps.
        if show_per_step_progress_bar:
            pbar_inner.close()

        # Scale the latents back to the original scale
        # x = x * self.ddpm_model.latent_space_std + self.ddpm_model.latent_space_mean

        x = x * (1 / self.ddpm_model.latent_space_scaling_factor)

        # Decode the latents to get the distance maps
        x = self.encoder_model.decode(x)

        return x

    @torch.no_grad()
    def p_sample_plms(
        self,
        x,
        c,
        t,
        attention_mask,
        index,
        temperature=1.0,
        old_eps=None,
        t_next=None,
    ):
        b, *_, device = *x.shape, x.device

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * torch.randn(x.shape, device=device) * temperature

            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = self.ddpm_model.model(x, t, c, attention_mask)

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = self.ddpm_model.model(x_prev, t_next, c, attention_mask)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (
                55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]
            ) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t
