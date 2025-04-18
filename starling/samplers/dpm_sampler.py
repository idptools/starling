from typing import Tuple

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from torch import nn, sqrt
from torch.special import expm1
from tqdm.auto import tqdm

from starling.data.data_wrangler import one_hot_encode


class DPMSampler(nn.Module):
    def __init__(self, ddpm_model, n_steps: int, sampler="full"):
        super(DPMSampler, self).__init__()
        self.ddpm_model = ddpm_model
        # Noise schedule used in the model
        self.log_snr = ddpm_model.log_snr
        self.sampler = sampler

        self.timesteps = torch.linspace(
            1.0, 0.0, n_steps + 1, device=self.ddpm_model.device
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
        labels = (
            torch.argmax(
                torch.from_numpy(one_hot_encode(labels.ljust(384, "0"))), dim=-1
            )
            .to(torch.int64)
            .squeeze()
            .to(self.ddpm_model.device)
        )

        labels = self.ddpm_model.sequence2labels(labels)

        labels = labels.unsqueeze(0)

        return labels

    def p_mean_variance(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        time_next: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (
            (-log_snr).sigmoid(),
            (-log_snr_next).sigmoid(),
        )

        alpha, sigma, alpha_next = map(
            sqrt, (squared_alpha, squared_sigma, squared_alpha_next)
        )

        batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
        pred_noise = self.ddpm_model.model(x, batch_log_snr, labels)

        model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)
        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    def p_sample(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        time_next: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_mean, posterior_variance = self.p_mean_variance(
            x, time, time_next, labels
        )

        if time == 0:
            return model_mean

        noise = torch.randn_like(x)
        x = model_mean + sqrt(posterior_variance) * noise

        return x, noise

    @torch.no_grad()
    def full_ddpm_loop(
        self,
        num_conformations: int,
        labels: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        show_per_step_progress_bar: bool = True,
        batch_count: int = 1,
        max_batch_count: int = 1,
    ) -> torch.Tensor:
        device = self.ddpm_model.device

        # Initialize the latents with noise
        x = torch.randn(
            [
                num_conformations,
                self.ddpm_model.in_channels,
                self.ddpm_model.image_size,
                self.ddpm_model.image_size,
            ],
            device=device,
        )

        # Get the labels to condition the generative process on
        labels = self.generate_labels(labels)

        # initialize progress bar if we want to show it
        if show_per_step_progress_bar:
            pbar_inner = tqdm(
                total=len(self.timesteps),
                position=1,
                leave=False,
                desc=f"DDPM steps (batch {batch_count} of {max_batch_count})",
            )

        # Denoise the initial latent
        for i, step in enumerate(self.timesteps):
            time = self.timesteps[i]
            time_next = self.timesteps[i + 1]

            # Sample the generative process
            x, *_ = self.p_sample(
                x,
                time,
                time_next,
                labels,
            )
            # update progress bar if we are showing it
            if show_per_step_progress_bar:
                pbar_inner.update(1)

        # if we have progress bar, close after finishing the steps.
        if show_per_step_progress_bar:
            pbar_inner.close()

        # Scale the latents back to the original scale
        x = (1 / self.ddpm_model.latent_space_scaling_factor) * x

        # Decode the latents to get the distance maps
        x = self.ddpm_model.encoder_model.decode(x)

        return x
