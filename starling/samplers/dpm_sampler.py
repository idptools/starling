from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from torch import nn, sqrt
from torch.special import expm1
from tqdm.auto import tqdm

from starling.data.data_wrangler import one_hot_encode


class DPMSampler(nn.Module):
    """
    Denoising Probabilistic Model (DPM) Sampler for generating protein conformations.

    This class implements the sampling process for a pre-trained diffusion model,
    allowing generation of new protein structures conditioned on sequence information.
    """

    def __init__(
        self, ddpm_model: nn.Module, n_steps: int, sampler: Literal["full"] = "full"
    ) -> None:
        """
        Initialize the DPM sampler.

        Parameters
        ----------
        ddpm_model : nn.Module
            The pre-trained diffusion model
        n_steps : int
            Number of denoising steps
        sampler : str, default="full"
            Sampling strategy to use
        """
        super(DPMSampler, self).__init__()
        self.ddpm_model = ddpm_model
        self.log_snr = ddpm_model.log_snr  # Noise schedule used in the model
        self.sampler = sampler

        # Create timestep schedule from 1.0 to 0.0
        self.timesteps = torch.linspace(
            1.0, 0.0, n_steps + 1, device=self.ddpm_model.device
        )

    def generate_labels(self, sequence: str) -> torch.Tensor:
        """
        Generate labels to condition the generative process on.

        Parameters
        ----------
        sequence : str
            A protein sequence to generate labels from

        Returns
        -------
        torch.Tensor
            The labels to condition the generative process on
        """
        # Pad sequence to fixed length of 384
        padded_sequence = sequence.ljust(384, "0")

        # One-hot encode the sequence
        one_hot = torch.from_numpy(one_hot_encode(padded_sequence))

        # Convert to indices and move to device
        sequence_indices = torch.argmax(one_hot, dim=-1).to(torch.int64).squeeze()
        sequence_indices = sequence_indices.to(self.ddpm_model.device)

        # Convert sequence indices to model-specific labels
        labels = self.ddpm_model.sequence2labels(sequence_indices)

        # Add batch dimension
        labels = labels.unsqueeze(0)

        return labels

    def p_mean_variance(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        time_next: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate posterior mean and variance for the denoising step.

        Parameters
        ----------
        x : torch.Tensor
            Current noisy samples
        time : torch.Tensor
            Current timestep
        time_next : torch.Tensor
            Next timestep
        labels : torch.Tensor
            Conditioning labels

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Posterior mean and variance
        """
        # Get signal-to-noise ratios
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)

        # Coefficient for noise prediction
        c = -expm1(log_snr - log_snr_next)

        # Calculate alpha and sigma terms from log_snr
        squared_alpha = log_snr.sigmoid()
        squared_alpha_next = log_snr_next.sigmoid()
        squared_sigma = (-log_snr).sigmoid()
        squared_sigma_next = (-log_snr_next).sigmoid()

        # Get square roots for the update equation
        alpha = squared_alpha.sqrt()
        sigma = squared_sigma.sqrt()
        alpha_next = squared_alpha_next.sqrt()

        # Prepare log_snr for batch processing
        batch_size = x.shape[0]
        batch_log_snr = repeat(log_snr, " -> b", b=batch_size)

        # Predict noise using the model
        pred_noise = self.ddpm_model.model(x, batch_log_snr, labels)

        # Calculate posterior mean and variance
        model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)
        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    def p_sample(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        time_next: torch.Tensor,
        labels: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform a single sampling step.

        Parameters
        ----------
        x : torch.Tensor
            Current noisy samples
        time : torch.Tensor
            Current timestep
        time_next : torch.Tensor
            Next timestep
        labels : torch.Tensor
            Conditioning labels

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Denoised sample and noise (if not the final step), or just denoised sample
        """
        model_mean, posterior_variance = self.p_mean_variance(
            x, time, time_next, labels
        )

        # For the final step, just return the mean without adding noise
        if time == 0:
            return model_mean

        # Add noise scaled by the posterior variance
        noise = torch.randn_like(x)
        noised_mean = model_mean + posterior_variance.sqrt() * noise

        return noised_mean, noise

    @torch.no_grad()
    def full_ddpm_loop(
        self,
        num_conformations: int,
        labels: str,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        show_per_step_progress_bar: bool = True,
        batch_count: int = 1,
        max_batch_count: int = 1,
    ) -> torch.Tensor:
        """
        Run the complete denoising process to generate protein conformations.

        Parameters
        ----------
        num_conformations : int
            Number of conformations to generate
        labels : str
            Protein sequence to condition on
        repeat_noise : bool, default=False
            Whether to use the same noise across samples
        temperature : float, default=1.0
            Sampling temperature
        show_per_step_progress_bar : bool, default=True
            Whether to display a progress bar for denoising steps
        batch_count : int, default=1
            Current batch number (for progress display)
        max_batch_count : int, default=1
            Total number of batches (for progress display)

        Returns
        -------
        torch.Tensor
            Generated protein conformations as distance maps
        """
        device = self.ddpm_model.device

        # Initialize latents with random noise
        latents = torch.randn(
            [num_conformations, 1, 24, 24],
            device=device,
        )

        # Process input sequence to get conditioning labels
        conditioning_labels = self.generate_labels(labels)

        # Set up progress tracking if requested
        progress_bar = None
        if show_per_step_progress_bar:
            progress_bar = tqdm(
                total=len(self.timesteps) - 1,
                position=1,
                leave=False,
                desc=f"DDPM steps (batch {batch_count} of {max_batch_count})",
            )

        # Iteratively denoise the latents
        for i in range(len(self.timesteps) - 1):
            time = self.timesteps[i]
            time_next = self.timesteps[i + 1]

            # Perform a single denoising step
            latents, _ = self.p_sample(
                latents,
                time,
                time_next,
                conditioning_labels,
            )

            # Update progress display
            if progress_bar is not None:
                progress_bar.update(1)

        # Clean up progress bar
        if progress_bar is not None:
            progress_bar.close()

        # Rescale the latents to the original scale
        scaled_latents = latents / self.ddpm_model.latent_space_scaling_factor

        # Decode the latents to get the final distance maps
        distance_maps = self.ddpm_model.encoder_model.decode(scaled_latents)

        return distance_maps
