import pdb
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from torch import nn, sqrt
from torch.special import expm1
from tqdm.auto import tqdm, trange

from starling.data.data_wrangler import one_hot_encode


class DPMSampler(nn.Module):
    """
    Denoising Probabilistic Model (DPM) Sampler for generating protein conformations.

    This class implements the sampling process for a pre-trained diffusion model,
    allowing generation of new protein structures conditioned on sequence information.
    """

    def __init__(
        self,
        ddpm_model: nn.Module,
        n_steps: int,
        sampler: Literal["full", "ddim", "plms"] = "full",
        max_sequence_length: int = 384,
        time="continuous",
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
            Sampling strategy to use ("full", "ddim", "plms")
        max_sequence_length : int, default=384
            Maximum sequence length to support
        """
        super(DPMSampler, self).__init__()
        self.ddpm_model = ddpm_model
        self.device = ddpm_model.device
        self.log_snr = ddpm_model.log_snr  # Noise schedule used in the model
        self.sampler = sampler
        self.max_sequence_length = max_sequence_length
        self.time = time

        # Create timestep schedule from 1.0 to 0.0
        self.n_steps = n_steps

        # Define mapping between sampling strategies and methods
        self.sampling_strategies = {
            "full": self._full_sample,
            # Add other sampling strategies as they are implemented
            # "ddim": self._ddim_sample,
            # "plms": self._plms_sample,
        }

        if self.sampler not in self.sampling_strategies:
            raise ValueError(
                f"Sampling strategy '{sampler}' not supported. Use one of: {list(self.sampling_strategies.keys())}"
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

        Raises
        ------
        ValueError
            If sequence contains invalid characters
        """
        padded_sequence = sequence.ljust(self.max_sequence_length, "0")

        # One-hot encode the sequence
        one_hot = torch.from_numpy(one_hot_encode(padded_sequence))

        # Convert to indices and move to device
        sequence_indices = torch.argmax(one_hot, dim=-1).to(torch.int64).squeeze()
        sequence_indices = sequence_indices.to(self.device)

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
        if self.time == "discrete":
            timestep = (1000 - 1) * time / 1
            pdb.set_trace()
            pred_noise = self.ddpm_model.model(x, timestep, labels)
        if self.time == "continuous":
            pred_noise = self.ddpm_model.model(x, batch_log_snr, labels)
        else:
            raise ValueError(
                f"Unknown time type: {self.time}. Supported types are 'discrete' and 'continuous'."
            )

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
        if time_next == 0:
            return model_mean

        # Add noise scaled by the posterior variance
        noise = torch.randn_like(x)
        noised_mean = model_mean + posterior_variance.sqrt() * noise

        return noised_mean

    def _full_sample(
        self,
        latents: torch.Tensor,
        conditioning_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a full sampling step.

        Parameters
        ----------
        latents : torch.Tensor
            Current noisy samples
        time_next : torch.Tensor
            Next timestep
        conditioning_labels : torch.Tensor
            Conditioning labels

        Returns
        -------
        torch.Tensor
            Denoised sample
        """

        steps = torch.linspace(1.0, 0.0, self.n_steps + 1, device=self.device)
        # Iteratively denoise the latents
        for i in trange(
            self.n_steps, desc="sampling loop time step", total=self.n_steps
        ):
            time = steps[i]
            time_next = steps[i + 1]

            latents = self.p_sample(
                latents,
                time,
                time_next,
                conditioning_labels,
            )

        return latents

    @torch.no_grad()
    def sample(
        self,
        num_conformations: int,
        labels: str,
        latent_shape: Tuple[int, int] = (24, 24),
    ) -> torch.Tensor:
        """
        Generate protein conformations by running the complete denoising process.

        This method performs the following steps:
        1. Validate input parameters
        2. Generate initial random noise
        3. Convert protein sequence to conditioning labels
        4. Apply the selected sampling strategy to denoise the latents
        5. Scale and decode the latents to produce distance maps

        Parameters
        ----------
        num_conformations : int
            Number of protein conformations to generate
        labels : str
            Protein sequence to condition the generation on
        latent_shape : Tuple[int, int], default=(24, 24)
            Shape of the latent space (height, width)

        Returns
        -------
        torch.Tensor
            Generated protein conformations as distance maps

        Raises
        ------
        ValueError
            If num_conformations is not positive
        """
        # Validate inputs
        if num_conformations <= 0:
            raise ValueError(
                f"num_conformations must be positive, got {num_conformations}"
            )

        # Step 1: Initialize with random noise
        initial_noise = torch.randn(
            [num_conformations, 1, latent_shape[0], latent_shape[1]],
            device=self.device,
        )

        # Step 2: Process input sequence to get conditioning labels
        sequence_conditioning = self.generate_labels(labels)

        # Step 3: Apply the selected sampling strategy to denoise
        # This gradually transforms noise into a meaningful latent representation
        denoised_latents = self.sampling_strategies[self.sampler](
            initial_noise,
            sequence_conditioning,
        )

        # Step 4: Rescale the latents to the original scale
        # This undoes the normalization applied during training
        scaled_latents = denoised_latents / self.ddpm_model.latent_space_scaling_factor

        # Step 5: Decode the latents to get the final distance maps
        # This transforms the latent representation into protein distance maps
        protein_distance_maps = self.ddpm_model.encoder_model.decode(scaled_latents)

        return protein_distance_maps
