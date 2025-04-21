import sys
from typing import Tuple

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from starling.data.data_wrangler import one_hot_encode


def extract(
    constants: torch.Tensor, timestamps: torch.Tensor, shape: int
) -> torch.Tensor:
    """
    Extract values from a tensor based on given timestamps.

    Parameters
    ----------
    constants : torch.Tensor
        The tensor to extract values from.
    timestamps : torch.Tensor
        A 1D tensor containing the indices for extraction.
    shape : int
        The desired shape of the output tensor.

    Returns
    -------
    torch.Tensor
        The tensor with extracted values.
    """
    batch_size = timestamps.shape[0]
    out = constants.gather(-1, timestamps)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(timestamps.device)


class DDPMSampler(nn.Module):
    def __init__(self, ddpm_model):
        super(DDPMSampler, self).__init__()
        self.ddpm_model = ddpm_model
        self.n_steps = self.ddpm_model.num_timesteps
        self.in_channels = ddpm_model.in_channels
        self.image_size = ddpm_model.image_size

        self.encoder_model = ddpm_model.encoder_model
        self.device = ddpm_model.device

        self.alpha_bar = self.ddpm_model.alphas_cumprod
        self.betas = self.ddpm_model.betas
        self.sqrt_recip_alphas = self.ddpm_model.sqrt_recip_alphas
        self.sqrt_one_minus_alphas_cumprod = (
            self.ddpm_model.sqrt_one_minus_alphas_cumprod
        )
        self.posterior_variance = self.ddpm_model.posterior_variance
        self.latent_space_scaling_factor = self.ddpm_model.latent_space_scaling_factor

    def generate_labels(self, sequence: str) -> torch.Tensor:
        """
        Generate labels to condition the generative process on.

        Parameters
        ----------
        sequence : str
            A sequence to generate labels from.

        Returns
        -------
        torch.Tensor
            The labels to condition the generative process on.
        """
        # Constants
        PADDING_LENGTH = 384
        PADDING_CHAR = "0"

        # Step 1: Pad the sequence to fixed length
        padded_sequence = sequence.ljust(PADDING_LENGTH, PADDING_CHAR)

        # Step 2: Convert to one-hot encoding and transform to tensor
        one_hot = torch.from_numpy(one_hot_encode(padded_sequence))

        # Step 3: Get indices of maximum values and move to correct device/dtype
        sequence_indices = (
            torch.argmax(one_hot, dim=-1)
            .to(torch.int64)
            .squeeze()
            .to(self.ddpm_model.device)
        )

        # Step 4: Convert sequence indices to model-specific label format
        model_labels = self.ddpm_model.sequence2labels(sequence_indices)

        # Step 5: Add batch dimension
        batched_labels = model_labels.unsqueeze(0)

        return batched_labels

    @torch.inference_mode()
    def p_sample(
        self, x: torch.Tensor, timestamp: int, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        One denoising step of the diffusion model. This function is
        used in p_sample_loop to denoise the initial tensor sampled from N(0, I)

        Parameters
        ----------
        x : torch.Tensor
            A tensor to denoise
        timestamp : int
            The timestep of the denoising-diffusion process to denoise
        labels : torch.Tensor
            Labels (sequences) to condition the model on

        Returns
        -------
        torch.Tensor
            Returns the denoised tensor (t-1)
        """
        b, *_, device = *x.shape, x.device

        # Batch the timestep to the same size as the input tensor x
        batched_timestamps = torch.full(
            (b,), timestamp, device=device, dtype=torch.long
        )

        # Run the model to predict the noise at the current timestamp
        preds = self.ddpm_model.model(x, batched_timestamps, labels)

        # Extract the necessary values from the buffers to calculate the predicted mean
        betas_t = extract(self.betas, batched_timestamps, x.shape)
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas, batched_timestamps, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, batched_timestamps, x.shape
        )

        # Calculate the predicted mean based on the model prediction of the noise
        predicted_mean = sqrt_recip_alphas_t * (
            x - betas_t * preds / sqrt_one_minus_alphas_cumprod_t
        )

        # If the timestamp is 0, return the predicted mean
        if timestamp == 0:
            return predicted_mean
        else:
            posterior_variance = extract(
                self.posterior_variance, batched_timestamps, x.shape
            )
            noise = torch.randn_like(x)
            return predicted_mean + torch.sqrt(posterior_variance) * noise

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape: tuple,
        labels: torch.Tensor,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """
        Sampling loop for the diffusion model. It loops over the timesteps
        to denoise the initial tensor sampled from N(0, I)

        Parameters
        ----------
        shape : tuple
            The shape of the tensor to sample from N(0, I)
        labels : torch.Tensor
            Labels to condition the sampling on
        steps : int, optional
            Number of steps to sample, by default None (all timesteps)
        return_all_timesteps : bool, optional
            Whether to return the full trajectory of denoising,
            by default False

        Returns
        -------
        torch.Tensor
            Returns the fully denoised tensor, in this case a latent space
            that can be decoded using a pre-trained VAE
        """
        batch_size, device = shape[0], self.device

        # Initialize noise for latent representation
        latents = torch.randn(shape, device=device)

        # Track denoising trajectory if requested
        denoising_trajectory = [] if return_all_timesteps else None

        # Use all timesteps
        timesteps = torch.arange(0, self.n_steps)

        # Reverse timesteps to go from noisy to clean
        for timestep in tqdm(
            reversed(timesteps),
            desc="Denoising latents",
            total=len(timesteps),
        ):
            # Store current state if tracking trajectory
            if return_all_timesteps:
                denoising_trajectory.append(latents)

            # Perform single denoising step
            latents = self.p_sample(latents, timestep, labels)

        # Scale latents back to original range
        scaled_latents = latents / self.latent_space_scaling_factor

        # Return appropriate result based on tracking option
        if return_all_timesteps:
            return (
                torch.stack(denoising_trajectory, dim=1)
                / self.latent_space_scaling_factor
            )

        return scaled_latents

    @torch.inference_mode()
    def sample(
        self,
        num_conformations: int,
        labels: torch.Tensor,
        show_per_step_progress_bar: bool = True,
        batch_count: int = 1,
        max_batch_count: int = 1,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """
        Sample conformations from the trained diffusion model.

        Parameters
        ----------
        num_conformations : int
            The batch size to sample, higher the better if enough VRAM
        labels : str
            Sequence to condition the sampling on
        return_all_timesteps : bool, optional
            Whether to return all the tensors along the denoising trajectory, by default False

        Returns
        -------
        torch.Tensor
            Distance map representing protein conformations
        """
        # Define the shape for latent space sampling
        latent_shape = (
            num_conformations,
            self.in_channels,
            self.image_size,
            self.image_size,
        )

        # Convert sequence to model-compatible label format
        model_labels = self.generate_labels(labels)

        # Generate latent representations through the denoising process
        latents = self.p_sample_loop(
            shape=latent_shape,
            labels=model_labels,
            return_all_timesteps=return_all_timesteps,
        )

        # Convert latent representations to distance maps
        distance_maps = self.encoder_model.decode(latents)

        return distance_maps
