from typing import Tuple

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from starling.data.data_wrangler import one_hot_encode
from starling.inference.constraints import (
    ConstraintLogger,
    DistanceConstraint,
    HelicityConstraint,
    RgConstraint,
)
from starling.utilities import helix_dm


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
        with torch.inference_mode():
            model_labels = self.ddpm_model.sequence2labels(sequence_indices)

        # Step 5: Add batch dimension
        batched_labels = model_labels.unsqueeze(0)

        return batched_labels

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

    def p_sample_loop(
        self,
        shape: tuple,
        labels: torch.Tensor,
        return_all_timesteps: bool = False,
        constraint=None,
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

        sequence_length = len(labels)

        # Convert sequence to model-compatible label format
        model_labels = self.generate_labels(labels)

        # Initialize noise for latent representation
        latents = torch.randn(shape, device=device)

        # Track denoising trajectory if requested
        denoising_trajectory = [] if return_all_timesteps else None

        # Use all timesteps
        timesteps = torch.arange(0, self.n_steps)

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

        # Reverse timesteps to go from noisy to clean
        for timestep in tqdm(
            reversed(timesteps),
            desc="Denoising latents",
            total=len(timesteps),
            position=0,
        ):
            # Store current state if tracking trajectory
            if return_all_timesteps:
                denoising_trajectory.append(latents)

            # Use inference mode only for the model's prediction step
            with torch.inference_mode():
                # Perform single denoising step
                latents = self.p_sample(latents, timestep, model_labels)

            # Apply custom constraint
            if constraint is not None and timestep != 0:
                latents = constraint.apply(latents, timestep, logger=constraint_logger)

        if constraint is not None:
            constraint_logger.close()

        # Scale latents back to original range
        scaled_latents = latents / self.latent_space_scaling_factor

        # Return appropriate result based on tracking option
        if return_all_timesteps:
            return (
                torch.stack(denoising_trajectory, dim=1)
                / self.latent_space_scaling_factor
            )

        return scaled_latents

    def sample(
        self,
        num_conformations: int,
        labels: torch.Tensor,
        show_per_step_progress_bar: bool = True,
        batch_count: int = 1,
        max_batch_count: int = 1,
        return_all_timesteps: bool = False,
        constraint=None,
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

        # Generate latent representations through the denoising process
        latents = self.p_sample_loop(
            shape=latent_shape,
            labels=labels,
            return_all_timesteps=return_all_timesteps,
            constraint=constraint,
        )

        # Convert latent representations to distance maps
        distance_maps = self.encoder_model.decode(latents)

        return distance_maps

    def helix_guidance_loss(self, D_pred, start=50, end=150):
        helix_pairs = [(1, 3.8), (2, 5.4), (3, 6.0), (4, 5.4)]
        batch_size = D_pred.shape[0]
        D_pred = D_pred.squeeze()

        # Initialize per-sample losses
        per_sample_loss = torch.zeros(batch_size, device=D_pred.device)
        count = 0

        for i in range(start, end):
            for offset, target in helix_pairs:
                j = i + offset
                if j <= end:
                    dist = D_pred[:, i, j]  # (B,) distance values
                    per_sample_loss += (dist - target) ** 2  # Add to per-sample loss
                    count += 1

        # Normalize by count for each sample
        if count > 0:
            per_sample_loss = per_sample_loss / count

        # For backward compatibility, also return mean loss
        total_loss = per_sample_loss.mean()

        return total_loss, per_sample_loss

    def cosine_weight(self, timestep, total_steps=1000, s=0.008):
        t = timestep / total_steps
        f_t = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        return f_t

    def sample_with_helicity_constraint(
        self,
        num_conformations: int,
        labels: str,
        show_per_step_progress_bar: bool = True,
        batch_count: int = 1,
        max_batch_count: int = 1,
        return_all_timesteps: bool = False,
        resid_start: int = 40,
        resid_end: int = 80,
        constraint_weight: float = 0.5,
        guidance_scale: float = 1.0,
        guidance_schedule: str = "cosine",
        guidance_start_timestep: int = 50,
        verbose: bool = False,
    ):
        """
        Sample conformations with gradient-based helicity guidance in a specified region.

        Parameters
        ----------
        num_conformations : int
            The number of conformations to sample
        labels : str
            The protein sequence to condition on
        show_per_step_progress_bar : bool, optional
            Whether to show progress bar, by default True
        batch_count : int, optional
            Current batch count for logging, by default 1
        max_batch_count : int, optional
            Maximum batch count for logging, by default 1
        return_all_timesteps : bool, optional
            Whether to return the full trajectory of denoising, by default False
        resid_start, resid_end : int
            The start and end residue indices for the helical constraint
        constraint_weight : float
            Strength of the helicity constraint
        guidance_scale : float
            Scale factor for gradient guidance strength
        guidance_schedule : str, optional
            Type of guidance schedule to use ("linear" or "cosine"), by default "linear"
        guidance_start_timestep : int, optional
            Timestep to start applying guidance, by default 50
        verbose : bool, optional
            Whether to print debug information, by default False

        Returns
        -------
        torch.Tensor
            Distance maps representing protein conformations with helical regions
        """
        # Setup latent space and labels
        latent_shape = (
            num_conformations,
            self.in_channels,
            self.image_size,
            self.image_size,
        )
        model_labels = self.generate_labels(labels)
        latents = torch.randn(latent_shape, device=self.device)

        # Generate reference helix distance map
        helix_ref = torch.from_numpy(helix_dm(L=384)).to(self.device)

        # Create mask for the constrained region - upper triangular part only
        mask = torch.zeros((384, 384), device=self.device)
        mask[resid_start:resid_end, resid_start:resid_end] = torch.triu(
            torch.ones((resid_end - resid_start, resid_end - resid_start)), diagonal=1
        )

        # Create weights inversely proportional to the reference distances
        weights = 1.0 / (helix_ref + 1e-2)

        # Setup for tracking denoising trajectory
        denoising_trajectory = [] if return_all_timesteps else None

        # Process all timesteps in reverse order (from noisy to clean)
        for timestep in tqdm(
            reversed(range(self.n_steps)),
            desc="Denoising with helicity guidance",
            total=self.n_steps,
            disable=not show_per_step_progress_bar,
        ):
            # Store current state if tracking trajectory
            if return_all_timesteps:
                denoising_trajectory.append(latents.clone())

            # Perform standard denoising step
            latents = self.p_sample(latents, timestep, model_labels)

            # Apply gradient-based guidance at specific timesteps
            if timestep < guidance_start_timestep:
                latents = self._apply_helicity_guidance(
                    latents=latents,
                    helix_ref=helix_ref,
                    mask=mask,
                    weights=weights,
                    timestep=timestep,
                    constraint_weight=constraint_weight,
                    guidance_scale=guidance_scale,
                    guidance_schedule=guidance_schedule,
                    verbose=verbose,
                )

        # Final step: Get the distance maps from the guided latents
        scaled_latents = latents.detach() / self.latent_space_scaling_factor
        distance_maps = self.encoder_model.decode(scaled_latents)

        # Return appropriate result based on tracking option
        if return_all_timesteps:
            trajectory_latents = (
                torch.stack(denoising_trajectory, dim=1)
                / self.latent_space_scaling_factor
            )
            return self.encoder_model.decode(trajectory_latents)

        return distance_maps

    def _apply_helicity_guidance(
        self,
        latents: torch.Tensor,
        helix_ref: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor,
        timestep: int,
        constraint_weight: float,
        guidance_scale: float,
        guidance_schedule: str = "cosine",
        verbose: bool = False,
    ) -> torch.Tensor:
        """Apply gradient-based helicity guidance to the latent vectors.

        Parameters
        ----------
        latents : torch.Tensor
            Current latent vectors to update
        helix_ref : torch.Tensor
            Reference helix distance map
        mask : torch.Tensor
            Mask for the constrained region
        weights : torch.Tensor
            Per-position weight factors
        timestep : int
            Current timestep
        constraint_weight : float
            Overall strength of the helicity constraint
        guidance_scale : float
            Scale factor for guidance strength
        guidance_schedule : str, optional
            Type of schedule to use, by default "linear"
        verbose : bool, optional
            Whether to print debug info, by default False

        Returns
        -------
        torch.Tensor
            Updated latent vectors
        """
        with torch.inference_mode(False):
            # Create a fresh copy with gradient tracking
            latents_copy = latents.clone().requires_grad_(True)

            # Decode to get current distance maps
            scaled_latents = latents_copy / self.latent_space_scaling_factor
            distance_maps = self.encoder_model.decode(scaled_latents)

            # Calculate loss in the target region
            region_loss = ((distance_maps - helix_ref) ** 2) * weights * mask
            normalization_factor = (weights * mask).sum()
            per_batch_loss = region_loss.sum(dim=(1, 2, 3)) / normalization_factor

            # Get total loss and compute gradients
            total_loss = per_batch_loss.mean()
            base_grad = torch.autograd.grad(total_loss, latents_copy)[0]

            # Determine time-dependent guidance strength
            if guidance_schedule == "cosine":
                time_scale = self.cosine_weight(
                    timestep, total_steps=self.n_steps, s=0.008
                )
            else:  # linear or any other default
                time_scale = 1.0 - (timestep / self.n_steps)

            if verbose:
                print(
                    f"Time scale: {time_scale:.4f}, Loss: {total_loss.item():.4f}, "
                    f"Gradient norm: {base_grad.norm().item():.4f}"
                )

            # Calculate per-sample loss scaling (vectorized)
            # Each conformation gets scaled by how much its loss differs from the mean
            loss_scale = per_batch_loss / per_batch_loss.mean()

            # Match dimensions for broadcasting
            loss_scale = loss_scale.view(-1, 1, 1, 1)

            # Apply all scaling factors at once
            scaled_update = (
                -guidance_scale
                * constraint_weight
                * time_scale
                * loss_scale
                * base_grad
            )

            # Clip gradient update if too large
            update_norm = scaled_update.norm().item()
            if update_norm > 1.0:
                scaled_update = scaled_update * (1.0 / update_norm)

            # Apply the gradient update
            return latents + scaled_update.detach()
