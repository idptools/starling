import math
import time
from abc import ABC
from typing import Tuple

import torch
from tqdm import tqdm

from starling.utilities import helix_dm


class Constraint(ABC):
    def __init__(self, encoder_model, latent_space_scaling_factor, n_steps):
        """Initialize base constraint with common parameters.

        Parameters
        ----------
        encoder_model : nn.Module
            The encoder model used to decode latents into distance maps
        latent_space_scaling_factor : float
            Scaling factor for latent space
        n_steps : int
            Total number of diffusion steps
        """
        self.encoder_model = encoder_model
        self.latent_space_scaling_factor = latent_space_scaling_factor
        self.n_steps = n_steps

    def cosine_weight(self, t, total_steps, s=0.008):
        """Cosine schedule for time-dependent guidance strength."""
        t_scaled = t / total_steps
        return math.cos(t_scaled * math.pi / 2) ** 2

    def compute_loss(
        self, distance_maps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for this constraint without applying gradients.

        Parameters
        ----------
        distance_maps : torch.Tensor
            Pre-computed distance maps from the latents

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (per_batch_loss, total_loss) - Individual sample losses and mean loss
        """
        raise NotImplementedError("Subclasses should implement compute_loss")

    def apply(self, latents: torch.Tensor, timestep: int) -> torch.Tensor:
        """Apply the constraint to the given latents."""
        with torch.inference_mode(False):
            latents_copy = latents.clone().requires_grad_(True)
            scaled_latents = latents_copy / self.latent_space_scaling_factor
            distance_maps = self.encoder_model.decode(scaled_latents)

            # Get per-sample losses and total loss
            per_batch_loss, loss = self.compute_loss(distance_maps)

            # Compute gradients
            base_grad = torch.autograd.grad(loss, latents_copy)[0]

            # Get time-dependent scaling
            time_scale = self.get_time_scale(timestep)

            # Calculate per-sample loss scaling
            loss_scale = per_batch_loss / per_batch_loss.mean()
            loss_scale = loss_scale.view(-1, 1, 1, 1)

            base_grad_norm = base_grad.norm().item()
            print(f"[{self.__class__.__name__}] Timestep: {timestep}/{self.n_steps}")
            print(f"  - Loss: {loss.item():.6f}")
            print(f"  - Base gradient norm: {base_grad_norm:.6f}")
            print(f"  - Time scale: {time_scale:.4f}")
            print(
                f"  - Min/Max loss scale: {loss_scale.min().item():.4f}/{loss_scale.max().item():.4f}"
            )

            # Apply all scaling factors
            scaled_update = (
                -self.guidance_scale
                * self.constraint_weight
                * time_scale
                * loss_scale
                * base_grad
            )

            # Clip gradient update if too large
            update_norm = scaled_update.norm().item()
            if update_norm > 1.0:
                scaled_update = scaled_update * (1.0 / update_norm)

            return latents + scaled_update.detach()

    def get_time_scale(self, timestep: int) -> float:
        """Get the time-dependent scaling factor."""
        if self.schedule == "cosine":
            return self.cosine_weight(timestep, total_steps=self.n_steps)
        else:
            return 1.0 - (timestep / self.n_steps)


class HelicityConstraint(Constraint):
    def __init__(
        self,
        encoder_model,
        latent_space_scaling_factor,
        resid_start,
        resid_end,
        n_steps,
        constraint_weight,
        guidance_scale,
        schedule="cosine",
        verbose=False,
    ):
        super().__init__(encoder_model, latent_space_scaling_factor, n_steps)
        self.helix_ref = torch.from_numpy(helix_dm(L=384)).to(encoder_model.device)

        # Create mask for the constrained region - upper triangular part only
        self.mask = torch.zeros((384, 384), device=encoder_model.device)
        self.mask[resid_start:resid_end, resid_start:resid_end] = torch.triu(
            torch.ones((resid_end - resid_start, resid_end - resid_start)), diagonal=1
        )

        # Create weights inversely proportional to the reference distances
        self.weights = 1.0 / (self.helix_ref + 1e-2)

        self.constraint_weight = constraint_weight
        self.guidance_scale = guidance_scale
        self.schedule = schedule
        self.verbose = verbose

    def compute_loss(self, distance_maps: torch.Tensor) -> torch.Tensor:
        # Calculate loss in the target region

        region_loss = ((distance_maps - self.helix_ref) ** 2) * self.weights * self.mask
        normalization_factor = (self.weights * self.mask).sum()
        per_batch_loss = region_loss.sum(dim=(1, 2, 3)) / normalization_factor

        # Return mean loss
        return per_batch_loss, per_batch_loss.mean()


class DistanceConstraint(Constraint):
    def __init__(
        self,
        encoder_model,
        latent_space_scaling_factor,
        n_steps,
        resid1,
        resid2,
        target_distances,
        constraint_weight,
        guidance_scale,
        schedule="cosine",
    ):
        super().__init__(encoder_model, latent_space_scaling_factor, n_steps)
        self.target_distances = target_distances
        self.constraint_weight = constraint_weight
        self.guidance_scale = guidance_scale
        self.schedule = schedule

        self.resid1 = resid1
        self.resid2 = resid2

    def compute_loss(self, distance_maps: torch.Tensor) -> torch.Tensor:
        # Calculate loss in the target region
        per_batch_loss = (
            distance_maps[:, :, self.resid1, self.resid2] - self.target_distances
        ) ** 2

        # Return mean loss
        return per_batch_loss, per_batch_loss.mean()


class RgConstraint(Constraint):
    def __init__(
        self,
        encoder_model,
        latent_space_scaling_factor,
        n_steps,
        target_distances,
        mask,
        weights,
        constraint_weight,
        guidance_scale,
        schedule="cosine",
    ):
        super().__init__(encoder_model, latent_space_scaling_factor, n_steps)
        self.target_distances = target_distances
        self.mask = mask
        self.weights = weights
        self.constraint_weight = constraint_weight
        self.guidance_scale = guidance_scale
        self.schedule = schedule


class ReConstraint(Constraint):
    def __init__(
        self,
        encoder_model,
        latent_space_scaling_factor,
        n_steps,
        target_distances,
        mask,
        weights,
        constraint_weight,
        guidance_scale,
        schedule="cosine",
    ):
        super().__init__(encoder_model, latent_space_scaling_factor, n_steps)
        self.target_distances = target_distances
        self.mask = mask
        self.weights = weights
        self.constraint_weight = constraint_weight
        self.guidance_scale = guidance_scale
        self.schedule = schedule


class MultiConstraint(Constraint):
    """Combines multiple constraints into a single optimization step."""

    def __init__(
        self,
        encoder_model,
        latent_space_scaling_factor,
        n_steps,
        constraints,
        constraint_weights=None,
        verbose=False,
    ):
        """
        Parameters
        ----------
        encoder_model : nn.Module
            The encoder model used to decode latents
        latent_space_scaling_factor : float
            Scaling factor for latent space
        n_steps : int
            Total number of diffusion steps
        constraints : list
            List of constraint objects to combine
        constraint_weights : list, optional
            Relative weights for each constraint (defaults to equal weights)
        verbose : bool, optional
            Whether to print debug info
        """
        super().__init__(encoder_model, latent_space_scaling_factor, n_steps)
        self.constraints = constraints

        # Set default weights if not provided
        if constraint_weights is None:
            self.constraint_weights = [1.0] * len(constraints)
        else:
            self.constraint_weights = constraint_weights

        self.verbose = verbose

    def apply(self, latents: torch.Tensor, timestep: int) -> torch.Tensor:
        with torch.inference_mode(False):
            # Create a copy with gradient tracking
            latents_copy = latents.clone().requires_grad_(True)

            # Decode latents once to get distance maps
            scaled_latents = latents_copy / self.latent_space_scaling_factor
            distance_maps = self.encoder_model.decode(scaled_latents)

            # Compute loss for each constraint
            total_loss = 0.0
            losses = []

            for i, (constraint, weight) in enumerate(
                zip(self.constraints, self.constraint_weights)
            ):
                # Each constraint should have a compute_loss method instead of applying gradients directly
                constraint_loss = constraint.compute_loss(
                    latents_copy, distance_maps, timestep
                )
                weighted_loss = weight * constraint_loss
                total_loss += weighted_loss
                losses.append(constraint_loss.item())

            # Compute gradients once for the combined loss
            base_grad = torch.autograd.grad(total_loss, latents_copy)[0]

            # Determine time-dependent guidance strength (could be constraint-specific)
            time_scale = self.cosine_weight(timestep, total_steps=self.n_steps)

            if self.verbose:
                constraint_names = [type(c).__name__ for c in self.constraints]
                loss_info = ", ".join(
                    [
                        f"{name}: {loss:.4f}"
                        for name, loss in zip(constraint_names, losses)
                    ]
                )
                print(
                    f"Time step: {timestep}, {loss_info}, Combined loss: {total_loss.item():.4f}"
                )

            # Apply all scaling factors at once
            scaled_update = -time_scale * base_grad

            # Clip gradient update if too large
            update_norm = scaled_update.norm().item()
            if update_norm > 1.0:
                scaled_update = scaled_update * (1.0 / update_norm)

            # Apply the gradient update
            return latents + scaled_update.detach()


class ConstraintLogger:
    """Logger that updates constraint metrics in-place using tqdm."""

    def __init__(self, n_steps, verbose=True, update_freq=1):
        """Initialize logger.

        Parameters
        ----------
        n_steps : int
            Total number of steps
        verbose : bool
            Whether to log
        update_freq : int
            How often to update display (in steps)
        """
        self.n_steps = n_steps
        self.verbose = verbose
        self.update_freq = update_freq
        self.constraint_data = {}
        self.progress_bar = None
        self.start_time = None

    def setup(self):
        """Set up the progress bar."""
        if self.verbose:
            self.progress_bar = tqdm(
                total=self.n_steps,
                desc="Applying constraints",
                position=1,  # Position below main progress bar
                leave=False,  # Don't leave the progress bar
            )
            self.start_time = time.time()

    def update(self, timestep, constraint_name, metrics):
        """Update logger with new constraint metrics."""
        if not self.verbose:
            return

        # Store most recent data
        self.constraint_data[constraint_name] = metrics

        # Update progress bar every update_freq steps or at the end
        if timestep % self.update_freq == 0 or timestep == self.n_steps - 1:
            # Create status message
            status_parts = []

            # Add elapsed time
            elapsed = time.time() - self.start_time
            status_parts.append(f"elapsed: {elapsed:.1f}s")

            # Add constraint info
            for name, data in self.constraint_data.items():
                loss = data.get("loss", 0)
                grad_norm = data.get("grad_norm", 0)
                update_norm = data.get("update_norm", 0)
                status_parts.append(
                    f"{name[:3]} loss: {loss:.4f} grad: {grad_norm:.2f}"
                )

            # Update progress bar
            self.progress_bar.set_postfix_str(" | ".join(status_parts))
            self.progress_bar.update(self.update_freq)

    def close(self):
        """Close the logger."""
        if self.verbose and self.progress_bar is not None:
            self.progress_bar.close()
