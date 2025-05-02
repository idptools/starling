import math
from abc import ABC
from typing import Tuple

import torch
from einops import rearrange
from tqdm import tqdm

from starling.utilities import helix_dm


class Constraint(ABC):
    def __init__(self, constraint_weight=1.0, schedule="cosine", verbose=True):
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

        # These will be set by the sampler
        self.encoder_model = None
        self.latent_space_scaling_factor = None
        self.n_steps = None
        self.device = None

        # User-controlled parameters
        self.constraint_weight = constraint_weight
        self.schedule = schedule
        self.verbose = verbose

    def _setup_constraint(self):
        """Set up constraint-specific resources."""
        pass  # Implemented by subclasses

    def initialize(
        self, encoder_model, latent_space_scaling_factor, n_steps, sequence_length
    ):
        """Called by the sampler to set model parameters."""
        self.encoder_model = encoder_model
        self.latent_space_scaling_factor = latent_space_scaling_factor
        self.n_steps = n_steps
        self.device = encoder_model.device
        self.sequence_length = sequence_length
        self._setup_constraint()
        return self

    def cosine_weight(self, t, total_steps, s=0.008):
        """Cosine schedule for time-dependent guidance strength."""
        t_scaled = t / total_steps
        return math.cos(t_scaled * math.pi / 2) ** 2

    def get_adaptive_clip_threshold(self, timestep):
        """Get an adaptive clipping threshold that follows a cosine schedule."""
        max_threshold = 2.0  # Maximum threshold at beginning
        min_threshold = 1.0  # Minimum threshold at end

        # Cosine decay from max_threshold to min_threshold
        fraction_complete = 1 - (timestep / self.n_steps)
        cosine_factor = math.cos(fraction_complete * math.pi / 2)
        return min_threshold + cosine_factor**2 * (max_threshold - min_threshold)

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

    def apply(self, latents: torch.Tensor, timestep: int, logger=None) -> torch.Tensor:
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
            loss_scale = rearrange(loss_scale, "b 1 -> b 1 1 1")

            # Now apply meaningful scaling
            update = -self.constraint_weight * time_scale * loss_scale * base_grad

            # Per-sample gradient norms
            grad_flat = rearrange(update, "b c h w -> b (c h w)")
            grad_norms = grad_flat.norm(dim=1, keepdim=True)

            # Compute per-sample clipping factors
            max_allowed_grad_norm = 1.0
            clip_factors = (max_allowed_grad_norm / (grad_norms + 1e-6)).clamp(max=1.0)
            clip_factors = rearrange(clip_factors, "b 1 -> b 1 1 1")

            # Apply clipping
            update = update * clip_factors

            # Log if logger is provided
            if logger is not None and self.verbose:
                logger.update(
                    timestep,
                    self.__class__.__name__,
                    {
                        "loss": loss.item(),
                        "grad_norm": update.norm().item(),
                        # "update_norm": update_norm,
                        "time_scale": time_scale,
                        "min_loss_scale": loss_scale.min().item(),
                        "max_loss_scale": loss_scale.max().item(),
                        # "clipped": update_norm > 1.0,
                    },
                )

            return latents + update.detach()

    def get_time_scale(self, timestep: int) -> float:
        """Get the time-dependent scaling factor."""
        if self.schedule == "cosine":
            return self.cosine_weight(timestep, total_steps=self.n_steps)
        else:
            return 1.0 - (timestep / self.n_steps)


class BondConstraint(Constraint):
    def __init__(
        self, bond_length=3.81, tolerance=0.0, force_constant=2.0, constraint_weight=1.0
    ):
        super().__init__(constraint_weight=constraint_weight)
        self.bond_length = bond_length
        self.tolerance = tolerance
        self.force_constant = force_constant

    def compute_loss(
        self, distance_maps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bond loss based on distance maps.
        This loss penalizes deviations from the ideal bond length of 3.81 Å
        and applies a flat-bottom potential for deviations beyond 1.0 Å.

        Parameters
        ----------
        distance_maps : torch.Tensor
            Pre-computed distance maps from the latents

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Per-batch loss and mean loss
        """

        distance_maps = distance_maps[
            :, :, : self.sequence_length, : self.sequence_length
        ].squeeze()

        # Take the one off diagonal
        bonds = torch.diagonal(distance_maps, offset=1, dim1=1, dim2=2)

        # Calculate deviation from the ideal bond length
        deviation = torch.abs(bonds - self.bond_length)

        # Apply flat-bottom: only penalize deviations beyond tolerance
        excess = torch.nn.functional.relu(deviation - self.tolerance)

        # Calculate harmonic potential for the excess deviation
        per_batch_loss = (0.5 * self.force_constant * excess**2).mean(dim=1)

        return per_batch_loss, per_batch_loss.mean()


class StericClashConstraint(Constraint):
    def __init__(
        self, steric_clash_definition=5.0, force_constant=2.0, constraint_weight=1.0
    ):
        super().__init__(constraint_weight=constraint_weight)
        self.steric_clash_definition = steric_clash_definition
        self.force_constant = force_constant

    def compute_loss(
        self, distance_maps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute steric clash loss based on distance maps.
        This loss penalizes distances below a certain threshold (default 5.0 Å)
        and applies a flat-bottom potential for distances below this threshold.

        Parameters
        ----------
        distance_maps : torch.Tensor
            Pre-computed distance maps from the latents

        Returns
        -------
        torch.Tensor
            Per-batch loss and mean loss
        """
        mask = torch.triu(
            torch.ones(
                self.sequence_length, self.sequence_length, device=distance_maps.device
            ),
            diagonal=2,
        )

        distance_maps = distance_maps[
            :, :, : self.sequence_length, : self.sequence_length
        ]

        # Calculate the deviation from steric_clash_definition (only when distances are smaller)
        deviation = torch.relu(self.steric_clash_definition - distance_maps)

        # Apply harmonic potential formula: 0.5 * force_constant * deviation^2
        steric_clash = 0.5 * self.force_constant * deviation**2

        # Apply mask to consider only upper triangle without diagonals
        steric_clash = steric_clash * mask

        # Sum across all residue pairs and normalize
        steric_clash = steric_clash.sum(dim=(1, 2, 3))
        normalization_factor = mask.sum()
        per_batch_loss = steric_clash / normalization_factor

        return per_batch_loss, per_batch_loss.mean()


class HelicityConstraint(Constraint):
    def __init__(
        self,
        resid_start,
        resid_end,
        tolerance=0.0,
        force_constant=2.0,
        constraint_weight=1.0,
    ):
        super().__init__(constraint_weight=constraint_weight)

        self.resid_start = resid_start
        self.resid_end = resid_end
        self.tolerance = tolerance
        self.force_constant = force_constant

        # These will be initialized when the model is available
        self.helix_ref = None
        self.mask = None
        self.weights = None

    def _setup_constraint(self):
        """Set up device-specific tensors."""
        if not self.encoder_model:
            return

        # Create helix reference
        self.helix_ref = torch.from_numpy(helix_dm(L=384)).to(self.device)

        # Create mask
        self.mask = torch.zeros((384, 384), device=self.device)
        self.mask[
            self.resid_start : self.resid_end, self.resid_start : self.resid_end
        ] = torch.triu(
            torch.ones(
                (self.resid_end - self.resid_start, self.resid_end - self.resid_start)
            ),
            diagonal=1,
        )

        # Create weights
        self.weights = 1.0 / (self.helix_ref + 1e-2)

    def compute_loss(
        self, distance_maps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate deviation from reference helix
        deviation = torch.abs(distance_maps - self.helix_ref)

        # Apply flat-bottom potential: only penalize deviations beyond tolerance
        excess = torch.nn.functional.relu(deviation - self.tolerance)

        # Calculate harmonic potential for the excess deviation
        region_loss = 0.5 * self.force_constant * (excess**2) * self.mask

        normalization_factor = self.mask.sum()
        per_batch_loss = region_loss.sum(dim=(2, 3)) / normalization_factor

        # Return per-batch and mean loss
        return per_batch_loss, per_batch_loss.mean()


class DistanceConstraint(Constraint):
    def __init__(
        self,
        resid1,  # Make this a list of integers
        resid2,  # Make this a list of integers
        target,  # Make this a list of floats
        tolerance=0.0,  # Make this a list of floats
        force_constant=2.0,  # Make this a list of floats
        constraint_weight=1.0,
    ):
        """Create constraint for distance between two residues."""
        super().__init__(constraint_weight=constraint_weight)
        self.resid1 = resid1
        self.resid2 = resid2
        self.target = target
        self.tolerance = tolerance
        self.force_constant = force_constant

    def compute_loss(
        self, distance_maps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract distances between specified residues
        distances = distance_maps[:, :, self.resid1, self.resid2]

        # Calculate deviation from target
        deviation = torch.abs(distances - self.target)

        # Apply flat-bottom: only penalize deviations beyond tolerance
        excess = torch.nn.functional.relu(deviation - self.tolerance)

        # Calculate harmonic potential for the excess deviation
        per_batch_loss = 0.5 * self.force_constant * excess**2

        return per_batch_loss, per_batch_loss.mean()


class RgConstraint(Constraint):
    def __init__(self, target, tolerance, force_constant=2.0, constraint_weight=1.0):
        super().__init__(constraint_weight=constraint_weight)
        self.target = target
        self.tolerance = tolerance
        self.force_constant = force_constant

    def __compute_rg(self, distance_maps: torch.Tensor) -> torch.Tensor:
        distance_maps = distance_maps[
            :, :, : self.sequence_length, : self.sequence_length
        ]
        distances = torch.sum(torch.square(distance_maps), axis=(1, 2, 3))
        rg_vals = torch.sqrt(distances / (2 * torch.pow(self.sequence_length, 2)))

        return rg_vals

    def compute_loss(
        self, distance_maps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_rg = self.__compute_rg(distance_maps)

        # Calculate deviation from target
        deviation = torch.abs(predicted_rg - self.target)

        # Apply flat-bottom: only penalize deviations beyond tolerance
        excess = torch.nn.functional.relu(deviation - self.tolerance)

        # Calculate harmonic potential for the excess deviation
        per_batch_loss = 0.5 * self.force_constant * excess**2

        return per_batch_loss, per_batch_loss.mean()


class ReConstraint(Constraint):
    def __init__(self, target, tolerance, force_constant=2.0, constraint_weight=1.0):
        super().__init__(constraint_weight=constraint_weight)
        self.target = target
        self.tolerance = tolerance
        self.force_constant = force_constant

    def compute_loss(
        self, distance_maps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distances = distance_maps[:, :, 0, self.sequence_length]

        # Calculate deviation from target
        deviation = torch.abs(distances - self.target)

        # Apply flat-bottom: only penalize deviations beyond tolerance
        excess = torch.nn.functional.relu(deviation - self.tolerance)

        # Calculate harmonic potential for the excess deviation
        per_batch_loss = 0.5 * self.force_constant * excess**2

        return per_batch_loss, per_batch_loss.mean()


class MultiConstraint(Constraint):
    """Combines multiple constraints into a single optimization step."""

    def __init__(self, constraints, schedule="cosine", verbose=True):
        """
        Parameters
        ----------
        constraints : list
            List of constraint objects to combine
        constraint_weights : list, optional
            Relative weights for each constraint (defaults to equal weights)
        guidance_scale : float
            Overall guidance scale for the combined constraint
        schedule : str
            Time schedule for constraint application ("cosine" or "linear")
        verbose : bool
            Whether to print debug info
        """
        super().__init__(schedule=schedule, verbose=verbose)
        self.constraints = constraints
        self.constraint_weights = [
            constraint.constraint_weight for constraint in constraints
        ]

    def initialize(
        self, encoder_model, latent_space_scaling_factor, n_steps, sequence_length
    ):
        """Initialize all constraints with the model parameters."""
        super().initialize(
            encoder_model, latent_space_scaling_factor, n_steps, sequence_length
        )

        # Initialize all subconstraints
        for constraint in self.constraints:
            constraint.initialize(
                encoder_model, latent_space_scaling_factor, n_steps, sequence_length
            )

        return self

    def compute_loss(
        self, distance_maps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted combination of all constraint losses."""
        total_per_batch_loss = None
        total_loss = 0.0

        for i, (constraint, weight) in enumerate(
            zip(self.constraints, self.constraint_weights)
        ):
            # Get per-batch and mean loss from each constraint
            per_batch_loss, mean_loss = constraint.compute_loss(distance_maps)

            # Apply weight to both
            weighted_per_batch = weight * per_batch_loss
            weighted_loss = weight * mean_loss

            # Accumulate
            if total_per_batch_loss is None:
                total_per_batch_loss = weighted_per_batch
            else:
                total_per_batch_loss = total_per_batch_loss + weighted_per_batch

            total_loss += weighted_loss

        return total_per_batch_loss, total_loss


class ConstraintLogger:
    def __init__(self, n_steps, verbose=True, update_freq=1):
        self.n_steps = n_steps
        self.verbose = verbose
        self.update_freq = update_freq
        self.constraint_data = {}
        self.progress_bar = None
        self.start_time = None
        self.steps_applied = 0  # Add a counter for steps where constraint was applied

    def setup(self):
        """Set up the progress bar."""
        if self.verbose:
            self.progress_bar = tqdm(
                total=self.n_steps,
                desc="Applying constraints",
                position=1,
                leave=False,
            )

    def update(self, timestep, constraint_name, metrics):
        """Update logger with new constraint metrics."""
        if not self.verbose or self.progress_bar is None:
            return

        # Increment our internal counter of steps where constraint was applied
        self.steps_applied += 1

        # Store most recent data
        self.constraint_data[constraint_name] = metrics

        # Update the progress bar's position directly
        self.progress_bar.n = self.steps_applied

        # Create status message
        status_parts = []

        # Add constraint info
        for name, data in self.constraint_data.items():
            loss = data.get("loss", 0)
            grad_norm = data.get("grad_norm", 0)
            status_parts.append(f"{name[:3]} loss: {loss:.4f} grad: {grad_norm:.2f}")

        # Update status text and refresh
        self.progress_bar.set_postfix_str(" | ".join(status_parts))
        self.progress_bar.refresh()

    def close(self):
        """Close the logger."""
        if self.verbose and self.progress_bar is not None:
            self.progress_bar.close()
