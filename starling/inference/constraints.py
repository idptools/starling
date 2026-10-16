from __future__ import annotations

import math
import warnings
from abc import ABC
from typing import TYPE_CHECKING, Tuple

import torch
from einops import rearrange, reduce
from tqdm.auto import tqdm

from starling import configs
from starling.utilities import deterministic_cudnn, helix_dm
from starling.utilities import symmetrize_tensor_distance_maps as symmetrize_distance_maps

if TYPE_CHECKING:
    from starling.models.vae import VAE

# Batch size the existing constraint_weight values were tuned at (the default
# generate() batch size, configs.DEFAULT_BATCH_SIZE, at the time). Guidance used
# to differentiate the batch-mean loss, so each conformer's update scaled as
# 1 / batch_size. Each conformer now gets the gradient of its own loss divided by
# this constant, which leaves guidance at this batch size unchanged and makes it
# independent of batch size.
REFERENCE_BATCH_SIZE = 100

# Accepted values for a constraint's time-dependent guidance schedule; None
# keeps a constant weight of 1.0 throughout sampling
VALID_SCHEDULES = ("cosine", "bell_shaped", "linear", None)


class Constraint(ABC):
    """
    Base class for constraints that guide sampling towards a target.

    At each guided diffusion step the current latents are decoded into
    distance maps, the constraint's loss is evaluated for every conformer,
    and the latents are nudged down the gradient of that loss. Each conformer
    is moved by the gradient of its own loss, divided by
    ``REFERENCE_BATCH_SIZE`` (100), so the guidance a conformer receives does
    not depend on how many conformers are sampled alongside it. Before this
    the gradient of the batch-mean loss was used, which made guidance scale
    as 1 / batch_size; dividing by ``REFERENCE_BATCH_SIZE`` keeps existing
    ``constraint_weight`` values (tuned at the default batch size of 100)
    meaning what they did at that batch size.
    """

    def __init__(
        self,
        constraint_weight=1.0,
        schedule="cosine",
        verbose=True,
        guidance_start=0.0,
        guidance_end=1.0,
    ):
        """Initialize base constraint with common parameters.

        Parameters
        ----------
        constraint_weight : float, default=1.0
            Weight factor for the constraint. Guidance does not depend on
            batch size (see the class docstring).
        schedule : {"cosine", "bell_shaped", "linear"} or None, default="cosine"
            Scheduling function for time-dependent guidance strength.
            ``"cosine"`` fades guidance in smoothly from zero at the start of
            sampling to full strength at the end; ``"bell_shaped"`` peaks
            about 43% of the way through sampling; ``"linear"`` ramps
            linearly from zero at the start to full strength at the end; and
            None keeps a constant weight of 1.0 throughout.
        verbose : bool, default=True
            Whether to print debug information
        guidance_start : float, default=0.0
            Normalized timestep to start applying guidance (0.0 = beginning)
        guidance_end : float, default=1.0
            Normalized timestep to stop applying guidance (1.0 = end)

        Raises
        ------
        ValueError
            If ``schedule`` is not one of the accepted values.
        """
        if schedule not in VALID_SCHEDULES:
            raise ValueError(f"schedule must be one of {VALID_SCHEDULES}; got {schedule!r}")

        # These will be set by the sampler
        self.encoder_model: VAE | None = None
        self.latent_space_scaling_factor: torch.Tensor | None = None
        self.n_steps: int | None = None
        self.device: torch.device | None = None

        # User-controlled parameters
        self.constraint_weight = constraint_weight
        self.schedule = schedule
        self.verbose = verbose

        self.guidance_start = guidance_start
        self.guidance_end = guidance_end

    def _setup_constraint(self):
        """Set up constraint-specific resources."""
        pass  # Implemented by subclasses

    def initialize(
        self,
        encoder_model: VAE,
        latent_space_scaling_factor: torch.Tensor,
        n_steps: int,
        sequence_length: int,
    ) -> "Constraint":
        """Called by the sampler to set model parameters."""
        self.encoder_model = encoder_model
        self.latent_space_scaling_factor = latent_space_scaling_factor
        self.n_steps = n_steps
        self.device = encoder_model.device
        self.sequence_length = sequence_length
        self._setup_constraint()
        return self

    def _require_n_steps(self) -> int:
        if self.n_steps is None:
            raise RuntimeError("Constraint must be initialized by a sampler first")
        return self.n_steps

    def should_apply_guidance(self, timestep: int, total_steps: int) -> bool:
        """
        Check if guidance should be applied at the current timestep.

        Parameters
        ----------
        timestep : int
            Current diffusion timestep.
        total_steps : int
            Total number of diffusion steps.

        Returns
        -------
        bool
            True if current timestep is within the guidance window.
        """
        t_frac = timestep / total_steps
        reverse = 1 - t_frac
        return self.guidance_start <= reverse <= self.guidance_end

    def cosine_weight(self, t, total_steps):
        """
        Cosine schedule for time-dependent guidance strength.

        The weight is cos^2(pi t / (2 T)), which is 0 at the start of sampling
        (t = T) and rises smoothly to 1 at the end (t = 0).

        Parameters
        ----------
        t : int
            Current timestep.
        total_steps : int
            Total number of steps.

        Returns
        -------
        float
            Guidance weight following cosine schedule.
        """
        t_scaled = t / total_steps
        return math.cos(t_scaled * math.pi / 2) ** 2

    def bell_shaped_schedule(self, timestep: int) -> float:
        """Bell-shaped schedule for time-dependent guidance strength.

        Creates a schedule that gradually increases guidance strength,
        peaks at timestep t/T ~ 0.57 (about 43% of the way through sampling,
        which runs from t = T down to 0), and then decreases again.

        Parameters
        ----------
        timestep : int
            Current diffusion timestep

        Returns
        -------
        float
            Guidance strength factor (peaks a little before the middle of
            sampling)
        """
        normalized_t = timestep / self._require_n_steps()
        # The Gaussian factor is centred on t/T = 0.6, but multiplying by the
        # sine pulls the peak to t/T ~ 0.57, i.e. ~43% of the way through sampling
        return math.sin(normalized_t * math.pi) * math.exp(-((normalized_t - 0.6) ** 2) / 0.1)

    def get_adaptive_clip_threshold(self, timestep):
        """
        Get an adaptive clipping threshold that follows a cosine schedule.

        The threshold starts high at the beginning of sampling and gradually
        decreases, allowing larger gradients early on and more refined
        adjustments later.

        Parameters
        ----------
        timestep : int
            Current diffusion timestep.

        Returns
        -------
        float
            Clipping threshold for gradient magnitudes.
        """
        max_threshold = 2.0  # Maximum threshold at beginning
        min_threshold = 1.0  # Minimum threshold at end

        # Cosine decay from max_threshold to min_threshold
        fraction_complete = 1 - (timestep / self._require_n_steps())
        cosine_factor = math.cos(fraction_complete * math.pi / 2)
        return min_threshold + cosine_factor**2 * (max_threshold - min_threshold)

    def compute_loss(self, distance_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _loss_for_guidance(
        self, distance_maps: torch.Tensor, timestep: int, total_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compute_loss(distance_maps)

    def apply(self, latents: torch.Tensor, timestep: int, logger=None) -> torch.Tensor:
        """Apply the constraint to the given latents."""

        n_steps = self._require_n_steps()
        if self.encoder_model is None or self.latent_space_scaling_factor is None:
            raise RuntimeError("Constraint must be initialized by a sampler first")

        # Check if the constraint should be applied
        if not self.should_apply_guidance(timestep, n_steps):
            return latents

        with torch.inference_mode(False):
            per_batch_loss, base_grad = self._per_conformer_loss_and_gradient(latents, timestep, n_steps)

            # Get time-dependent scaling
            time_scale = self.get_time_scale(timestep)

            # Calculate per-sample loss scaling
            mean_loss = per_batch_loss.mean()
            if mean_loss > 1e-6:
                loss_scale = per_batch_loss / mean_loss
            else:
                # When mean loss is very small, use a uniform scale
                loss_scale = torch.ones_like(per_batch_loss)
            # loss_scale = per_batch_loss / per_batch_loss.mean()

            # Prevent extreme scaling
            max_scale_factor = 2.0
            loss_scale = torch.clamp(loss_scale, max=max_scale_factor)

            # Reshape loss_scale to match the shape of the latents
            loss_scale = rearrange(loss_scale, "b -> b 1 1 1")

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
                        "loss": mean_loss.item(),
                        "grad_norm": update.norm().item(),
                        # "update_norm": update_norm,
                        "time_scale": time_scale,
                        "min_loss_scale": loss_scale.min().item(),
                        "max_loss_scale": loss_scale.max().item(),
                        # "clipped": update_norm > 1.0,
                    },
                )

            return latents + update.detach()

    @deterministic_cudnn()
    def _per_conformer_loss_and_gradient(
        self, latents: torch.Tensor, timestep: int, n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate every conformer's loss and its gradient, a few conformers at a time.

        Backpropagating through the VAE decoder needs about 320 MB of GPU
        memory per conformer, so a whole sampling batch is done
        configs.VAE_DECODE_CHUNK_SIZE conformers at a time. Each
        conformer's loss depends only on its own distance map, so this does
        not change what is computed. cuDNN runs deterministically here (see
        starling.utilities.deterministic_cudnn), so identically seeded runs
        give identical guidance. Must be called with autograd enabled.

        Parameters
        ----------
        latents : torch.Tensor
            Latents in the diffusion model's scaled space, shape
            (batch, 1, 24, 24).
        timestep : int
            Current diffusion timestep.
        n_steps : int
            Total number of diffusion timesteps.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Each conformer's loss, shape (batch,), and the gradient of that
            loss with respect to its latents divided by REFERENCE_BATCH_SIZE,
            same shape as latents.
        """
        assert self.encoder_model is not None and self.latent_space_scaling_factor is not None

        losses = []
        gradients = []
        for chunk in torch.split(latents, configs.VAE_DECODE_CHUNK_SIZE):
            chunk_copy = chunk.clone().requires_grad_(True)
            distance_maps = self.encoder_model.decode(chunk_copy / self.latent_space_scaling_factor)
            distance_maps = symmetrize_distance_maps(distance_maps)
            chunk_loss, _ = self._loss_for_guidance(distance_maps, timestep, n_steps)

            # Differentiating the summed loss gives every conformer the gradient
            # of its own loss, whatever the batch size; see REFERENCE_BATCH_SIZE
            # for why it is then divided by that constant. The division also
            # makes a new tensor, so nothing here aliases a CUDA-graph buffer
            # that the next chunk's call would overwrite.
            gradients.append(torch.autograd.grad(chunk_loss.sum(), chunk_copy)[0] / REFERENCE_BATCH_SIZE)
            losses.append(chunk_loss.detach())

        return torch.cat(losses), torch.cat(gradients)

    def get_time_scale(self, timestep: int) -> float:
        """
        Get the time-dependent scaling factor.

        Parameters
        ----------
        timestep : int
            Current diffusion timestep.

        Returns
        -------
        float
            Guidance weight for this timestep under ``self.schedule``
            (1.0 throughout when the schedule is None).

        Raises
        ------
        ValueError
            If ``self.schedule`` has been set to an unknown value.
        """
        n_steps = self._require_n_steps()
        if self.schedule is None:
            return 1.0
        if self.schedule == "cosine":
            return self.cosine_weight(timestep, total_steps=n_steps)
        elif self.schedule == "bell_shaped":
            return self.bell_shaped_schedule(timestep)
        elif self.schedule == "linear":
            return 1.0 - (timestep / n_steps)
        raise ValueError(f"schedule must be one of {VALID_SCHEDULES}; got {self.schedule!r}")


class BondConstraint(Constraint):
    def __init__(self, bond_length=3.81, tolerance=0.0, force_constant=2.0, **kwargs):
        super().__init__(**kwargs)
        self.bond_length = bond_length
        self.tolerance = tolerance
        self.force_constant = force_constant

    def compute_loss(self, distance_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bond loss based on distance maps.
        This loss penalizes deviations from the ideal bond length
        (``bond_length``, 3.81 Å by default) with a harmonic potential. Only
        deviations beyond ``tolerance`` are penalized (a flat bottom); the
        default tolerance of 0.0 Å means there is no flat bottom.

        Parameters
        ----------
        distance_maps : torch.Tensor
            Pre-computed distance maps from the latents

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Per-batch loss and mean loss
        """

        distance_maps = distance_maps[:, 0, : self.sequence_length, : self.sequence_length]

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
    def __init__(self, steric_clash_definition=5.0, force_constant=2.0, **kwargs):
        super().__init__(**kwargs)
        self.steric_clash_definition = steric_clash_definition
        self.force_constant = force_constant

    def compute_loss(self, distance_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Tuple[torch.Tensor, torch.Tensor]
            Per-batch loss and mean loss. Both are zero for sequences of two
            or fewer residues, which have no residue pairs to check.
        """
        mask = torch.triu(
            torch.ones(self.sequence_length, self.sequence_length, device=distance_maps.device),
            diagonal=2,
        )

        distance_maps = distance_maps[:, :, : self.sequence_length, : self.sequence_length]

        # Calculate the deviation from steric_clash_definition (only when distances are smaller)
        deviation = torch.relu(self.steric_clash_definition - distance_maps)

        # Apply harmonic potential formula: 0.5 * force_constant * deviation^2
        steric_clash = 0.5 * self.force_constant * deviation**2

        # Apply mask to consider only upper triangle without diagonals
        steric_clash = steric_clash * mask

        # Sum across all residue pairs and normalize
        steric_clash = reduce(steric_clash, "b c h w -> b", "sum")
        normalization_factor = mask.sum()

        # with two or fewer residues no pair is |i - j| >= 2 apart, so the
        # (all-zero) loss must not be divided by a pair count of zero, which
        # would turn the guided latents into NaN
        if normalization_factor == 0:
            return steric_clash, steric_clash.mean()

        per_batch_loss = steric_clash / normalization_factor

        return per_batch_loss, per_batch_loss.mean()


class HelicityConstraint(Constraint):
    def __init__(self, resid_start, resid_end, tolerance=0.0, force_constant=2.0, **kwargs):
        """Create constraint pushing a region towards ideal helical distances.

        Parameters
        ----------
        resid_start : int
            Index of the first residue in the helical region. Indices are
            0-based (the first residue is 0).
        resid_end : int
            Index one past the last residue in the helical region (the region
            is ``resid_start`` to ``resid_end - 1`` inclusive, like a Python
            slice). The region must lie within the sequence and contain at
            least two residues; this is checked when the sampler initializes
            the constraint.
        tolerance : float, default=0.0
            Allowed deviation (Angstroms) from the helical distance before a
            penalty applies.
        force_constant : float, default=2.0
            Force constant for the harmonic potential.
        **kwargs
            Additional parameters passed to parent Constraint class
        """
        super().__init__(**kwargs)

        self.resid_start = resid_start
        self.resid_end = resid_end
        self.tolerance = tolerance
        self.force_constant = force_constant

        # These will be initialized when the model is available
        self.helix_ref = None
        self.mask = None
        self.weights = None

    def _setup_constraint(self):
        """Check the helical region fits the sequence, then set up device-specific tensors."""
        # a region needs at least two residues to contain a pair; with fewer the
        # mask is empty and the loss would divide by zero
        if self.resid_start < 0 or self.resid_end > self.sequence_length or self.resid_end - self.resid_start < 2:
            raise ValueError(
                f"HelicityConstraint region resid_start={self.resid_start}, resid_end={self.resid_end} does not fit "
                f"a sequence of {self.sequence_length} residues. Indices are 0-based and resid_end is exclusive, so "
                f"0 <= resid_start, resid_end <= {self.sequence_length} and resid_end - resid_start >= 2 are required."
            )
        if self.encoder_model is None:
            return
        if self.device is None:
            raise RuntimeError("Constraint device is unavailable before initialization")

        # Create helix reference
        self.helix_ref = torch.from_numpy(helix_dm(L=384)).to(self.device)

        # Create mask
        self.mask = torch.zeros((384, 384), device=self.device)
        self.mask[self.resid_start : self.resid_end, self.resid_start : self.resid_end] = torch.triu(
            torch.ones((self.resid_end - self.resid_start, self.resid_end - self.resid_start)),
            diagonal=1,
        )

        # Create weights
        self.weights = 1.0 / (self.helix_ref + 1e-2)

    def compute_loss(self, distance_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.helix_ref is None or self.mask is None:
            raise RuntimeError("HelicityConstraint must be initialized first")
        # Calculate deviation from reference helix
        deviation = torch.abs(distance_maps - self.helix_ref)

        # Apply flat-bottom potential: only penalize deviations beyond tolerance
        excess = torch.nn.functional.relu(deviation - self.tolerance)

        # Calculate harmonic potential for the excess deviation
        region_loss = 0.5 * self.force_constant * (excess**2) * self.mask
        normalization_factor = self.mask.sum()
        per_batch_loss = reduce(region_loss, "b c h w -> b", "sum") / normalization_factor

        # Return per-batch and mean loss
        return per_batch_loss, per_batch_loss.mean()


class DistanceConstraint(Constraint):
    def __init__(self, resid1, resid2, target, tolerance=0.0, force_constant=2.0, **kwargs):
        """Create constraint for distance between two residues.

        Parameters
        ----------
        resid1, resid2 : int
            Indices of the two residues. Indices are 0-based (the first
            residue is 0) and must lie within the sequence; this is checked
            when the sampler initializes the constraint.
        target : float
            Target distance between the two residues in Angstroms.
        tolerance : float, default=0.0
            Allowed deviation from target (Angstroms) before penalty applies.
        force_constant : float, default=2.0
            Force constant for the harmonic potential.
        **kwargs
            Additional parameters passed to parent Constraint class
        """
        super().__init__(**kwargs)
        self.resid1 = resid1
        self.resid2 = resid2
        self.target = target
        self.tolerance = tolerance
        self.force_constant = force_constant

    def _setup_constraint(self):
        """Check both residue indices lie within the sequence being sampled."""
        for resid in (self.resid1, self.resid2):
            if not 0 <= resid < self.sequence_length:
                raise ValueError(
                    f"DistanceConstraint residue index {resid} is outside a sequence of {self.sequence_length} "
                    f"residues. Indices are 0-based, so they must lie between 0 and {self.sequence_length - 1}."
                )

    def compute_loss(self, distance_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract distances between specified residues
        distances = distance_maps[:, :, self.resid1, self.resid2]

        # Calculate deviation from target
        deviation = torch.abs(distances - self.target)

        # Apply flat-bottom: only penalize deviations beyond tolerance
        excess = torch.nn.functional.relu(deviation - self.tolerance)

        # Calculate harmonic potential for the excess deviation
        per_batch_loss = 0.5 * self.force_constant * excess**2

        per_batch_loss = rearrange(per_batch_loss, "b 1 -> b")

        return per_batch_loss, per_batch_loss.mean()


class RgConstraint(Constraint):
    def __init__(self, target, tolerance=0.0, force_constant=2.0, **kwargs):
        """Create constraint for radius of gyration (Rg).

        Parameters
        ----------
        target : float
            Target Rg value in Angstroms
        tolerance : float, default=0.0
            Allowed deviation from target before penalty applies
        force_constant : float, default=2.0
            Force constant for the harmonic potential
        **kwargs
            Additional parameters passed to parent Constraint class
        """
        super().__init__(**kwargs)
        self.target = target
        self.tolerance = tolerance
        self.force_constant = force_constant

    def __compute_rg(self, distance_maps: torch.Tensor) -> torch.Tensor:
        """Calculate radius of gyration from distance maps.

        Rg = sqrt(sum(d_ij^2) / (2*N^2)) where d_ij are pairwise distances.

        Parameters
        ----------
        distance_maps : torch.Tensor
            Protein distance maps

        Returns
        -------
        torch.Tensor
            Calculated Rg values for each protein in the batch
        """
        sequence_length = torch.tensor(self.sequence_length, device=self.device)
        distance_maps = distance_maps[:, :, : self.sequence_length, : self.sequence_length]

        squared_distances = torch.square(distance_maps)

        distances = reduce(squared_distances, "b c h w -> b", "sum")
        rg_vals = torch.sqrt(distances / (2 * torch.pow(sequence_length, 2)))
        return rg_vals

    def compute_loss(self, distance_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss based on deviation from target Rg.

        Parameters
        ----------
        distance_maps : torch.Tensor
            Pre-computed distance maps from the latents

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Per-batch loss and mean loss
        """
        predicted_rg = self.__compute_rg(distance_maps)

        # Calculate deviation from target
        deviation = torch.abs(predicted_rg - self.target)

        # Apply flat-bottom: only penalize deviations beyond tolerance
        excess = torch.nn.functional.relu(deviation - self.tolerance)

        # Calculate harmonic potential for the excess deviation
        per_batch_loss = 0.5 * self.force_constant * excess**2

        return per_batch_loss, per_batch_loss.mean()


class ReConstraint(Constraint):
    def __init__(self, target, tolerance=0.0, force_constant=2.0, **kwargs):
        """Create constraint for end-to-end distance."""
        super().__init__(**kwargs)
        self.target = target
        self.tolerance = tolerance
        self.force_constant = force_constant

    def compute_loss(self, distance_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distances = distance_maps[:, 0, 0, self.sequence_length - 1]

        # Calculate deviation from target
        deviation = torch.abs(distances - self.target)

        # Apply flat-bottom: only penalize deviations beyond tolerance
        excess = torch.nn.functional.relu(deviation - self.tolerance)

        # Calculate harmonic potential for the excess deviation
        per_batch_loss = 0.5 * self.force_constant * excess**2

        return per_batch_loss, per_batch_loss.mean()


class MultiConstraint(Constraint):
    """Combines multiple constraints into a single optimization step."""

    def __init__(
        self,
        constraints,
        schedule="cosine",
        verbose=True,
    ):
        """
        Combine constraints so their losses are applied as one guidance step.

        Each sub-constraint's loss is weighted by its own
        ``constraint_weight`` and only counts at timesteps inside its own
        ``guidance_start``/``guidance_end`` window. The time-dependent
        schedule and verbosity come from this MultiConstraint; the
        sub-constraints' own ``schedule`` and ``verbose`` settings are
        ignored (with a warning if any schedule differs from this one's).

        Parameters
        ----------
        constraints : list
            List of constraint objects to combine
        schedule : {"cosine", "bell_shaped", "linear"} or None, default="cosine"
            Time schedule applied to the combined guidance (see
            :class:`Constraint`).
        verbose : bool
            Whether to print debug info
        """
        super().__init__(schedule=schedule, verbose=verbose)

        ignored_schedules = [type(constraint).__name__ for constraint in constraints if constraint.schedule != schedule]
        if ignored_schedules:
            warnings.warn(
                f"MultiConstraint applies its own schedule ({schedule!r}); the different schedules set on "
                f"{', '.join(ignored_schedules)} are ignored.",
                stacklevel=2,
            )

        self.constraints = constraints
        self.constraint_weights = [constraint.constraint_weight for constraint in constraints]

        self.guidance_starts = [constraint.guidance_start for constraint in constraints]
        self.guidance_ends = [constraint.guidance_end for constraint in constraints]

    def initialize(self, encoder_model, latent_space_scaling_factor, n_steps, sequence_length):
        """Initialize all constraints with the model parameters."""
        super().initialize(encoder_model, latent_space_scaling_factor, n_steps, sequence_length)

        # Initialize all subconstraints
        for constraint in self.constraints:
            constraint.initialize(encoder_model, latent_space_scaling_factor, n_steps, sequence_length)

        return self

    def compute_loss(self, distance_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted combination of all constraint losses."""
        if not self.constraints:
            raise ValueError("MultiConstraint requires at least one constraint")
        total_per_batch_loss = distance_maps.new_zeros(distance_maps.shape[0])
        total_loss = distance_maps.new_zeros(())

        for i, (constraint, weight) in enumerate(zip(self.constraints, self.constraint_weights)):
            # Get per-batch and mean loss from each constraint
            per_batch_loss, mean_loss = constraint.compute_loss(distance_maps)

            # Apply weight to both
            weighted_per_batch = weight * per_batch_loss
            weighted_loss = weight * mean_loss

            # Accumulate
            total_per_batch_loss = total_per_batch_loss + weighted_per_batch
            total_loss += weighted_loss

        return total_per_batch_loss, total_loss

    def _loss_for_guidance(
        self, distance_maps: torch.Tensor, timestep: int, total_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        losses = [
            constraint.compute_loss(distance_maps)[0] * weight
            for constraint, weight in zip(self.constraints, self.constraint_weights)
            if constraint.should_apply_guidance(timestep, total_steps)
        ]
        if not losses:
            per_batch = distance_maps.flatten(start_dim=1).sum(dim=1) * 0
            return per_batch, per_batch.mean()
        per_batch = torch.stack(losses).sum(dim=0)
        return per_batch, per_batch.mean()


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
        """
        Set up the progress bar for constraint logging.

        Initializes the tqdm progress bar if verbose mode is enabled and
        resets the step counter.
        """
        self.steps_applied = 0  # Reset counter
        if self.verbose:
            self.progress_bar = tqdm(
                desc="Applying constraints",
                position=1,
                leave=False,
            )

    def update(self, timestep, constraint_name, metrics):
        """
        Update logger with new constraint metrics.

        Parameters
        ----------
        timestep : int
            Current diffusion timestep.
        constraint_name : str
            Name of the constraint being logged.
        metrics : dict
            Dictionary containing constraint metrics (loss, scale, etc.).
        """
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
        """
        Close the progress bar and clean up.

        Closes the tqdm progress bar if it exists in verbose mode.
        """
        if self.verbose and self.progress_bar is not None:
            self.progress_bar.close()
