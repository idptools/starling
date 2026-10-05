"""
The base class every STARLING sampler derives from.

All of STARLING's samplers generate conformations the same way: start from
Gaussian noise in the VAE's latent space, repeatedly ask the diffusion model
how much noise the latents contain and remove some of it, then decode the
clean latents into distance maps. The samplers differ only in which
timesteps they visit and in how one step turns the model's noise prediction
into less noisy latents.

BaseSampler implements everything they share (encoding the sequence,
drawing the initial noise, the denoising loop, constraint guidance, progress
bars and decoding). Each sampler subclasses it, sets ``self.timesteps`` and
implements ``denoising_step()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Final

import torch
from tqdm.auto import tqdm

from starling import configs
from starling.data.tokenizer import StarlingTokenizer
from starling.inference.constraints import ConstraintLogger
from starling.samplers.sampler_utilities import check_conformation_count

if TYPE_CHECKING:
    from starling.inference.constraints import Constraint
    from starling.models.diffusion import DiffusionModel
    from starling.models.vae import VAE

# Shape of one conformation's latent (channels, height, width): the VAE
# compresses a 384 x 384 distance map into a single 24 x 24 channel
LATENT_SHAPE: Final[tuple[int, int, int]] = (1, 24, 24)


class BaseSampler(ABC):
    """
    Shared machinery for sampling distance maps from a STARLING diffusion model.

    Subclasses must set ``name`` and fill ``self.timesteps`` in their
    constructor, and implement ``denoising_step()``.

    Attributes
    ----------
    name : str
        Name used in progress bars and error messages.
    supports_constraints : bool
        Whether ``sample()`` accepts a constraint to guide sampling.
    timesteps : list of int
        The timestep each step starts from (where the model is evaluated),
        noisiest first. One entry per step.
    """

    name: ClassVar[str]
    supports_constraints: ClassVar[bool] = True

    def __init__(
        self,
        ddpm_model: DiffusionModel,
        encoder_model: VAE,
        ionic_strength: float = configs.DEFAULT_IONIC_STRENGTH,
    ) -> None:
        """
        Set up the parts every sampler needs.

        Parameters
        ----------
        ddpm_model : DiffusionModel
            The trained diffusion model, which provides the noise predictor
            and its noise schedule.
        encoder_model : VAE
            The trained VAE, used to decode latents into distance maps.
        ionic_strength : float, optional
            Ionic strength in mM to condition the model on. Default
            configs.DEFAULT_IONIC_STRENGTH (150 mM).
        """
        self.ddpm_model = ddpm_model
        self.encoder_model = encoder_model
        self.device = ddpm_model.device
        self.tokenizer = StarlingTokenizer()
        self.ionic_strength = torch.tensor([[ionic_strength]], device=self.device)

        self.num_timesteps: int = ddpm_model.num_timesteps
        self.latent_space_scaling_factor = ddpm_model.latent_space_scaling_factor

        # every sampler does its step arithmetic in float32 on the model's device
        self.alphas_cumprod = ddpm_model.alphas_cumprod.detach().to(device=self.device, dtype=torch.float32)

        # set by each subclass
        self.timesteps: list[int] = []

    @property
    def n_steps(self) -> int:
        """Number of denoising steps one call to sample() takes."""
        return len(self.timesteps)

    def encode_sequence(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Turn a sequence into the context the diffusion model is conditioned on.

        Parameters
        ----------
        sequence : str
            Amino acid sequence.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            The sequence (and ionic strength) context and its attention mask,
            each for a batch of one sequence.
        """
        tokens = torch.tensor([self.tokenizer.encode(sequence)], device=self.device)
        attention_mask = torch.ones_like(tokens, dtype=torch.bool)
        context = self.ddpm_model.sequence2labels(tokens, attention_mask, self.ionic_strength)
        return context, attention_mask

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: int,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the noise in a batch of latents at one timestep.

        Parameters
        ----------
        latents : torch.Tensor
            Noisy latents, shape (batch, 1, 24, 24).
        timestep : int
            Timestep (noise level) of the latents.
        context : torch.Tensor
            Sequence context from encode_sequence().
        attention_mask : torch.Tensor
            Attention mask from encode_sequence().

        Returns
        -------
        torch.Tensor
            Predicted noise, same shape as latents, in float32 whatever
            precision the network runs in.
        """
        batched_timesteps = torch.full((latents.shape[0],), timestep, device=latents.device, dtype=torch.long)
        predicted_noise: torch.Tensor = self.ddpm_model.model(latents, batched_timesteps, context, attention_mask)
        return predicted_noise.float()

    @abstractmethod
    def denoising_step(
        self,
        latents: torch.Tensor,
        step_index: int,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
        history: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Take one denoising step, starting from timestep self.timesteps[step_index].

        Parameters
        ----------
        latents : torch.Tensor
            Latents at the start of the step, shape (batch, 1, 24, 24).
        step_index : int
            Which step this is (0 for the first, noisiest step).
        context : torch.Tensor
            Sequence context from encode_sequence().
        attention_mask : torch.Tensor
            Attention mask from encode_sequence().
        history : list of torch.Tensor
            Somewhere multistep samplers can keep what earlier steps of this
            trajectory computed. It is empty at the start of every call to
            sample(), and samplers that do not need it ignore it.

        Returns
        -------
        torch.Tensor
            Latents at the end of the step.
        """

    def _start_guidance(self, constraint: Constraint | None, sequence_length: int) -> ConstraintLogger | None:
        """
        Prepare a constraint for this trajectory and return its logger.

        Constraints work on the model's timesteps (0 to num_timesteps - 1)
        whatever grid the sampler steps through, so they are always told the
        full number of training timesteps.

        Parameters
        ----------
        constraint : Constraint or None
            Constraint to guide sampling with, if any.
        sequence_length : int
            Number of residues.

        Returns
        -------
        ConstraintLogger or None
            A logger that has been set up, or None without a constraint.

        Raises
        ------
        ValueError
            If a constraint is given to a sampler that does not support them.
        """
        if constraint is None:
            return None

        if not self.supports_constraints:
            raise ValueError(f"{self.name} does not support constraints; use DDIM, PLMS or DDPM")

        constraint.initialize(
            self.encoder_model,
            self.latent_space_scaling_factor,
            self.num_timesteps,
            sequence_length,
        )
        logger = ConstraintLogger(n_steps=self.num_timesteps, verbose=True)
        logger.setup()
        return logger

    @torch.no_grad()
    def sample(
        self,
        num_conformations: int,
        sequence: str,
        show_per_step_progress_bar: bool = True,
        batch_count: int = 1,
        max_batch_count: int = 1,
        constraint: Constraint | None = None,
    ) -> torch.Tensor:
        """
        Generate distance maps for one sequence.

        Parameters
        ----------
        num_conformations : int
            Number of conformations to generate in this batch.
        sequence : str
            Amino acid sequence to condition on.
        show_per_step_progress_bar : bool, optional
            Whether to show a progress bar over the denoising steps. Default True.
        batch_count : int, optional
            Number of this batch, shown in the progress bar. Default 1.
        max_batch_count : int, optional
            Total number of batches, shown in the progress bar. Default 1.
        constraint : Constraint or None, optional
            Constraint that guides the latents after every step. Default None.

        Returns
        -------
        torch.Tensor
            Decoded, uncropped distance maps in Angstroms, shape
            (num_conformations, 1, 384, 384), on the model's device.

        Raises
        ------
        ValueError
            If num_conformations is not a positive integer, or a constraint is
            given to a sampler that does not support them.
        """
        check_conformation_count(num_conformations)
        guidance_logger = self._start_guidance(constraint, len(sequence))

        context, attention_mask = self.encode_sequence(sequence)
        latents = torch.randn((num_conformations, *LATENT_SHAPE), device=self.device)

        progress_bar = tqdm(
            total=self.n_steps,
            disable=not show_per_step_progress_bar,
            position=1,
            leave=False,
            desc=f"{self.name} steps (batch {batch_count} of {max_batch_count})",
        )

        history: list[torch.Tensor] = []
        for step_index, timestep in enumerate(self.timesteps):
            latents = self.denoising_step(latents, step_index, context, attention_mask, history)

            # guidance acts on the latents a step has produced, labelled by the
            # timestep the step started from; nothing is left to guide after
            # DDPM's step from timestep 0
            if constraint is not None and timestep != 0:
                latents = constraint.apply(latents, timestep, logger=guidance_logger)

            progress_bar.update(1)

        progress_bar.close()
        if guidance_logger is not None:
            guidance_logger.close()

        # undo the scaling the diffusion model was trained on, then decode
        return self.encoder_model.decode(latents / self.latent_space_scaling_factor)
