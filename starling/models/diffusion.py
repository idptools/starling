import math
from os import PathLike
from typing import Final, TypedDict, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.functional import F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
)

from starling.data.schedulers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    sigmoid_beta_schedule,
)
from starling.models.vae import VAE

# Adapted from https://github.com/Camaltra/this-is-not-real-aerial-imagery/blob/main/src/ai/diffusion_process.py
# and https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py#L720

# Fraction of all training steps over which LinearWarmupCosineAnnealingLR
# ramps the learning rate up linearly before the cosine decay starts
WARMUP_FRACTION: Final[float] = 0.01

# Value the latent_space_scaling_factor buffer holds until it has been
# estimated from the first training batch. A model loaded from a checkpoint
# (e.g. for fine-tuning) already holds its estimated value, which is kept.
UNSET_LATENT_SCALING_FACTOR: Final[float] = 1.0


# Helper function
def extract(
    constants: torch.Tensor, timestamps: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    """
    Extract values from a tensor based on given timestamps.

    Parameters
    ----------
    constants : torch.Tensor
        The tensor to extract values from.
    timestamps : torch.Tensor
        A 1D tensor containing the indices for extraction.
    shape : torch.Size
        Shape of the tensor the extracted values will be broadcast against;
        only its number of dimensions is used.

    Returns
    -------
    torch.Tensor
        The tensor with extracted values.
    """
    batch_size = timestamps.shape[0]
    out = constants.gather(-1, timestamps)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(timestamps.device)


class DiffusionBatch(TypedDict):
    data: torch.Tensor
    sequence: torch.Tensor
    attention_mask: torch.Tensor
    ionic_strengths: torch.Tensor


class DiffusionModel(pl.LightningModule):
    """
    Denoising diffusion probabilistic model for latent space generation.

    Implements the diffusion process described in:
    - Sohl-Dickstein et al. (2015): Nonequilibrium Thermodynamics
    - Ho et al. (2020): Denoising Diffusion Probabilistic Models
    - Rombach et al. (2021): High-resolution image synthesis with latent diffusion
    """

    SCHEDULER_MAPPING = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }
    latent_space_scaling_factor: torch.Tensor
    betas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        sequence_encoder: nn.Module,
        distance_map_encoder: str | PathLike[str] | None,
        beta_scheduler: str = "cosine",
        timesteps: int = 1000,
        set_lr: float = 1e-4,
        min_snr_loss: bool = False,
        min_snr_gamma: float = 5.0,
        config_scheduler: str = "LinearWarmupCosineAnnealingLR",
    ) -> None:
        """
        A discrete-time denoising-diffusion model framework for latent space diffusion models.
        The model is based on the work of Sohl-Dickstein et al. [1], Ho et al. [2], and Rombach et al. [3].

        References
        ----------
        1) Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S.
        Deep Unsupervised Learning using Nonequilibrium Thermodynamics.
        in Proceedings of the 32nd International Conference on Machine Learning
        (eds. Bach, F. & Blei, D.) vol. 37 2256–2265 (PMLR, Lille, France, 07--09 Jul 2015).

        2) Ho, J., Jain, A. & Abbeel, P. Denoising Diffusion Probabilistic Models. arXiv [cs.LG] (2020).

        3) Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B.
        High-resolution image synthesis with latent diffusion models. arXiv [cs.CV] (2021).


        Parameters
        ----------
        model : nn.Module
            A neural network model that takes in the noisy latents, the timesteps, the sequence
            context and its attention mask and outputs the predicted noise
        sequence_encoder : nn.Module
            The model that turns tokenized sequences, their attention mask and the ionic strengths
            into the context the noise predictor is conditioned on (see sequence2labels)
        distance_map_encoder : str or PathLike or None
            Path to a trained VAE checkpoint. If given, the VAE is loaded and frozen, and every
            training and validation batch of distance maps is encoded into the latent space on the
            fly. If None, the batches are expected to hold latents already.
        beta_scheduler : str, optional
            The name of the beta scheduler to use ("linear", "cosine" or "sigmoid"), by default "cosine"
        timesteps : int, optional
            The number of timesteps to run the diffusion process, by default 1000
        set_lr : float, optional
            The initial learning rate for the optimizer, by default 1e-4
        min_snr_loss : bool, optional
            Whether to weight each sample's loss by min(SNR, min_snr_gamma) / SNR (Min-SNR
            weighting, Hang et al. 2023, https://arxiv.org/abs/2303.09556), by default False
        min_snr_gamma : float, optional
            The SNR cap gamma used by the Min-SNR weighting, by default 5.0
        config_scheduler : str, optional
            The name of the learning rate scheduler to use ("LinearWarmupCosineAnnealingLR",
            "CosineAnnealingLR", "CosineAnnealingWarmRestarts" or "OneCycleLR"), by default
            "LinearWarmupCosineAnnealingLR"

        Raises
        ------
        ValueError
            If the beta scheduler is not implemented
        """
        super().__init__()

        # Save the hyperparameters of the model but ignore the encoder_model and the U-Net model
        self.save_hyperparameters(
            ignore=["model", "sequence_encoder", "distance_map_encoder"]
        )

        self.model = model
        self.sequence_encoder = sequence_encoder

        if distance_map_encoder is not None:
            # Load on the CPU: without a map_location, Lightning restores the
            # tensors to the GPU they were saved from (cuda:0), which starts a
            # CUDA context there even when the model is meant for the CPU or
            # another GPU. Lightning moves the module to the right device later.
            self.distance_map_encoder: VAE | None = VAE.load_from_checkpoint(
                str(distance_map_encoder), map_location="cpu"
            )

            self.__freeze_distance_map_encoder(self.distance_map_encoder)
        else:
            self.distance_map_encoder = None

        # Learning rate params
        self.set_lr = set_lr
        self.config_scheduler = config_scheduler

        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler)
        if self.beta_scheduler_fn is None:
            raise ValueError(f"unknown beta schedule {beta_scheduler}")

        self.min_snr_loss = min_snr_loss
        self.min_snr_gamma = min_snr_gamma

        # Register scaling factor buffer (calculated during first training step)
        # Used to normalize latent space to unit variance per Reference #3
        self.register_buffer(
            "latent_space_scaling_factor",
            torch.tensor(UNSET_LATENT_SCALING_FACTOR, dtype=torch.float32),
        )

        # Calculate diffusion process parameters
        betas = self.beta_scheduler_fn(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # Register diffusion process buffers
        buffers = {
            "betas": betas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
            "posterior_variance": posterior_variance,
        }

        for name, buffer in buffers.items():
            self.register_buffer(name, buffer)

        # Store timesteps information
        self.num_timesteps = int(betas.shape[0])
        self.monitor = "epoch_val_loss"

    def _encode_distance_maps(self, distance_maps: torch.Tensor) -> torch.Tensor:
        """
        Encode distance maps into latents with the frozen VAE, in float32.

        Training runs under bf16 mixed precision, where autocast would run the
        encoder in bf16 and return bf16 latents. Their spacing (0.004-0.008 for
        |x| between 0.5 and 2) is comparable to the smallest forward-process
        noise level (0.01), the noise drawn for them with randn_like would be
        bf16 too, and latents precomputed by
        devtools/scripts/extract_latents_tar.py are float32, so the two data
        paths trained on differently rounded inputs. Encoding with autocast
        off keeps everything downstream in float32.

        Parameters
        ----------
        distance_maps : torch.Tensor
            Distance maps in Angstroms, shape (batch, 1, 384, 384).

        Returns
        -------
        torch.Tensor
            float32 latents (the mode of the VAE's posterior), shape
            (batch, 1, 24, 24).
        """
        assert self.distance_map_encoder is not None
        with torch.no_grad(), torch.autocast(device_type=distance_maps.device.type, enabled=False):
            return self.distance_map_encoder.encode(distance_maps.float()).mode()

    def __freeze_distance_map_encoder(self, encoder: VAE) -> None:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    # Remove mixed precision from this function, I've experienced numerical instability here
    @autocast(device_type="cuda", enabled=False)
    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Add the noise to x_start tensor based on the timestamp t

        Parameters
        ----------
        x_start : torch.Tensor
            The starting image tensor
        t : int
            The timestep of the denoising-diffusion process
        noise : torch.Tensor, optional
            Sampled noise to add, by default None

        Returns
        -------
        torch.Tensor
            Returns the properly (according to the timestamp) noised tensor
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Extract the necessary values from the buffers to calculate the noise to be added
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # Return the noised tensor based on the timestamp
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sequence2labels(
        self,
        sequences: torch.Tensor,
        sequence_mask: torch.Tensor,
        ionic_strength: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts tokenized sequences into the context the noise predictor is conditioned on,
        using the sequence encoder.

        Parameters
        ----------
        sequences : torch.Tensor
            Tokenized sequences, shape (batch, sequence_length)
        sequence_mask : torch.Tensor
            Attention mask for the sequences, True for real residues and False for padding,
            shape (batch, sequence_length)
        ionic_strength : torch.Tensor
            Ionic strength of each sequence in mM, shape (batch, 1)

        Returns
        -------
        torch.Tensor
            The sequence context, shape (batch, sequence_length, embed_dim)
        """
        encoded = self.sequence_encoder(sequences, sequence_mask, ionic_strength)

        return encoded

    def p_loss(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        ionic_strengths: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        A function that runs the model and calculates the loss based on the
        predicted noise and the actual noise. The loss is the mean squared error,
        optionally with Min-SNR weighting (see min_snr_loss in __init__).

        Parameters
        ----------
        x_start : torch.Tensor
            The starting (scaled) latents, shape (batch, 1, 24, 24)
        t : torch.Tensor
            The timestep of each sample along the denoising-diffusion process, shape (batch,)
        labels : torch.Tensor
            Tokenized sequences to condition the model on, shape (batch, sequence_length)
        mask : torch.Tensor
            Attention mask for the sequences, True for real residues, shape (batch, sequence_length)
        ionic_strengths : torch.Tensor
            Ionic strength of each sequence in mM, shape (batch, 1)
        noise : torch.Tensor, optional
            Sampled noise from N(0,I), by default None (drawn here)

        Returns
        -------
        torch.Tensor
            Returns the loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            # Offset noise that seems to improve the inference
            # According to https://www.crosslabs.org/blog/diffusion-with-offset-noise
            # noise += 0.1 * torch.randn(
            #     x_start.shape[0], x_start.shape[1], 1, 1, device=self.device
            # )

        # Noise the input data
        x_noised = self.q_sample(x_start, t, noise=noise)

        # Get the labels to condition the model on
        labels = self.sequence2labels(labels, mask, ionic_strengths)

        # Run the model to predict the noise
        predicted_noise = self.model(x_noised, t, labels, mask)

        # The following adapted from:
        # https://github.com/huggingface/diffusers/blob/78a78515d64736469742e5081337dbcf60482750/examples/text_to_image/train_text_to_image.py#L927
        if self.min_snr_loss:
            # Apply min-SNR weighting as per Section 3.4 of https://arxiv.org/abs/2303.09556
            # This improves training stability by reweighting timestep losses
            snr = self.compute_snr(t)

            # min(snr, gamma) / snr, with the zero-SNR limit defined as 1.
            snr_weight = torch.where(
                snr > 0,
                torch.clamp(snr, max=self.min_snr_gamma)
                / snr.clamp_min(torch.finfo(snr.dtype).tiny),
                torch.ones_like(snr),
            )

            # Apply the SNR-weighted MSE loss
            # First compute per-element losses, then average across spatial dimensions
            # Finally, apply SNR weights and average across batch
            mse_loss_raw = F.mse_loss(noise, predicted_noise, reduction="none")
            mse_loss_per_sample = mse_loss_raw.mean(
                dim=list(range(1, len(mse_loss_raw.shape)))
            )
            loss = (mse_loss_per_sample * snr_weight).mean()
        else:
            loss = F.mse_loss(noise, predicted_noise)

        return loss

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        ionic_strengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model, calculates the loss based on the
        predicted noise and the actual noise.

        Parameters
        ----------
        x : torch.Tensor
            The starting tensor to noise/denoise
        labels : torch.Tensor
            Tokenized sequences to condition the model on, shape (batch, sequence_length)
        mask : torch.Tensor
            Attention mask for the sequences, True for real residues, shape (batch, sequence_length)
        ionic_strengths : torch.Tensor
            Ionic strength of each sequence in mM, shape (batch, 1)

        Returns
        -------
        torch.Tensor
            Returns the loss
        """
        b, c, h, w, device = *x.shape, x.device

        # Generate random timestamps to noise the tensor and learn the denoising process
        timestamps = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_loss(x, timestamps, labels, mask, ionic_strengths)

    def _initialize_latent_scaling(self, latent_encoding: torch.Tensor) -> None:
        """
        Set the latent space scaling factor to 1 / std of the first batch of latents
        (averaged over processes), so the scaled latents have roughly unit variance.
        The latents are only rescaled, not centred.

        Parameters
        ----------
        latent_encoding : torch.Tensor
            Batch of encoded latent vectors
        """
        # Calculate local mean and standard deviation
        # local_mean = latent_encoding.mean()
        local_std = latent_encoding.std()

        # Gather from all processes and compute global mean and standard deviation
        # gathered_mean = self.all_gather(local_mean)
        gathered_std = cast(torch.Tensor, self.all_gather(local_std))

        # mean_mean = gathered_mean.mean()
        mean_std = 1 / gathered_std.mean()

        # Update the registered buffers with computed values
        # self.latent_space_mean = mean_mean.float().to(self.device)
        self.latent_space_scaling_factor = mean_std.float().to(self.device)

    def training_step(self, batch: DiffusionBatch, batch_idx: int) -> torch.Tensor:
        """Training step that computes diffusion loss on a batch."""
        latent_encoding, sequences, sequence_attention_mask, ionic_strengths = (
            batch["data"],
            batch["sequence"],
            batch["attention_mask"],
            batch["ionic_strengths"],
        )

        if self.distance_map_encoder is not None:
            latent_encoding = self._encode_distance_maps(latent_encoding)

        # Calculate scaling factor on first batch (only once during training).
        # A model fine-tuned from a checkpoint keeps the factor it was trained
        # with rather than replacing it with an estimate from one new batch.
        if (
            self.global_step == 0
            and batch_idx == 0
            and float(self.latent_space_scaling_factor) == UNSET_LATENT_SCALING_FACTOR
        ):
            self._initialize_latent_scaling(latent_encoding)

        # Z-score the latent encoding
        latent_encoding = latent_encoding * self.latent_space_scaling_factor

        # Compute loss
        loss = self.forward(
            latent_encoding,
            labels=sequences,
            mask=sequence_attention_mask,
            ionic_strengths=ionic_strengths,
        )

        # Log training metrics
        self.log("train_loss", loss, prog_bar=True, batch_size=latent_encoding.size(0))

        return loss

    def validation_step(self, batch: DiffusionBatch, batch_idx: int) -> torch.Tensor:
        """Validation step that evaluates diffusion loss on a batch."""
        latent_encoding, sequences, sequence_attention_mask, ionic_strengths = (
            batch["data"],
            batch["sequence"],
            batch["attention_mask"],
            batch["ionic_strengths"],
        )

        if self.distance_map_encoder is not None:
            latent_encoding = self._encode_distance_maps(latent_encoding)

        # Z-score the latent encoding
        latent_encoding = latent_encoding * self.latent_space_scaling_factor

        # Compute loss
        loss = self.forward(
            latent_encoding,
            labels=sequences,
            mask=sequence_attention_mask,
            ionic_strengths=ionic_strengths,
        )

        self.log(
            "epoch_val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=latent_encoding.size(0),
        )

        return loss

    def compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        alpha = sqrt_alphas_cumprod[timesteps]
        sigma = sqrt_one_minus_alphas_cumprod[timesteps]
        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def configure_optimizers(self):
        """
        Configure the optimizer and the learning rate scheduler for the model.
        The noise predictor and the sequence encoder are trained with AdamW
        (learning rate set_lr, weight decay 0.01); the frozen VAE is not
        optimized. The learning rate scheduler is chosen by config_scheduler.

        Returns
        -------
        List
            Returns the optimizer and the learning rate scheduler

        Raises
        ------
        ValueError
            If the scheduler is not implemented
        """
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.sequence_encoder.parameters()),
            lr=self.set_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
            amsgrad=False,
        )

        if self.config_scheduler == "CosineAnnealingWarmRestarts":
            lr_scheduler = {
                "scheduler": CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, eta_min=1e-4
                ),
                "monitor": self.monitor,
                "interval": "epoch",
            }

        elif self.config_scheduler == "OneCycleLR":
            lr_scheduler = {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=0.01,
                    total_steps=int(self.trainer.estimated_stepping_batches),
                ),
                "monitor": self.monitor,
                "interval": "step",
            }

        elif self.config_scheduler == "CosineAnnealingLR":
            num_epochs = self.trainer.max_epochs
            if num_epochs is None:
                raise RuntimeError("max_epochs must be set for this scheduler")
            lr_scheduler = {
                "scheduler": CosineAnnealingLR(
                    optimizer,
                    T_max=num_epochs,
                    eta_min=1e-8,
                ),
                "monitor": self.monitor,
                "interval": "epoch",
            }
        elif self.config_scheduler == "LinearWarmupCosineAnnealingLR":
            num_epochs = self.trainer.max_epochs
            if num_epochs is None:
                raise RuntimeError("max_epochs must be set for this scheduler")
            total_steps = int(self.trainer.estimated_stepping_batches)
            steps_per_epoch = total_steps // num_epochs
            # Warmup for 1% (WARMUP_FRACTION) of the total steps
            warmup_steps = int(total_steps * WARMUP_FRACTION)

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup phase
                    return current_step / max(1, warmup_steps)
                else:
                    # Cosine annealing phase
                    eta_min = 1e-8
                    remaining_steps = current_step - warmup_steps
                    current_epoch = remaining_steps // steps_per_epoch
                    cosine_factor = 0.5 * (
                        1 + math.cos(math.pi * current_epoch / num_epochs)
                    )
                    return eta_min + (1 - eta_min) * cosine_factor

            lr_scheduler = {
                "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
                "monitor": self.monitor,
                "interval": "step",
            }

        else:
            raise ValueError(f"{self.config_scheduler} lr_scheduler is not implemented")

        return [optimizer], [lr_scheduler]
