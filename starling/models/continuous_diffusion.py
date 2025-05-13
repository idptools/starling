import math
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import reduce, repeat
from torch import nn, sqrt
from torch.amp import autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
)
from torch.special import expm1

# Adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/continuous_time_gaussian_diffusion.py

# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# diffusion helpers


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


# continuous schedules

# equations are taken from https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material
# @crowsonkb Katherine's repository also helped here https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

# log(snr) that approximates the original linear schedule


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t**2)))


def alpha_cosine_log_snr(t, s=0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


# From paper https://arxiv.org/abs/2206.00364; equation 5
def karras_log_snr(t, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """
    Implements the noise schedule from Karras et al. (2022)
    "Elucidating the Design Space of Diffusion-Based Generative Models"
    """
    # Convert t from [0,1] to the sigma space
    inverse_rho = 1.0 / rho
    sigma = sigma_min**inverse_rho + t * (
        sigma_max**inverse_rho - sigma_min**inverse_rho
    )
    sigma = sigma**rho

    # Convert sigma to log(SNR)
    return -2 * torch.log(sigma)


class ContinuousDiffusion(pl.LightningModule):
    def __init__(
        self,
        model,
        set_lr,
        config_scheduler,
        noise_schedule="karras",
        min_snr_loss_weight=False,
        min_snr_gamma=5,
    ):
        super().__init__()

        # Save the hyperparameters of the model but ignore the encoder_model and the U-Net model
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        self.set_lr = set_lr
        self.config_scheduler = config_scheduler

        self.monitor = "epoch_val_loss"

        # continuous noise schedule related stuff

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == "karras":
            self.log_snr = karras_log_snr
        else:
            raise ValueError(f"unknown noise schedule {noise_schedule}")

        # proposed https://arxiv.org/abs/2303.09556
        # can converge 3.4 times faster than baseline if used

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        self.sequence_embedding = nn.Embedding(21, self.model.labels_dim)

        latent_space_scaling_factor = torch.tensor(1.0, dtype=torch.float32)

        # Register the buffer
        self.register_buffer("latent_space_scaling_factor", latent_space_scaling_factor)

    @property
    def device(self):
        return next(self.model.parameters()).device

    # training related functions - noise prediction

    def sequence2labels(self, sequences: List) -> torch.Tensor:
        """
        Converts sequences to labels based on user defined models,

        Parameters
        ----------
        sequences : List
            A list of sequences to convert to labels

        Returns
        -------
        torch.Tensor
            Returns the labels for the decoder

        Raises
        ------
        ValueError
            If the labels are not one of the three options
        """

        encoded = self.sequence_embedding(sequences)

        return encoded

    @autocast("cuda", enabled=False)
    def q_sample(self, x_start, times, masks=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma

        if masks is not None:
            x_noised = x_noised * masks + x_start * (1 - masks)

        return x_noised, log_snr

    def random_times(self, batch_size):
        # times are now uniform from 0 to 1
        return torch.zeros((batch_size,), device=self.device).float().uniform_(0, 1)

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor = None,
        noise: torch.Tensor = None,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate model loss based on predicted vs actual noise.

        Parameters
        ----------
        x_start : torch.Tensor
            The starting tensor to denoise
        t : torch.Tensor
            Timesteps along the denoising-diffusion process
        labels : torch.Tensor, optional
            Condition labels for the model
        noise : torch.Tensor, optional
            Optional pre-defined noise, otherwise sampled from N(0,I)
        masks : torch.Tensor, optional
            Optional masks for conditional generation

        Returns
        -------
        torch.Tensor
            Mean MSE loss between predicted and actual noise
        """
        # Use standard normal distribution if no noise provided
        noise = torch.randn_like(x_start) if noise is None else noise

        # Apply noise according to timestep
        noised_input, log_snr = self.q_sample(x_start=x_start, times=t, noise=noise)

        # Prepare condition labels
        condition_labels = self.sequence2labels(labels)

        # Predict the noise
        predicted_noise = self.model(noised_input, log_snr, condition_labels)

        # Calculate per-element loss and reduce to per-batch loss
        per_element_loss = F.mse_loss(predicted_noise, noise, reduction="none")
        per_batch_loss = reduce(per_element_loss, "b ... -> b", "mean")

        # Apply minimum SNR loss weighting if enabled
        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(min=self.min_snr_gamma) / snr
            per_batch_loss = per_batch_loss * loss_weight

        return per_batch_loss.mean()

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass that samples random timesteps and calculates loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        labels : torch.Tensor
            Condition labels
        masks : torch.Tensor, optional
            Optional masks

        Returns
        -------
        torch.Tensor
            Loss value
        """
        batch_size = x.shape[0]
        random_timesteps = self.random_times(batch_size)

        return self.p_losses(x, random_timesteps, labels, masks=masks)

    def _initialize_latent_scaling(self, latent_encoding: torch.Tensor) -> None:
        """
        Initialize the latent space scaling factor using the first batch.

        Parameters
        ----------
        latent_encoding : torch.Tensor
            Batch of encoded latent vectors
        """
        # Calculate local standard deviation
        local_std = latent_encoding.std()

        # Gather from all processes and compute global standard deviation
        gathered_std = self.all_gather(local_std)
        mean_std = gathered_std.mean()

        # Set consistent scaling factor across all GPUs
        scaling_factor = 1 / mean_std
        self.latent_space_scaling_factor = scaling_factor.float().to(self.device)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step that encodes inputs and calculates diffusion loss.

        Parameters
        ----------
        batch : torch.Tensor
            Batch containing data and sequence labels
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Training loss
        """
        latent_encoding, sequences = batch

        # Initialize scaling factor on first batch
        if self.global_step == 0 and batch_idx == 0:
            self._initialize_latent_scaling(latent_encoding)

        # Scale latent vectors to have unit standard deviation
        normalized_latents = self.latent_space_scaling_factor * latent_encoding

        # Calculate diffusion loss
        loss = self.forward(normalized_latents, labels=sequences)

        # Log the training loss
        self.log("train_loss", loss, prog_bar=True, batch_size=latent_encoding.size(0))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        latent_encoding, sequences = batch

        # Scale the latent encoding to have unit std
        latent_encoding = self.latent_space_scaling_factor * latent_encoding

        loss = self.forward(latent_encoding, labels=sequences)

        self.log(
            "epoch_val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=latent_encoding.size(0),
        )

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and the learning rate scheduler for the model.
        Here I am using NVIDIA suggested settings for learning rate and weight
        decay. For ResNet50 they have seen best performance with CosineAnnealingLR,
        initial learning rate of 0.256 for batch size of 256 and linearly scaling
        it down/up for other batch sizes. The weight decay is set to 1/32768 for all
        parameters except the batch normalization layers. For further information check:
        https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch

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
            self.model.parameters(),
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
                    total_steps=self.trainer.estimated_stepping_batches,
                ),
                "monitor": self.monitor,
                "interval": "step",
            }

        elif self.config_scheduler == "CosineAnnealingLR":
            num_epochs = self.trainer.max_epochs
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
            total_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = total_steps // num_epochs
            # Warmup for 5% of the total steps
            warmup_steps = steps_per_epoch * int(num_epochs * 0.05)

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
