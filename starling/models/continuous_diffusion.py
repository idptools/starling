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
        encoder_model,
        set_lr,
        config_scheduler,
        features=512,
        noise_schedule="karras",
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        labels="None",
    ):
        super().__init__()

        # Save the hyperparameters of the model but ignore the encoder_model and the U-Net model
        self.save_hyperparameters(ignore=["encoder_model", "model"])

        self.model = model
        self.encoder_model = encoder_model

        for param in self.encoder_model.parameters():
            param.requires_grad = False
        self.encoder_model.eval()

        self.features = features
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
        t: int,
        labels: torch.Tensor = None,
        noise: torch.Tensor = None,
        masks: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        A function that runs the model and calculates the loss based on the
        predicted noise and the actual noise. The loss can either be L1 or L2.

        Parameters
        ----------
        x_start : torch.Tensor
            The starting image tensor
        t : int
            The timestep along the denoising-diffusion process
        labels : torch.Tensor, optional
            Labels to condition the model on, by default None
        noise : torch.Tensor, optional
            Sampled noise from N(0,I), by default None
        loss_type : str, optional
            The type of loss to calculate between the
            amount of added noise and predicted noise, by default "l2"

        Returns
        -------
        torch.Tensor
            Returns the loss

        Raises
        ------
        ValueError
            If the loss type is not one of the two options (l1, l2)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            # Offset noise that seems to improve the inference
            # According to https://www.crosslabs.org/blog/diffusion-with-offset-noise
            # noise += 0.1 * torch.randn(
            #     x_start.shape[0], x_start.shape[1], 1, 1, device=self.device
            # )

        # Noise the input data
        x, log_snr = self.q_sample(x_start=x_start, times=t, noise=noise)

        # Get the labels to condition the model on
        labels = self.sequence2labels(labels)

        # Run the model to predict the noise
        predicted_noise = self.model(x, log_snr, labels)

        losses = F.mse_loss(predicted_noise, noise, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(min=self.min_snr_gamma) / snr
            losses = losses * loss_weight

        return losses.mean()

    def forward(self, x, labels, masks=None):
        b = x.shape[0]

        times = self.random_times(b)

        return self.p_losses(x, times, labels, masks)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        data, sequences = batch

        with torch.no_grad():
            latent_encoding = self.encoder_model.encode(data)
            latent_encoding = latent_encoding.sample()

        # Figure out the standard deviation of the latent space using
        # the first batch of the data. This is to scale it to have unit variance
        # stabilizes denoising-diffusion training
        if self.global_step == 0 and batch_idx == 0:
            latent_space_scaling_factor = 1 / latent_encoding.std()
            self.latent_space_scaling_factor = latent_space_scaling_factor.float().to(
                self.device
            )

        # Scale the latent encoding to have unit std
        latent_encoding = self.latent_space_scaling_factor * latent_encoding

        loss = self.forward(latent_encoding, labels=sequences)

        self.log("train_loss", loss, prog_bar=True, batch_size=data.size(0))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        data, sequences = batch

        with torch.no_grad():
            latent_encoding = self.encoder_model.encode(data)
            latent_encoding = latent_encoding.sample()

        # Scale the latent encoding to have unit std
        latent_encoding = self.latent_space_scaling_factor * latent_encoding

        loss = self.forward(latent_encoding, labels=sequences)

        self.log(
            "epoch_val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=data.size(0),
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
