import math
from typing import List, Union

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

torch.set_float32_matmul_precision("high")


# Helper function
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

    def __init__(
        self,
        unet_model: nn.Module,
        sequence_encoder: nn.Module,
        distance_map_encoder: nn.Module,
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
            A neural network model that takes in an image, a timestamp, and optionally labels to condition on
            and outputs the predicted noise
        encoder_model : nn.Module
            A VAE model that takes in the data (e.g., a distance map) and outputs the compressed representation of
            the data (e.g., a latent space). The denoising-diffusion model is then trained to denoise the latent space.
        image_size : int
            The size of the latent space (height and width)
        beta_scheduler : str, optional
            The name of the beta scheduler to use, by default "cosine"
        timesteps : int, optional
            The number of timesteps to run the diffusion process, by default 1000
        schedule_fn_kwargs : Union[dict, None], optional
            Additional arguments to pass to the beta scheduler function, by default None
        labels : str, optional
            The type of labels to condition the model on, by default "learned-embeddings"
        set_lr : float, optional
            The initial learning rate for the optimizer, by default 1e-4
        config_scheduler : str, optional
            The name of the learning rate scheduler to use, by default "CosineAnnealingLR"

        Raises
        ------
        ValueError
            If the beta scheduler is not implemented
        """
        super().__init__()

        # Save the hyperparameters of the model but ignore the encoder_model and the U-Net model
        self.save_hyperparameters(
            ignore=["unet_model", "sequence_encoder", "distance_map_encoder"]
        )

        self.unet_model = unet_model
        self.sequence_encoder = sequence_encoder

        if distance_map_encoder is not None:
            self.distance_map_encoder = VAE.load_from_checkpoint(distance_map_encoder)

            self.__freeze_distance_map_encoder()
        else:
            self.distance_map_encoder = None

        # self.in_channels = self.unet_model.in_channels

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
            "latent_space_scaling_factor", torch.tensor(1.0, dtype=torch.float32)
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

    def __freeze_distance_map_encoder(self):
        self.distance_map_encoder.eval()
        for param in self.distance_map_encoder.parameters():
            param.requires_grad = False

    # Remove mixed precision from this function, I've experienced numerical instability here
    @autocast(device_type="cuda", enabled=False)
    def q_sample(
        self, x_start: torch.Tensor, t: int, noise: torch.Tensor = None
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

    def sequence2labels(self, sequences: List, sequence_mask) -> torch.Tensor:
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
        encoded = self.sequence_encoder(sequences, sequence_mask)

        return encoded

    def p_loss(
        self,
        x_start: torch.Tensor,
        t: int,
        labels: torch.Tensor,
        mask: torch.Tensor,
        noise: torch.Tensor = None,
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
        x_noised = self.q_sample(x_start, t, noise=noise)

        # Get the labels to condition the model on
        labels = self.sequence2labels(labels, mask)

        # Run the model to predict the noise
        predicted_noise = self.unet_model(x_noised, t, labels, mask)

        # The following adapted from:
        # https://github.com/huggingface/diffusers/blob/78a78515d64736469742e5081337dbcf60482750/examples/text_to_image/train_text_to_image.py#L927
        if self.min_snr_loss:
            # Apply min-SNR weighting as per Section 3.4 of https://arxiv.org/abs/2303.09556
            # This improves training stability by reweighting timestep losses
            snr = self.compute_snr(t)

            # Calculate weight using min(snr, γ) / snr formula
            # Handle zero SNR case by replacing potential infinities with 1.0
            snr_weight = torch.clamp(self.min_snr_gamma / snr, min=1.0)

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

    def forward(self, x: torch.Tensor, labels: torch.Tensor, mask) -> torch.Tensor:
        """
        Forward pass of the model, calculates the loss based on the
        predicted noise and the actual noise.

        Parameters
        ----------
        x : torch.Tensor
            The starting tensor to noise/denoise
        labels : torch.Tensor, optional
            Sequences to condition the model on, by default None

        Returns
        -------
        torch.Tensor
            Returns the loss
        """
        b, c, h, w, device = *x.shape, x.device

        # Generate random timestamps to noise the tensor and learn the denoising process
        timestamps = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_loss(x, timestamps, labels, mask)

    def _initialize_latent_scaling(self, latent_encoding: torch.Tensor) -> None:
        """
        Initialize the latent space mean and standard deviation using the first batch
        for z-scoring (standardization).

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
        gathered_std = self.all_gather(local_std)

        # mean_mean = gathered_mean.mean()
        mean_std = 1 / gathered_std.mean()

        # Update the registered buffers with computed values
        # self.latent_space_mean = mean_mean.float().to(self.device)
        self.latent_space_scaling_factor = mean_std.float().to(self.device)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        latent_encoding, sequences, sequence_attention_mask = (
            batch["data"],
            batch["sequence"],
            batch["attention_mask"],
        )

        if self.distance_map_encoder is not None:
            with torch.no_grad():
                latent_encoding = self.distance_map_encoder.encode(
                    latent_encoding
                ).mode()

        # Calculate scaling factor on first batch (only once during training)
        if self.global_step == 0 and batch_idx == 0:
            self._initialize_latent_scaling(latent_encoding)

        # Z-score the latent encoding
        latent_encoding = latent_encoding * self.latent_space_scaling_factor

        # Compute loss
        loss = self.forward(
            latent_encoding, labels=sequences, mask=sequence_attention_mask
        )

        # Log training metrics
        self.log("train_loss", loss, prog_bar=True, batch_size=latent_encoding.size(0))

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        latent_encoding, sequences, sequence_attention_mask = (
            batch["data"],
            batch["sequence"],
            batch["attention_mask"],
        )
        if self.distance_map_encoder is not None:
            with torch.no_grad():
                latent_encoding = self.distance_map_encoder.encode(
                    latent_encoding
                ).mode()

        # Z-score the latent encoding
        latent_encoding = latent_encoding * self.latent_space_scaling_factor

        loss = self.forward(
            latent_encoding, labels=sequences, mask=sequence_attention_mask
        )

        self.log(
            "epoch_val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=latent_encoding.size(0),
        )

        return loss

    def compute_snr(self, timesteps):
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
            list(self.unet_model.parameters())
            + list(self.sequence_encoder.parameters()),
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
            warmup_steps = steps_per_epoch * int(num_epochs * 0.01)

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
