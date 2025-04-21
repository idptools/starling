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
from tqdm import tqdm

from starling.data.data_wrangler import one_hot_encode
from starling.data.schedulers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    sigmoid_beta_schedule,
)

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
        model: nn.Module,
        encoder_model: nn.Module,
        image_size: int,
        *,
        beta_scheduler: str = "cosine",
        timesteps: int = 1000,
        schedule_fn_kwargs: Union[dict, None] = None,
        labels: str = "learned-embeddings",
        set_lr: float = 1e-4,
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
        self.save_hyperparameters(ignore=["encoder_model", "model"])

        self.model = model
        self.labels = labels

        # Freeze the encoder model parameters we don't want to keep training it (should already be trained)
        self.encoder_model = encoder_model
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        self.encoder_model.eval()

        self.in_channels = self.model.in_channels
        self.image_size = image_size

        # Learning rate params
        self.set_lr = set_lr
        self.config_scheduler = config_scheduler

        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler)
        if self.beta_scheduler_fn is None:
            raise ValueError(f"unknown beta schedule {beta_scheduler}")

        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}

        # Register scaling factor buffer (calculated during first training step)
        # Used to normalize latent space to unit variance per Reference #3
        self.register_buffer(
            "latent_space_scaling_factor", torch.tensor(1.0, dtype=torch.float32)
        )

        # Calculate diffusion process parameters
        betas = self.beta_scheduler_fn(timesteps, **schedule_fn_kwargs)
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

        # Set up sequence embedding if using learned embeddings
        if self.labels == "learned-embeddings":
            self.sequence_embedding = nn.Embedding(21, self.model.labels_dim)

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
        if self.labels == "learned-embeddings":
            encoded = self.sequence_embedding(sequences)

        return encoded

    def p_loss(
        self,
        x_start: torch.Tensor,
        t: int,
        labels: torch.Tensor = None,
        noise: torch.Tensor = None,
        loss_type: str = "l2",
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
        x_noised = self.q_sample(x_start, t, noise=noise)

        # Get the labels to condition the model on
        labels = self.sequence2labels(labels)

        # Run the model to predict the noise
        predicted_noise = self.model(x_noised, t, labels)

        # Calculate the loss based on the predicted noise and the actual noise
        if loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        else:
            raise ValueError(f"unknown loss type {loss_type}")

        return loss

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
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
        b, c, h, w, device, img_size = *x.shape, x.device, self.image_size
        assert h == w == img_size, f"image size must be {img_size}"

        # Generate random timestamps to noise the tensor and learn the denoising process
        timestamps = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_loss(x, timestamps, labels)

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
