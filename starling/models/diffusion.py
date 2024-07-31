from typing import List, Union

import esm
import pytorch_lightning as pl
import torch
import torch.nn as nn
from IPython import embed
from memory_profiler import profile
from torch.cuda.amp import autocast
from torch.functional import F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)
from tqdm import tqdm

from starling.data.data_wrangler import MaxPad, one_hot_encode, symmetrize
from starling.data.esm_embeddings import BatchConverter, esm_embeddings
from starling.data.schedulers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    sigmoid_beta_schedule,
)

# Adapted from https://github.com/Camaltra/this-is-not-real-aerial-imagery/blob/main/src/ai/diffusion_process.py
# and https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py#L720


def extract(
    constants: torch.Tensor, timestamps: torch.Tensor, shape: int
) -> torch.Tensor:
    batch_size = timestamps.shape[0]
    out = constants.gather(-1, timestamps)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(timestamps.device)


torch.set_float32_matmul_precision("high")


class DiffusionModel(pl.LightningModule):
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
        labels="learned-embeddings",
        set_lr=1e-4,
        config_scheduler="CosineAnnealingLR",
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["encoder_model", "model"])

        self.model = model
        self.labels = labels

        # Freeze the encoder model parameters
        self.encoder_model = encoder_model
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        self.encoder_model.eval()

        # self.channels = self.model.config.in_channels
        # self.channels = self.model.in_channels
        self.channels = 1
        self.image_size = image_size

        # Learning rate params
        self.set_lr = set_lr
        self.config_scheduler = config_scheduler

        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler)
        if self.beta_scheduler_fn is None:
            raise ValueError(f"unknown beta schedule {beta_scheduler}")

        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}

        # This will be calculated later on during the first global step
        # Need to register here so pytorch_lightning doesn't freak out
        # Assuming the expected shape for the buffer is [1]
        latent_space_scaling_factor = torch.tensor(1.0, dtype=torch.float32)

        # Register the buffer
        self.register_buffer("latent_space_scaling_factor", latent_space_scaling_factor)

        betas = self.beta_scheduler_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        timesteps, *_ = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = timesteps

        self.monitor = "epoch_val_loss"

        # ESM models currently supported
        self.esm = {
            "esm2_t6_8M_UR50D": {
                "model_name": esm.pretrained.esm2_t6_8M_UR50D(),
                "layers": 6,
                "latent_dim": 320,
            },
            "esm2_t12_35M_UR50D": {
                "model_name": esm.pretrained.esm2_t12_35M_UR50D(),
                "layers": 12,
                "latent_dim": 480,
            },
            "esm2_t30_150M_UR50D": {
                "model_name": esm.pretrained.esm2_t30_150M_UR50D(),
                "layers": 30,
                "latent_dim": 640,
            },
        }

        # ESM2 model to generate labels
        if self.labels in self.esm.keys():
            self.esm_model, self.esm_alphabet = self.esm[self.labels]["model_name"]
            self.esm_layers = self.esm[self.labels]["layers"]

            # Freeze the parameters, we don't want to keep training ESM
            for param in self.esm_model.parameters():
                param.requires_grad = False

            # Don't do any dropout within ESM model
            self.esm_model.eval()

        elif self.labels == "learned-embeddings":
            self.sequence_embedding = nn.Embedding(21, self.model.labels_dim)

    @torch.inference_mode()
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
        batched_timestamps = torch.full(
            (b,), timestamp, device=device, dtype=torch.long
        )

        preds = self.model(x, batched_timestamps, labels)

        betas_t = extract(self.betas, batched_timestamps, x.shape)
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas, batched_timestamps, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, batched_timestamps, x.shape
        )

        predicted_mean = sqrt_recip_alphas_t * (
            x - betas_t * preds / sqrt_one_minus_alphas_cumprod_t
        )

        if timestamp == 0:
            return predicted_mean
        else:
            posterior_variance = extract(
                self.posterior_variance, batched_timestamps, x.shape
            )
            noise = torch.randn_like(x)
            return predicted_mean + torch.sqrt(posterior_variance) * noise

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape: tuple,
        labels,
        steps: int = None,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """
        Sampling loop for the diffusion model. It loops over the timesteps
        to denoise the initial tensor sampled from N(0, I)

        Parameters
        ----------
        shape : tuple
            The shape of the tensor to sample from N(0, I)
        labels : _type_
            Sequence to condition the sampling on
        return_all_timesteps : bool, optional
            Whether to return the full trajectory of denoising,
            by default False

        Returns
        -------
        torch.Tensor
            Returns the fully denoised tensor, in this case a latent
            that can be decoded using a pre-trained VAE
        """

        batch, device = shape[0], self.device

        all_latents = []

        latents = torch.randn(shape, device=device)

        if steps is not None:
            timesteps = (
                torch.linspace(1, self.num_timesteps - 1, steps).round().to(torch.int64)
            )
        else:
            timesteps = range(0, self.num_timesteps)

        for t in tqdm(
            reversed(timesteps),
            desc="Generating Latents",
            total=len(timesteps),
        ):
            if return_all_timesteps:
                all_latents.append(latents)

            latents = self.p_sample(latents, t, labels)

        # Scale the latents back to the original scale
        latents = (1 / self.latent_space_scaling_factor) * latents

        # if not return_all_timesteps else torch.stack(imgs, dim=1)
        if return_all_timesteps:
            return torch.stack(all_latents, dim=1) * (
                1 / self.latent_space_scaling_factor
            )
        else:
            return latents

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        labels,
        steps: int = None,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """
        Sample from the trained diffusion model

        Parameters
        ----------
        batch_size : int
            The batch size to sample, higher the better if enough VRAM
        labels : _type_
            Sequence to condition the sampling on
        return_all_timesteps : bool, optional
            Whether to return all the tensors along the
            denoising trajectory, by default False

        Returns
        -------
        torch.Tensor
            Returns the sampled fully denoised tensor
        """

        shape = (batch_size, self.channels, self.image_size, self.image_size)

        with torch.no_grad():
            if self.labels in self.esm.keys():
                labels = self.sequence2labels([labels])
            elif self.labels == "learned-embeddings":
                labels = (
                    torch.argmax(
                        torch.from_numpy(one_hot_encode(labels.ljust(384, "0"))), dim=-1
                    )
                    .to(torch.int64)
                    .squeeze()
                    .to(self.device)
                )
                labels = self.sequence2labels(labels)

        labels = labels.repeat(batch_size, 1, 1)
        latents = self.p_sample_loop(
            shape, labels, steps=steps, return_all_timesteps=return_all_timesteps
        )

        distance_map = self.encoder_model.decode(latents)

        return distance_map, latents, labels

    # Remove mixed precision from this function, I've experienced numerical instability here
    @autocast(enabled=False)
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

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def sequence2labels(self, sequences: List) -> torch.Tensor:
        """
        Converts sequences to labels based on user defined models,
        It can either use some ESM model to generate labels, one-hot-encode,
        or learn the embeddings.

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
        # Generate labels using one of the ESM models
        if self.labels in self.esm.keys():
            encoded = esm_embeddings(
                self.esm_model,
                self.esm_alphabet,
                sequences,
                self.device,
                self.esm_layers,
            )
        elif self.labels == "learned-embeddings":
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
        predicted noise and the actual noise. The loss can be either L1 or L2.

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

        x_noised = self.q_sample(x_start, t, noise=noise)

        labels = self.sequence2labels(labels)

        # predicted_noise = self.model(x_noised, t, labels)[0]
        predicted_noise = self.model(x_noised, t, labels)

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

        timestamps = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_loss(x, timestamps, labels)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        data, sequences, labels = batch

        with torch.no_grad():
            latent_encoding = self.encoder_model.encode(
                data, labels=labels.unsqueeze(1)
            )
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
        data, sequences, labels = batch

        with torch.no_grad():
            latent_encoding = self.encoder_model.encode(
                data, labels=labels.unsqueeze(1)
            )
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

    @torch.inference_mode()
    def timestep_tests(self, latent_encoding, sequence) -> torch.Tensor:
        # Scale the latent encoding to have unit std
        latent_encoding = self.latent_space_scaling_factor * latent_encoding

        t = torch.arange(0, self.num_timesteps, device=self.device)

        noise = torch.randn_like(latent_encoding)
        x_noised = self.q_sample(latent_encoding, t, noise=noise)

        # nn.Embedding is 0-indexes, so we subtract 1 from the lengths
        lengths = list(map(lambda label: len(label) - 1, [sequence]))

        length_labels = torch.tensor(
            lengths,
            device=latent_encoding.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            length_labels = self.embed_length(length_labels)
            # length_labels = self.length_mlp(length_labels)
            labels = self.sequence2labels([sequence])
            labels = self.mlp(labels)
            labels += length_labels
            predicted_noise = self.model(x_noised, t, labels)[0]

        loss = abs(noise - predicted_noise)
        # loss = loss.mean(dim=0)

        return x_noised

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

        else:
            raise ValueError(f"{self.config_scheduler} lr_scheduler is not implemented")

        return [optimizer], [lr_scheduler]


def reduce_sampling_steps(T, K):
    # Calculate the spacing between each sample
    spacing = (T - 1) / (K - 1)

    # Initialize an empty list to store the rounded numbers
    samples = []

    # Generate K evenly spaced real numbers between 1 and T (inclusive)
    for i in range(K):
        # Calculate the current sample
        sample = 1 + spacing * i

        # Round the sample to the nearest integer
        rounded_sample = round(sample)

        # Append the rounded sample to the list
        samples.append(rounded_sample)

    return samples
