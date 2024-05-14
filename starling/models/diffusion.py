import math
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from IPython import embed
from torch.functional import F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)
from tqdm import tqdm

from starling.data.schedulers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    sigmoid_beta_schedule,
)

# Adapted from https://github.com/Camaltra/this-is-not-real-aerial-imagery/blob/main/src/ai/diffusion_process.py


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
        image_size: int,
        *,
        beta_scheduler: str = "linear",
        timesteps: int = 1000,
        schedule_fn_kwargs: Union[dict, None] = None,
    ) -> None:
        super().__init__()
        self.model = model

        self.channels = self.model.in_channels
        self.image_size = image_size

        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler)
        if self.beta_scheduler_fn is None:
            raise ValueError(f"unknown beta schedule {beta_scheduler}")

        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}

        betas = self.beta_scheduler_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        timesteps, *_ = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = timesteps

        self.monitor = "epoch_val_loss"

    @torch.inference_mode()
    def p_sample(self, x: torch.Tensor, timestamp: int) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        batched_timestamps = torch.full(
            (b,), timestamp, device=device, dtype=torch.long
        )

        preds = self.model(x, batched_timestamps)

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
        self, shape: tuple, return_all_timesteps: bool = False
    ) -> torch.Tensor:
        batch, device = shape[0], "mps"

        img = torch.randn(shape, device=device)
        # This cause me a RunTimeError on MPS device due to MPS back out of memory
        # No ideas how to resolve it at this point

        # imgs = [img]

        for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            img = self.p_sample(img, t)
            # imgs.append(img)

        ret = img  # if not return_all_timesteps else torch.stack(imgs, dim=1)

        return ret

    def sample(
        self, batch_size: int = 16, return_all_timesteps: bool = False
    ) -> torch.Tensor:
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        return self.p_sample_loop(shape, return_all_timesteps=return_all_timesteps)

    def q_sample(
        self, x_start: torch.Tensor, t: int, noise: torch.Tensor = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def symmetrize(self, data_reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Symmetrizes the reconstructed data so that the weights can learn other patterns.
        Loss calculated only on the reconstruction faithfulness of the upper triangle
        of the distance map

        Parameters
        ----------
        data_reconstructed : torch.Tensor
            Reconstructed data; output of the decoder

        Returns
        -------
        torch.Tensor
            Symmetric version of the reconstructed data
        """
        upper_triangle = data_reconstructed.triu()
        symmetrized_array = upper_triangle + upper_triangle.t()
        return symmetrized_array.fill_diagonal_(0)

    def calculate_loss(
        self,
        data: torch.Tensor,
        noise: torch.Tensor,
        noise_predicted: torch.Tensor,
    ) -> dict:
        """
        Calculates the loss of the VAE, using the sum between the KLD loss
        of the latent space to N(0, I) and either mean squared error
        between the reconstructed data and the ground truth or
        the negative log likelihood of the input data given the latent space
        under a Gaussian assumption. Additional loss is added to ensure the
        contacts are reconstructed correctly.

        Parameters
        ----------
        data_reconstructed : torch.Tensor
            Reconstructed data; output of the VAE
        data : torch.Tensor
            Ground truth data, input to the VAE
        mu : torch.Tensor
            Means of the normal distributions of the latent space
        logvar : torch.Tensor
            Log variances of the normal distributions of the latent space
        KLD_weight : int, optional
            How much to weight the importance of the regularization term of the
            latent space. Setting this to lower than 1 will lead to less regular
            and interpretable latent space, by default None

        Returns
        -------
        dict
            Returns a dictionary containing the total loss, reconstruction loss, and KLD loss

        Raises
        ------
        ValueError
            If the loss type is not mse or elbo
        """

        # Find where the padding starts by counting the number of
        start_of_padding = torch.sum(data != 0, dim=(1, 2))[:, 0] + 1

        # Initialize the losses
        recon = 0

        # Input is padded, so the padding needs to be removed before calculating loss
        for num, padding_start in enumerate(start_of_padding):
            noise_predicted_no_padding = noise_predicted[num][0][
                :padding_start, :padding_start
            ]

            # Make the reconstructed map symmetric so that weights are freed to learn other
            # patterns
            noise_predicted_no_padding = self.symmetrize(noise_predicted_no_padding)

            # Get unpadded ground truth
            noise_no_padding = noise[num][0][:padding_start, :padding_start]
            noise_no_padding = self.symmetrize(noise_no_padding)

            # Get the weights for the loss
            # weights = self.get_weights(data_no_padding, scale=self.weights_type)

            # Mean squared error weighted by ground truth distance
            mse_loss = F.mse_loss(
                noise_no_padding, noise_predicted_no_padding, reduction="none"
            )
            recon += mse_loss.sum()

        recon /= num + 1

        return recon

    def p_loss(
        self,
        x_start: torch.Tensor,
        t: int,
        labels: torch.Tensor = None,
        noise: torch.Tensor = None,
        loss_type: str = "l2",
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noised = self.q_sample(x_start, t, noise=noise)
        if labels is not None:
            x_noised = torch.cat([x_noised, labels], dim=1)

        predicted_noise = self.model(x_noised, t)

        if loss_type == "l2":
            # loss = F.mse_loss(noise, predicted_noise, reduction="sum")
            loss = self.calculate_loss(
                data=x_start, noise=noise, noise_predicted=predicted_noise
            )
        elif loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        else:
            raise ValueError(f"unknown loss type {loss_type}")
        return loss

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        b, c, h, w, device, img_size = *x.shape, x.device, self.image_size
        assert h == w == img_size, f"image size must be {img_size}"

        timestamps = torch.randint(0, self.num_timesteps, (b,)).long().to(device)

        return self.p_loss(x, timestamps, labels)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        data = batch["data"]
        labels = batch["encoder_condition"]

        loss = self.forward(data, labels)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        data = batch["data"]
        labels = batch["encoder_condition"]

        loss = self.forward(data, labels)

        self.log("epoch_val_loss", loss, prog_bar=True, sync_dist=True)

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

        optimizer_params = [
            {
                "params": [
                    param
                    for name, param in self.named_parameters()
                    if not any(nd in name for nd in ["bn"]) and name != "log_std"
                ],
                "weight_decay": 1 / 32768,  # Include weight decay for other parameters
            },
            {
                "params": [
                    param
                    for name, param in self.named_parameters()
                    if any(nd in name for nd in ["bn"])
                ],
                "weight_decay": 0.0,  # Exclude weight decay for parameters with 'bn' in name
            },
        ]

        self.set_lr = 0.01
        self.config_scheduler = "CosineAnnealingLR"

        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=self.set_lr,
            momentum=0.875,
            nesterov=True,
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
                    eta_min=1e-4,
                ),
                "monitor": self.monitor,
                "interval": "epoch",
            }
        else:
            raise ValueError(f"{self.config_scheduler} lr_scheduler is not implemented")

        return [optimizer], [lr_scheduler]
