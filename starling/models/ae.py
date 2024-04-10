import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

from starling.models import resnets_original, vae_components


class PrintLayer(nn.Module):
    def __init__(self, layer_name):
        super(PrintLayer, self).__init__()
        self.layer_name = layer_name

    def forward(self, x):
        print(f"Intermediate output of {self.layer_name}: {x.shape}")
        return x


torch.set_float32_matmul_precision("high")


class AE(pl.LightningModule):
    def __init__(
        self,
        model,
        dimension,
        loss_type,
        weights_type,
        lr_scheduler,
        set_lr,
        kernel_size=3,
        in_channels=1,
        encoder_block="original",
        decoder_block="original",
    ):
        super().__init__()

        self.save_hyperparameters()

        # Set up the ResNet Encoder and Decoder combinations
        resnets = {
            "Resnet18": {
                "encoder": {
                    "original": resnets_original.Resnet18_Encoder,
                    "modified": vae_components.Resnet18_Encoder,
                },
                "decoder": {
                    "original": resnets_original.Resnet18_Decoder,
                    "modified": vae_components.Resnet18_Decoder,
                },
            },
            "Resnet34": {
                "encoder": {
                    "original": resnets_original.Resnet34_Encoder,
                    "modified": vae_components.Resnet34_Encoder,
                },
                "decoder": {
                    "original": resnets_original.Resnet34_Decoder,
                    "modified": vae_components.Resnet34_Decoder,
                },
            },
            "Resnet50": {
                "encoder": {
                    "original": resnets_original.Resnet50_Encoder,
                    # "modified": vae_components.Resnet50_Encoder,
                },
                "decoder": {
                    "original": resnets_original.Resnet50_Decoder,
                    # "modified": vae_components.Resnet50_Decoder,
                },
            },
            "Resnet101": {
                "encoder": {
                    "original": resnets_original.Resnet101_Encoder,
                    # "modified": vae_components.Resnet101_Encoder,
                },
                "decoder": {
                    "original": resnets_original.Resnet101_Decoder,
                    # "modified": vae_components.Resnet101_Decoder,
                },
            },
            "Resnet152": {
                "encoder": {
                    "original": resnets_original.Resnet152_Encoder,
                    # "modified": vae_components.Resnet152_Encoder,
                },
                "decoder": {
                    "original": resnets_original.Resnet152_Decoder,
                    # "modified": vae_components.Resnet152_Decoder,
                },
            },
        }

        # Loss params
        self.loss_type = loss_type
        self.weights_type = weights_type

        # Learning rate params
        self.config_scheduler = lr_scheduler
        self.set_lr = set_lr

        # these are used to monitor the training losses for the *EPOCH*
        self.total_train_step_losses = []
        self.recon_step_losses = []

        self.monitor = "epoch_val_loss"

        # Encoder
        self.encoder = resnets[model]["encoder"][encoder_block](
            in_channels=in_channels,
            kernel_size=kernel_size,
            dimension=dimension,
        )

        # Decoder
        self.decoder = resnets[model]["decoder"][decoder_block](
            out_channels=in_channels,
            kernel_size=kernel_size,
            dimension=dimension,
        )

        # Params to learn for reconstruction loss
        if self.loss_type == "elbo":
            self.log_std = nn.Parameter(torch.zeros((dimension * (dimension + 1) // 2)))

    def encode(self, data: torch.Tensor):
        data = self.encoder(data)
        return data

    def decode(self, data: torch.Tensor):
        data = self.decoder(data)
        return data

    def get_weights(self, ground_truth, scale):
        if scale == "linear":
            max_distance = ground_truth.max()
            min_distance = ground_truth.min()
            weights = 1 - (ground_truth - min_distance) / (max_distance - min_distance)
            weights = weights / weights.sum()
            return weights
        elif scale == "reciprocal":
            weights = torch.reciprocal(ground_truth)
            weights[weights == float("inf")] = 0
            weights = weights / weights.sum()
            return weights
        else:
            raise ValueError(f"Variable name '{scale}' for get_weights does not exist")

    def symmetrize(self, data_reconstructed):
        upper_triangle = data_reconstructed.triu()
        symmetrized_array = upper_triangle + upper_triangle.t()
        return symmetrized_array.fill_diagonal_(0)

    def gaussian_likelihood(self, data_hat, log_std, data):
        std = torch.exp(log_std)
        mean = data_hat
        input_size = mean.shape[0]
        matrix_std = torch.zeros(input_size, input_size).to(self.device)
        triu_indices = torch.triu_indices(input_size, input_size, offset=0)
        matrix_std[triu_indices[0], triu_indices[1]] = std[
            : input_size * (input_size + 1) // 2
        ]
        matrix_std = matrix_std + matrix_std.t() - torch.diag(matrix_std.diag())
        dist = torch.distributions.Normal(mean, matrix_std)
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(data)
        weights = self.get_weights(data, scale=self.weights_type)
        log_pxz *= weights
        return log_pxz.sum()

    def AE_loss(self, data_reconstructed, data):
        # Find where the padding starts by counting the number of
        start_of_padding = torch.sum(data != 0, dim=(1, 2))[:, 0] + 1

        if self.loss_type == "mse" or self.loss_type == "weighted_mse":
            # Loss function for VAE, here I am removing the padded region
            # from both the ground truth and prediction

            BCE = 0

            for num, padding_start in enumerate(start_of_padding):
                data_reconstructed_no_padding = data_reconstructed[num][0][
                    :padding_start, :padding_start
                ]
                # Make the reconstructed map symmetric so that weights are freed to learn other
                # patterns
                data_reconstructed_no_padding = self.symmetrize(
                    data_reconstructed_no_padding
                )

                # Get unpadded ground truth
                data_no_padding = data[num][0][:padding_start, :padding_start]

                # Mean squared error weighted by ground truth distance
                if self.loss_type == "weighted_mse":
                    weights = self.get_weights(data_no_padding, scale=self.weights_type)

                    mse_loss = F.mse_loss(
                        data_reconstructed_no_padding, data_no_padding, reduction="none"
                    )

                    BCE += (mse_loss * weights).sum()

                # Mean squared error not weighted by ground truth distance
                elif self.loss_type == "mse":
                    BCE += F.mse_loss(
                        data_reconstructed_no_padding, data_no_padding, reduction="mean"
                    )
                else:
                    raise ValueError(
                        f"loss type of name '{self.loss_type}' does not exist"
                    )

            # Taking the mean of the loss (could also be sum)
            BCE /= num + 1

            loss = BCE

            return {"loss": loss, "BCE": BCE}

        elif self.loss_type == "elbo":
            total_recon_loss = 0
            for num, padding_start in enumerate(start_of_padding):
                data_reconstructed_no_padding = data_reconstructed[num][0][
                    :padding_start, :padding_start
                ]
                # Make the reconstructed map symmetric so that weights are
                # freed to learn other patterns
                data_reconstructed_no_padding = self.symmetrize(
                    data_reconstructed_no_padding
                )

                # Get unpadded ground truth
                data_no_padding = data[num][0][:padding_start, :padding_start]

                # Get the reconstruction loss
                recon_loss = self.gaussian_likelihood(
                    data_hat=data_reconstructed_no_padding,
                    log_std=self.log_std,
                    data=data_no_padding,
                )
                total_recon_loss += recon_loss

            # Take the mean of all the losses in batch
            total_recon_loss /= num + 1
            elbo = -total_recon_loss

            return {"loss": elbo, "BCE": total_recon_loss}

    def forward(self, data):
        data = self.encode(data)

        data_reconstructed = self.decode(data)
        return data_reconstructed

    def training_step(self, batch, batch_idx):
        data = batch["input"]

        data_reconstructed = self.forward(data=data)

        loss = self.AE_loss(
            data_reconstructed=data_reconstructed,
            data=data,
        )

        if batch_idx % 100 == 0:
            self.total_train_step_losses.append(loss["loss"])
            self.recon_step_losses.append(loss["BCE"])

        self.log("train_loss", loss["loss"], prog_bar=True)
        self.log("recon_loss", loss["BCE"], prog_bar=True)

        return loss["loss"]

    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.total_train_step_losses).mean()
        self.log("epoch_train_loss", epoch_mean, prog_bar=True, sync_dist=True)

        recon_mean = torch.stack(self.recon_step_losses).mean()
        self.log("epoch_recon_loss", recon_mean, prog_bar=True, sync_dist=True)

        # free up the memory
        self.total_train_step_losses.clear()
        self.recon_step_losses.clear()

    def validation_step(self, batch, batch_idx):
        data = batch["input"]

        data_reconstructed = self.forward(data)

        loss = self.AE_loss(
            data_reconstructed=data_reconstructed,
            data=data,
        )

        self.log("epoch_val_loss", loss["loss"], prog_bar=True, sync_dist=True)

        return loss["loss"]

    def configure_optimizers(self):
        # NVIDIA configs for ResNet50, they used it with CosineAnnealingLR
        # https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.set_lr,  # 0.256 for batch of 256
        #     momentum=0.875,
        #     nesterov=True,
        #     weight_decay=1 / 32768,
        # )

        # Here we are not doing weight decay on batch normalization parameters
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [
                        param
                        for name, param in self.named_parameters()
                        if not any(nd in name for nd in ["bn"])
                    ]
                },
                {
                    "params": [
                        param
                        for name, param in self.named_parameters()
                        if any(nd in name for nd in ["bn"])
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=self.set_lr,  # 0.256 for batch of 256
            momentum=0.875,
            nesterov=True,
            weight_decay=1 / 32768,
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
