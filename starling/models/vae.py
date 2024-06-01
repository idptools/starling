from typing import List, Tuple

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

from starling.data.data_wrangler import one_hot_encode
from starling.data.distributions import DiagonalGaussianDistribution
from starling.models import resnets_original, vae_components


class PrintLayer(nn.Module):
    def __init__(self, layer_name):
        super(PrintLayer, self).__init__()
        self.layer_name = layer_name

    def forward(self, x):
        print(f"Intermediate output of {self.layer_name}: {x.shape}")
        return x


torch.set_float32_matmul_precision("high")


class VAE(pl.LightningModule):
    def __init__(
        self,
        model_type,
        in_channels,
        latent_dim,
        dimension,
        loss_type,
        weights_type,
        KLD_weight,
        lr_scheduler,
        set_lr,
        labels=None,
        norm="instance",
        encoder_model="modified",
        decoder_model="modified",
        base=64,
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

        # Input dimensions
        self.dimension = dimension

        # Loss params
        self.loss_type = loss_type
        self.weights_type = weights_type

        # Learning rate params
        self.config_scheduler = lr_scheduler
        self.set_lr = set_lr

        # KLD loss params
        self.KLD_weight = KLD_weight

        # these are used to monitor the training losses for the *EPOCH*
        self.total_train_step_losses = 0
        self.recon_step_losses = 0
        self.KLD_step_losses = 0
        self.num_batches = 0

        self.monitor = "epoch_val_loss"

        # Set up whether we are turning on cross-attention or not
        self.labels = labels
        conditional = True if self.labels is not None else False
        conditional_dim = 512 if self.labels is not None else None

        # Encoder
        encoder_chanel = in_channels
        self.encoder = resnets[encoder_model]["encoder"][model_type](
            in_channels=encoder_chanel, base=base, norm=norm
        )

        # This is usually 4 in ResNets
        num_stages = 4
        expansion = self.encoder.block_type.expansion
        exponent = num_stages - 1 if expansion == 1 else num_stages + 1

        # Spatial size of the distance map after the final encoding layer
        self.compressed_size = dimension / (2**num_stages)
        final_channels = int(base * 2**exponent)
        self.shape_from_final_encoding_layer = (
            final_channels,
            self.compressed_size,
            self.compressed_size,
        )

        # Latent space (some parts are hard coded in - need to change this)
        self.encoder_to_latent = nn.Sequential(
            nn.Conv2d(
                final_channels, 2 * latent_dim, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(2 * latent_dim, 2 * latent_dim, kernel_size=1, stride=1),
        )

        self.latent_to_decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1),
            nn.Conv2d(latent_dim, final_channels, kernel_size=3, stride=1, padding=1),
        )

        # Decoder
        self.decoder = resnets[decoder_model]["decoder"][model_type](
            out_channels=in_channels, dimension=dimension, base=base, norm=norm
        )

        if labels is not None:
            self.sequence_embedding = nn.Embedding(
                num_embeddings=21,
                embedding_dim=conditional_dim,
            )
        # Params to learn for reconstruction loss
        if self.loss_type == "nll":
            self.log_std = nn.Parameter(torch.zeros(dimension, dimension))

    def encode(
        self, data: torch.Tensor, labels: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Takes the data and encodes it into the latent space,
        by returning the mean and log variance

        Parameters
        ----------
        data : torch.Tensor
            Data in the shape of (batch, channel, height, width)
        labels : torch.Tensor
            Labels to be concatenated with the data

        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor]]
            Return the mean and log variance of the latent space
        """

        data = self.encoder(data)
        data = self.encoder_to_latent(data)
        moments = DiagonalGaussianDistribution(data)

        return moments

    def decode(self, latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent space back into the original data

        Parameters
        ----------
        latents : torch.Tensor
            latents in the shape of (batch, channel, height, width)
        labels : torch.Tensor
            Labels to be concatenated with the data

        Returns
        -------
        torch.Tensor
            Returns the reconstructed data
        """
        latents = self.latent_to_decoder(latents)

        latents = self.decoder(latents)
        return latents

    def gaussian_likelihood(
        self,
        data_reconstructed: torch.Tensor,
        log_std: torch.Tensor,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the likelihood of input data given latent space (p(x|z))
        under Gaussian assumption. The reconstructured data is treated as the mean
        of the Gaussian distributions and the log_std is a tensor of learned log standard
        deviations.

        Parameters
        ----------
        data_reconstructed : torch.Tensor
            A tensor containing the reconstructed data that will be treated as the mean to
            parameterize the Gaussian distribution
        log_std : torch.Tensor
            Learned the log standard deviations of the Gaussian distribution
        data : torch.Tensor
            The ground truth data that the likelihood will be calculated against

        Returns
        -------
        torch.Tensor
            Returns the likelihood of the input data given the latent space
        """

        # Create the normal distributions
        dist = torch.distributions.Normal(data_reconstructed, torch.exp(log_std))

        # Calculate log probability of seeing image under p(x|z)
        log_pxz = dist.log_prob(data)

        return log_pxz

    def vae_loss(
        self,
        data_reconstructed: torch.Tensor,
        data: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        KLD_weight: int = None,
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
        if KLD_weight is None:
            KLD_weight = float(self.KLD_weight)

        # Find out where 0s are in the data
        mask = (data != 0).float()
        # Remove the lower triangle of the mask so that loss is only calculated on the upper triangle of the distance map
        mask = mask - mask.tril()

        # Mean squared error weighted by ground truth distance
        if self.loss_type == "mse":
            recon = F.mse_loss(data_reconstructed, data, reduction="none")

        # Negative log likelihood of the input data given the latent space
        elif self.loss_type == "nll":
            # Get the reconstruction loss and convert it to positive values
            recon = -1 * self.gaussian_likelihood(
                data_reconstructed=data_reconstructed,
                log_std=self.log_std,
                data=data,
            )
        else:
            raise ValueError(
                f"loss type of name '{self.loss_type}' does not exist. Current implementations include 'mse' and 'nll'"
            )

        # Calculate the loss of only part of the distance map and take the mean
        recon = recon * mask
        recon = torch.sum(recon) / torch.sum(mask)

        # For more information of KLD loss check out Appendix B:
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        KLD = torch.logsumexp(KLD, dim=0) / mu.size(0)  # Mean over batch

        loss = recon + KLD_weight * KLD

        return {"loss": loss, "recon": recon, "KLD": KLD}

    def forward(
        self, data: torch.Tensor, labels: torch.Tensor = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the VAE

        Parameters
        ----------
        data : torch.Tensor
            Data in the shape of (batch, channel, height, width) to pass through the VAE

        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            Returns the reconstructed data, the mean of the latent space, and the log variance
        """

        moments = self.encode(data, labels)

        latent_encoding = moments.sample()

        data_reconstructed = self.decode(latent_encoding, labels)

        return data_reconstructed, moments

    def training_step(self, batch: dict, batch_idx) -> torch.Tensor:
        """
        Training step of the VAE compatible with Pytorch Lightning

        Parameters
        ----------
        batch : dict
            A batch of data read in using the DataLoader
        batch_idx : _type_
            Batch number the model is on during training

        Returns
        -------
        torch.Tensor
            Total training loss of this batch
        """
        data = batch["data"]

        # Prepare the labels for the model
        if self.labels is not None:
            sequences = batch["sequence"]
            sequences = [seq.ljust(self.dimension, "0") for seq in sequences]
            sequences = torch.argmax(
                torch.from_numpy(one_hot_encode(sequences)).to(self.device), dim=-1
            )
            sequences = self.sequence_embedding(sequences)
        else:
            sequences = None

        data_reconstructed, moments = self.forward(data=data, labels=sequences)

        loss = self.vae_loss(
            data_reconstructed=data_reconstructed,
            data=data,
            mu=moments.mean,
            logvar=moments.logvar,
        )

        self.total_train_step_losses += loss["loss"].item()
        self.recon_step_losses += loss["recon"].item()
        self.KLD_step_losses += loss["KLD"].item()
        self.num_batches += 1

        self.log("train_loss", loss["loss"], prog_bar=True, batch_size=data.size(0))
        self.log("recon_loss", loss["recon"], prog_bar=True, batch_size=data.size(0))

        return loss["loss"]

    def on_train_epoch_end(self) -> None:
        """
        At the end of the epoch, calculate the mean of the training losses.
        Clear the lists that have been filled with losses during the epoch
        for memory management.
        """
        epoch_mean = self.total_train_step_losses / self.num_batches
        self.log("epoch_train_loss", epoch_mean, prog_bar=True, sync_dist=True)

        recon_mean = self.recon_step_losses / self.num_batches
        self.log("epoch_recon_loss", recon_mean, prog_bar=True, sync_dist=True)

        KLD_mean = self.KLD_step_losses / self.num_batches
        self.log("epoch_KLD_loss", KLD_mean, prog_bar=True, sync_dist=True)

        # Reset the total losses
        self.total_train_step_losses = 0
        self.recon_step_losses = 0
        self.KLD_step_losses = 0
        self.num_batches = 0

    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        """
        Validation step of the VAE compatible with Pytorch Lightning. This is
        called after each epoch.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of data read in using the DataLoader
        batch_idx : _type_
            Batch number the model is on during the validation of the model

        Returns
        -------
        torch.Tensor
            Total validation loss of this batch
        """
        data = batch["data"]

        data_reconstructed, moments = self.forward(data=data)

        loss = self.vae_loss(
            data_reconstructed=data_reconstructed,
            data=data,
            mu=moments.mean,
            logvar=moments.logvar,
        )

        self.log(
            "epoch_val_loss",
            loss["loss"],
            prog_bar=True,
            sync_dist=True,
            batch_size=data.size(0),
        )

        return loss["loss"]

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

        # Conditionally add parameter group for log_std based on self.loss_type
        if self.loss_type == "nll":
            optimizer_params.append(
                {
                    "params": [self.log_std],  # Separate parameter group for log_std
                    "weight_decay": 0.0,  # Exclude weight decay for log_std
                }
            )

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
                    eta_min=1e-7,
                ),
                "monitor": self.monitor,
                "interval": "epoch",
            }
        else:
            raise ValueError(f"{self.config_scheduler} lr_scheduler is not implemented")

        return [optimizer], [lr_scheduler]

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
        # Get the upper triangular part of each tensor in the batch
        upper_triangles = torch.triu(data_reconstructed)

        # Symmetrize each tensor in the batch individually
        symmetrized_arrays = upper_triangles + torch.transpose(upper_triangles, -1, -2)

        # Fill diagonal elements with zeros for each tensor individually
        diag_values = torch.diagonal(symmetrized_arrays, dim1=-2, dim2=-1)
        symmetrized_arrays = symmetrized_arrays - torch.diag_embed(diag_values)

        return symmetrized_arrays

    def sample(self, num_samples: int, sequence: torch.Tensor = None) -> torch.Tensor:
        """
        Sample from the latent space of the VAE and optionally
        condition on a sequence

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        sequence : torch.Tensor
            Sequence to generate distance maps for (sequence needs to be < 385), by default None

        Returns
        -------
        torch.Tensor
            Returns the generated distance maps
        """
        # If no sequence is given, generate zeroes (no conditioning)
        if sequence is None:
            sequence = torch.zeros(
                (num_samples, self.hparams.dimension, self.vocab_size),
                dtype=torch.float32,
            ).to(self.device)
        else:
            sequence = [sequence] * num_samples
        # Sample the latent encoding from N(0, I)
        latent_samples = torch.randn(
            num_samples, 1, int(self.compressed_size), int(self.compressed_size)
        ).to(self.device)

        # Decode the samples conditioned on sequence/labels
        with torch.no_grad():
            generated_samples = self.decode(latent_samples, sequence)

        if sequence is not None:
            sequence_length = len(sequence[0])
            generated_samples = generated_samples[
                :, :, :sequence_length, :sequence_length
            ]
        return generated_samples
