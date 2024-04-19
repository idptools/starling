from typing import List, Tuple

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


class VAE(pl.LightningModule):
    def __init__(
        self,
        model,
        in_channels,
        latent_dim,
        kernel_size,
        dimension,
        loss_type,
        weights_type,
        KLD_weight,
        lr_scheduler,
        set_lr,
        encoder_block="original",
        decoder_block="original",
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

        # Loss params
        self.loss_type = loss_type
        self.weights_type = weights_type

        # Learning rate params
        self.config_scheduler = lr_scheduler
        self.set_lr = set_lr

        # KLD loss params
        self.KLD_weight = KLD_weight

        # these are used to monitor the training losses for the *EPOCH*
        self.total_train_step_losses = []
        self.recon_step_losses = []
        self.KLD_step_losses = []

        self.monitor = "epoch_val_loss"

        # Encoder
        self.encoder = resnets[model]["encoder"][encoder_block](
            in_channels=in_channels,
            kernel_size=kernel_size,
            dimension=dimension,
            base=base,
        )

        # This is usually 4 in ResNets
        num_stages = 4
        expansion = self.encoder.block_type.expansion
        exponent = num_stages - 1 if expansion == 1 else num_stages + 1
        linear_layer_params = int(base * 2**exponent)
        self.shape_from_final_encoding_layer = linear_layer_params, 1, 1

        # Latent space
        self.fc_mu = nn.Linear(linear_layer_params * 1 * 1, latent_dim)
        self.fc_var = nn.Linear(linear_layer_params * 1 * 1, latent_dim)
        self.first_decode_layer = nn.Linear(latent_dim, linear_layer_params * 1 * 1)

        # Decoder
        self.decoder = resnets[model]["decoder"][decoder_block](
            out_channels=in_channels,
            kernel_size=kernel_size,
            dimension=dimension,
            base=base,
        )

        # Params to learn for reconstruction loss
        if self.loss_type == "elbo":
            self.log_std = nn.Parameter(torch.zeros((dimension * (dimension + 1) // 2)))

    def encode(self, data: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Takes the data and encodes it into the latent space,
        by returning the mean and log variance

        Parameters
        ----------
        data : torch.Tensor
            Data in the shape of (batch, channel, height, width)

        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor]]
            Return the mean and log variance of the latent space
        """
        data = self.encoder(data)
        data = torch.flatten(data, start_dim=1)
        mu = self.fc_mu(data)
        log_var = self.fc_var(data)

        return [mu, log_var]

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent space back into the original data

        Parameters
        ----------
        latents : torch.Tensor
            latents in the shape of (batch, channel, height, width)

        Returns
        -------
        torch.Tensor
            Returns the reconstructed data
        """
        # Linear layer first to get the shape of the final encoding layer
        data = self.first_decode_layer(latents)
        data = data.view(-1, *self.shape_from_final_encoding_layer)
        data = self.decoder(data)
        return data

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparametarization trick that allows for the flow of gradients through the
        non-random process. Check out the paper for more details:
        https://arxiv.org/abs/1312.6114

        Parameters
        ----------
        mu : torch.Tensor
            A tensor containing means of the latent space
        logvar : torch.Tensor
            A tensor containg the log variance of the latent space

        Returns
        -------
        torch.Tensor
            Returns the latent encoding
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_weights(self, ground_truth: torch.Tensor, scale: str) -> torch.Tensor:
        """
        A function that calculates weights for the reconstruction loss based on the
        distance between the residues in the ground truth distance map. The weights
        are calculated based on the scale parameter.

        Parameters
        ----------
        ground_truth : torch.Tensor
            Input data or the ground truth distance map
        scale : str
            A string that determines how the weights will be calculated. The options
            are "linear", "reciprocal", and "equal"

        Returns
        -------
        torch.Tensor
            Returns the weights for the reconstruction loss

        Raises
        ------
        ValueError
            If the scale parameter is not one of the three options
        """
        #! not sure linear is correct rn
        if scale == "linear":
            max_distance = ground_truth.max()
            min_distance = ground_truth.min()
            weights = 1 - (ground_truth - min_distance) / (max_distance - min_distance)
            weights = weights / weights.sum()
            return weights
        # Reciprocal of the distance between the two residues is taken as the weight
        elif scale == "reciprocal":
            # Handling division by zero
            nonzero_indices = ground_truth != 0
            weights = torch.zeros_like(ground_truth)
            weights[nonzero_indices] = torch.reciprocal(ground_truth[nonzero_indices])
            weights = weights / weights.sum()
            return weights
        # We assign equal weight to each distance here
        elif scale == "equal":
            weights = torch.ones_like(ground_truth)
            # Set diagonal elements to zero
            weights = weights - torch.diag(torch.diag(weights))
            weights = weights / weights.sum()
            return weights
        else:
            raise ValueError(f"Variable name '{scale}' for get_weights does not exist")

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

    def gaussian_likelihood(
        self, data_hat: torch.Tensor, log_std: torch.Tensor, data: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the likelihood of input data given latent space (p(x|z))
        under Gaussian assumption. The reconstructured data is treated as the mean
        of the Gaussian distributions and the log_std is a tensor of learned log standard
        deviations.

        Parameters
        ----------
        data_hat : torch.Tensor
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
        std = torch.exp(log_std)
        mean = data_hat
        input_size = mean.shape[0]

        # Construct the covariance matrix
        matrix_std = torch.zeros_like(mean).to(torch.float32)
        triu_indices = torch.triu_indices(input_size, input_size, offset=0)
        matrix_std[triu_indices[0], triu_indices[1]] = std[
            : input_size * (input_size + 1) // 2
        ]
        matrix_std = matrix_std + matrix_std.t() - torch.diag(matrix_std.diag())

        # Create the normal distributions
        dist = torch.distributions.Normal(mean, matrix_std)

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
        under a Gaussian assumption.

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
            KLD_weight = self.KLD_weight

        # Find where the padding starts by counting the number of
        start_of_padding = torch.sum(data != 0, dim=(1, 2))[:, 0] + 1

        recon = 0

        # Since we are removing the padding, we need to calculate the loss
        # per distance map, since they are padded differently (different input sizes)
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

            # Get the weights for the loss
            weights = self.get_weights(data_no_padding, scale=self.weights_type)

            # Mean squared error weighted by ground truth distance
            if self.loss_type == "mse":
                mse_loss = F.mse_loss(
                    data_reconstructed_no_padding, data_no_padding, reduction="none"
                )

                recon += (mse_loss * weights).sum()

            elif self.loss_type == "elbo":
                # Get the reconstruction loss
                gaussian_likelihood = self.gaussian_likelihood(
                    data_hat=data_reconstructed_no_padding,
                    log_std=self.log_std,
                    data=data_no_padding,
                )
                recon += -(gaussian_likelihood * weights).sum()
            else:
                raise ValueError(f"loss type of name '{self.loss_type}' does not exist")

        # Taking the mean of the loss (could also be sum)
        recon /= num + 1

        # See Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        KLD = torch.logsumexp(KLD, dim=0) / mu.size(0)  # Mean over batch

        loss = recon + KLD_weight * KLD

        return {"loss": loss, "recon": recon, "KLD": KLD}

    def forward(
        self, data: torch.Tensor
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
        mu, logvar = self.encode(data)
        latent_encoding = self.reparameterize(mu, logvar)

        data_reconstructed = self.decode(latent_encoding)
        return data_reconstructed, mu, logvar

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

        data_reconstructed, mu, logvar = self.forward(data=data)

        loss = self.vae_loss(
            data_reconstructed=data_reconstructed,
            data=data,
            mu=mu,
            logvar=logvar,
        )

        if batch_idx % 100 == 0:
            self.total_train_step_losses.append(loss["loss"])
            self.recon_step_losses.append(loss["recon"])
            self.KLD_step_losses.append(loss["KLD"])

        self.log("train_loss", loss["loss"], prog_bar=True)
        self.log("recon_loss", loss["recon"], prog_bar=True)

        return loss["loss"]

    def on_train_epoch_end(self) -> None:
        """
        At the end of the epoch, calculate the mean of the training losses.
        Clear the lists that have been filled with losses during the epoch
        for memory management.
        """
        epoch_mean = torch.stack(self.total_train_step_losses).mean()
        self.log("epoch_train_loss", epoch_mean, prog_bar=True, sync_dist=True)

        recon_mean = torch.stack(self.recon_step_losses).mean()
        self.log("epoch_recon_loss", recon_mean, prog_bar=True, sync_dist=True)

        KLD_mean = torch.stack(self.KLD_step_losses).mean()
        self.log("epoch_KLD_loss", KLD_mean, prog_bar=True, sync_dist=True)

        # free up the memory
        self.total_train_step_losses.clear()
        self.recon_step_losses.clear()
        self.KLD_step_losses.clear()

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

        data_reconstructed, mu, logvar = self.forward(data)

        loss = self.vae_loss(
            data_reconstructed=data_reconstructed,
            data=data,
            mu=mu,
            logvar=logvar,
        )

        self.log("epoch_val_loss", loss["loss"], prog_bar=True, sync_dist=True)

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
        if self.loss_type == "elbo":
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
                    eta_min=1e-4,
                ),
                "monitor": self.monitor,
                "interval": "epoch",
            }
        else:
            raise ValueError(f"{self.config_scheduler} lr_scheduler is not implemented")

        return [optimizer], [lr_scheduler]
