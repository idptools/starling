import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR


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
        in_channels,
        latent_dim,
        num_layers,
        kernel_size,
        starting_hidden_dim,
        dimension,
        loss_type,
        weights_type,
        KLD_weight,
        lr_scheduler,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Loss params
        self.loss_type = loss_type
        self.weights_type = weights_type
        self.config_scheduler = lr_scheduler

        # KLD loss params
        self.KLD_weight = KLD_weight

        # these are used to monitor the training losses for the *EPOCH*
        self.total_train_step_losses = []
        self.recon_step_losses = []
        self.KLD_step_losses = []

        self.monitor = "epoch_val_loss"

        self.hidden_dims = [starting_hidden_dim * 2**i for i in range(num_layers)]

        shape = int(dimension / 2 ** len(self.hidden_dims))
        modules = []

        for num, hidden_dim in enumerate(self.hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=2
                        if kernel_size == 5
                        else (3 if kernel_size == 7 else 1),
                    ),
                    # PrintLayer("encoder"),
                    nn.LayerNorm(
                        [
                            hidden_dim,
                            int(dimension / (2 ** (num + 1))),
                            int(dimension / (2 ** (num + 1))),
                        ]
                    ),
                    # nn.GroupNorm(int(hidden_dim / 4), int(hidden_dim)),
                    # nn.InstanceNorm2d(hidden_dim),
                    nn.ReLU(),
                )
            )
            in_channels = hidden_dim

        # Get the shape of the output of the final layer
        self.shape_from_final_encoding_layer = self.hidden_dims[-1], shape, shape

        # Encoder
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * shape * shape, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * shape * shape, latent_dim)

        # Building a decoder
        modules = []

        self.first_decode_layer = nn.Linear(
            latent_dim, self.hidden_dims[-1] * shape * shape
        )

        reverse_hidden_dims = list(self.hidden_dims[::-1])

        for num in range(len(reverse_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        reverse_hidden_dims[num],
                        reverse_hidden_dims[num + 1],
                        kernel_size=kernel_size,
                        stride=2,
                        padding=2
                        if kernel_size == 5
                        else (3 if kernel_size == 7 else 1),
                        output_padding=1,
                    ),
                    # PrintLayer("Decoder"),
                    nn.LayerNorm(
                        [
                            reverse_hidden_dims[num + 1],
                            int(shape * (2 ** (num + 1))),
                            int(shape * (2 ** (num + 1))),
                        ]
                    ),
                    # nn.InstanceNorm2d(reverse_hidden_dims[num + 1]),
                    # nn.GroupNorm(
                    #     int(reverse_hidden_dims[num + 1] / 4),
                    #     int(reverse_hidden_dims[num + 1]),
                    # ),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_decode_layer = nn.Sequential(
            nn.ConvTranspose2d(
                reverse_hidden_dims[-1],
                out_channels=reverse_hidden_dims[-1],
                kernel_size=kernel_size,
                stride=2,
                padding=2 if kernel_size == 5 else (3 if kernel_size == 7 else 1),
                output_padding=1,
            ),
            # PrintLayer("Final layer"),
            nn.LayerNorm(
                [
                    reverse_hidden_dims[-1],
                    dimension,
                    dimension,
                ]
            ),
            # nn.InstanceNorm2d(reverse_hidden_dims[-1]),
            # nn.GroupNorm(
            #     int(reverse_hidden_dims[num + 1] / 4), int(reverse_hidden_dims[num + 1])
            # ),
            nn.Conv2d(
                reverse_hidden_dims[-1],
                out_channels=1,
                kernel_size=kernel_size,
                padding=2 if kernel_size == 5 else (3 if kernel_size == 7 else 1),
            ),
            # PrintLayer("Final layer"),
            nn.ReLU(),
        )

        if self.loss_type == "elbo":
            # self.log_std = nn.Parameter(
            #     torch.zeros(dimension, dimension)
            # )
            self.log_std = nn.Parameter(torch.zeros((dimension * (dimension + 1) // 2)))

    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        z = torch.flatten(z, start_dim=1)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)

        return [mu, log_var]

    def decode(self, z: torch.Tensor):
        x = self.first_decode_layer(z)
        x = x.view(-1, *self.shape_from_final_encoding_layer)
        x = self.decoder(x)
        x = self.final_decode_layer(x)

        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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

    def vae_loss(
        self,
        data_reconstructed,
        data,
        mu,
        logvar,
    ):
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

            # See Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            KLD = torch.logsumexp(KLD, dim=0) / mu.size(0)  # Mean over batch

            loss = BCE + self.KLD_weight * KLD

            return {"loss": loss, "BCE": BCE, "KLD": KLD}

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

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            KLD = torch.logsumexp(KLD, dim=0) / mu.size(0)  # Mean over batch
            elbo = self.KLD_weight * KLD - total_recon_loss

            return {"loss": elbo, "BCE": total_recon_loss, "KLD": KLD}

    def forward(self, data):
        mu, logvar = self.encode(data)
        latent_encoding = self.reparameterize(mu, logvar)

        data_reconstructed = self.decode(latent_encoding)
        return data_reconstructed, mu, logvar, latent_encoding

    def training_step(self, batch, batch_idx):
        data = batch["input"]

        data_reconstructed, mu, logvar, latent_encoding = self.forward(data=data)

        loss = self.vae_loss(
            data_reconstructed=data_reconstructed,
            data=data,
            mu=mu,
            logvar=logvar,
        )

        self.total_train_step_losses.append(loss["loss"])
        self.recon_step_losses.append(loss["BCE"])
        self.KLD_step_losses.append(loss["KLD"])

        self.log("train_loss", loss["loss"], prog_bar=True)
        self.log("recon_loss", loss["BCE"], prog_bar=True)

        return loss["loss"]

    def on_train_epoch_end(self):
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

    def validation_step(self, batch, batch_idx):
        data = batch["input"]

        data_reconstructed, mu, logvar, latent_encoding = self.forward(data)

        loss = self.vae_loss(
            data_reconstructed=data_reconstructed,
            data=data,
            mu=mu,
            logvar=logvar,
        )

        self.log("epoch_val_loss", loss["loss"], prog_bar=True, sync_dist=True)

        return loss["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.05, momentum=0.99, nesterov=True
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
        else:
            raise ValueError(f"{self.config_scheduler} lr_scheduler is not implemented")

        return [optimizer], [lr_scheduler]
