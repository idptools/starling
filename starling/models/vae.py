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
        in_channels: int,
        latent_dim: int,
        deep: int,
        kernel_size: int,
        loss_type: str,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.loss_type = loss_type

        # these are used to monitor the training losses for the *EPOCH*
        self.total_train_step_losses = []
        self.recon_step_losses = []
        self.KLD_step_losses = []

        self.monitor = "epoch_val_loss"

        # Hard coded, should we actually start at 8 or 16
        starting_hidden_dim = 32

        self.hidden_dims = [starting_hidden_dim * 2**i for i in range(deep)]
        # This is hard coded in
        size_of_distance_map = 192
        # size_of_distance_map = 384
        # size_of_distance_map = 768

        shape = int(size_of_distance_map / 2 ** len(self.hidden_dims))
        modules = []

        for num, hidden_dim in enumerate(self.hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=2 if kernel_size == 5 else 1,
                    ),
                    # PrintLayer("encoder"),
                    nn.LayerNorm(
                        [
                            hidden_dim,
                            int(size_of_distance_map / (2 ** (num + 1))),
                            int(size_of_distance_map / (2 ** (num + 1))),
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
                        padding=2 if kernel_size == 5 else 1,
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
                padding=2 if kernel_size == 5 else 1,
                output_padding=1,
            ),
            # PrintLayer("Final layer"),
            nn.LayerNorm(
                [
                    reverse_hidden_dims[-1],
                    size_of_distance_map,
                    size_of_distance_map,
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
                padding=2 if kernel_size == 5 else 1,
            ),
            # PrintLayer("Final layer"),
            nn.ReLU(),
        )

        if self.loss_type == "elbo":
            # self.log_scale = nn.Parameter(
            #     torch.zeros(size_of_distance_map, size_of_distance_map)
            # )
            self.log_scale = nn.Parameter(
                torch.zeros((size_of_distance_map * (size_of_distance_map + 1) // 2))
            )

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

    def get_weights(self, ground_truth, scale="reciprocal"):
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

    def linear_KLD_beta(self, start, stop, n_epoch, n_cycle=5, ratio=1):
        # Compute the linear schedule for KLD weighting in the loss function
        L = torch.ones(n_epoch).to(self.device)
        period = n_epoch / n_cycle  # Compute period length with floating-point division
        step = (stop - start) / (period * ratio)  # Linear schedule step size

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and int(i + c * period) < n_epoch:
                L[int(i + c * period)] = v
                v += step
                i += 1

        return L

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        input_size = mean.shape[0]
        matrix_scale = torch.zeros(input_size, input_size).to(self.device)
        triu_indices = torch.triu_indices(input_size, input_size, offset=0)
        matrix_scale[triu_indices[0], triu_indices[1]] = scale[
            : input_size * (input_size + 1) // 2
        ]
        matrix_scale = matrix_scale + matrix_scale.t() - torch.diag(matrix_scale.diag())
        dist = torch.distributions.Normal(mean, matrix_scale)
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        weights = self.get_weights(x, scale="reciprocal")
        log_pxz *= weights
        return log_pxz.sum()

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def vae_loss(
        self,
        x_reconstructed,
        x,
        latent_encoding,
        mu,
        logvar,
        loss_type: str,
        scale="reciprocal",
        beta=1,
    ):
        # Find where the padding starts by counting the number of
        start_of_padding = torch.sum(x != 0, dim=(1, 2))[:, 0] + 1

        if loss_type == "mse" or loss_type == "weighted_mse":
            # Loss function for VAE, here I am removing the padded region
            # from both the ground truth and prediction

            BCE = 0

            for num, padding_start in enumerate(start_of_padding):
                x_reconstructed_no_padding = x_reconstructed[num][0][
                    :padding_start, :padding_start
                ]
                # Make the reconstructed map symmetric so that weights are freed to learn other
                # patterns
                x_reconstructed_no_padding = self.symmetrize(x_reconstructed_no_padding)

                # Get unpadded ground truth
                x_no_padding = x[num][0][:padding_start, :padding_start]

                # Mean squared error weighted by ground truth distance
                if loss_type == "weighted_mse":
                    weights = self.get_weights(x_no_padding, scale=scale)

                    mse_loss = F.mse_loss(
                        x_reconstructed_no_padding, x_no_padding, reduction="none"
                    )

                    BCE += (mse_loss * weights).sum()

                # Mean squared error not weighted by ground truth distance
                elif loss_type == "mse":
                    BCE += F.mse_loss(
                        x_reconstructed_no_padding, x_no_padding, reduction="mean"
                    )
                else:
                    raise ValueError(f"loss type of name '{loss_type}' does not exist")

            # Taking the mean of the loss (could also be sum)
            BCE /= num + 1

            # See Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            KLD = torch.logsumexp(KLD, dim=0) / mu.size(0)  # Mean over batch

            loss = BCE + beta * KLD

            return {"loss": loss, "BCE": BCE, "KLD": KLD}

        elif loss_type == "elbo":
            total_recon_loss = 0
            for num, padding_start in enumerate(start_of_padding):
                x_reconstructed_no_padding = x_reconstructed[num][0][
                    :padding_start, :padding_start
                ]
                # Make the reconstructed map symmetric so that weights are
                # freed to learn other patterns
                x_reconstructed_no_padding = self.symmetrize(x_reconstructed_no_padding)

                # Get unpadded ground truth
                x_no_padding = x[num][0][:padding_start, :padding_start]

                # Get the reconstruction loss
                recon_loss = self.gaussian_likelihood(
                    x_reconstructed_no_padding, self.log_scale, x_no_padding
                )
                total_recon_loss += recon_loss

            # Take the mean of all the losses in batch
            total_recon_loss /= num + 1

            # std = torch.exp(logvar / 2)
            # KLD = self.kl_divergence(latent_encoding, mu, std)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            KLD = torch.logsumexp(KLD, dim=0) / mu.size(0)  # Mean over batch
            elbo = beta * KLD - total_recon_loss
            # elbo = KLD.mean() - total_recon_loss

            return {"loss": elbo, "BCE": total_recon_loss, "KLD": KLD}

    def forward(self, x):
        mu, logvar = self.encode(x)

        if self.loss_type in ["mse", "weighted_mse"]:
            latent_encoding = self.reparameterize(mu, logvar)

        elif self.loss_type == "elbo":
            # std = torch.exp(logvar / 2)
            # q = torch.distributions.Normal(mu, std)
            # latent_encoding = q.rsample()
            latent_encoding = self.reparameterize(mu, logvar)

        else:
            raise ValueError(f"loss type of name {self.loss_type} not implemented")

        x_reconstructed = self.decode(latent_encoding)

        return x_reconstructed, mu, logvar, latent_encoding

    def training_step(self, batch, batch_idx):
        x = batch["input"]

        # Calculate the schedule for linear scaling of beta for KLD loss
        # Only calculate if we just started to train
        # if self.trainer.global_step == 0:
        #     self.KLD_betas = self.linear_KLD_beta(
        #         start=0, stop=5, n_epoch=self.trainer.max_epochs, ratio=1
        #     )

        x_reconstructed, mu, logvar, latent_encoding = self.forward(x)

        loss = self.vae_loss(
            x_reconstructed=x_reconstructed,
            x=x,
            latent_encoding=latent_encoding,
            mu=mu,
            logvar=logvar,
            loss_type=self.loss_type,
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
        x = batch["input"]

        x_reconstructed, mu, logvar, latent_encoding = self.forward(x)

        # What loss do we want to use in validation step
        # A regular sum I think makes sense, instead of an
        # ever changing beta
        loss = self.vae_loss(
            x_reconstructed=x_reconstructed,
            x=x,
            latent_encoding=latent_encoding,
            mu=mu,
            logvar=logvar,
            loss_type=self.loss_type,
            beta=1,
        )

        self.log("epoch_val_loss", loss["loss"], prog_bar=True, sync_dist=True)

        return loss["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.05, momentum=0.99, nesterov=True
        )

        lr_scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-4),
            "monitor": self.monitor,
            "interval": "epoch",
        }

        # lr_scheduler = {
        #     "scheduler": OneCycleLR(
        #         optimizer,
        #         max_lr=0.01,
        #         total_steps=self.trainer.estimated_stepping_batches,
        #     ),
        #     "monitor": self.monitor,
        #     "interval": "step",
        # }

        return [optimizer], [lr_scheduler]
