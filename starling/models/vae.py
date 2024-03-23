import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class PrintLayer(nn.Module):
    def __init__(self, layer_name):
        super(PrintLayer, self).__init__()
        self.layer_name = layer_name

    def forward(self, x):
        print(f"Intermediate output of {self.layer_name}: {x.shape}")
        return x


torch.set_float32_matmul_precision("high")


class VAE(pl.LightningModule):
    def __init__(self, in_channels: int, latent_dim: int, deep: int, kernel_size: int):
        super().__init__()

        self.save_hyperparameters()

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
                    nn.ReLU(),
                )
            )
            in_channels = hidden_dim

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
            # nn.BatchNorm2d(1),
            nn.Conv2d(
                reverse_hidden_dims[-1],
                out_channels=1,
                kernel_size=kernel_size,
                padding=2 if kernel_size == 5 else 1,
            ),
            # PrintLayer("Final layer"),
            # nn.Tanh(),
            nn.ReLU(),
        )

    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        self.final_encoded_shape = z.shape
        z = torch.flatten(z, start_dim=1)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)

        return [mu, log_var]

    def decode(self, z: torch.Tensor):
        x = self.first_decode_layer(z)
        x = x.view(self.final_encoded_shape)
        x = self.decoder(x)
        x = self.final_decode_layer(x)

        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def vae_loss(self, x_reconstructed, x, mu, logvar, loss_type: str):
        if loss_type == "mse" or loss_type == "weighted_mse":
            # Loss function for VAE, here I am removing the padded region
            # from both the ground truth and prediction

            # Find where the padding starts by counting the number of
            start_of_padding = torch.sum(x != 0, dim=(1, 2))[:, 0] + 1

            BCE = 0

            for num, padding_start in enumerate(start_of_padding):
                x_reconstructed_no_padding = x_reconstructed[num][0][
                    :padding_start, :padding_start
                ]
                x_no_padding = x[num][0][:padding_start, :padding_start]

                # Mean squared error weighted by ground truth distance
                if loss_type == "weighted_mse":
                    weights = torch.reciprocal(x_no_padding)
                    weights[weights == float("inf")] = 0

                    mse_loss = F.mse_loss(
                        x_reconstructed_no_padding, x_no_padding, reduction="none"
                    )

                    BCE += ((mse_loss * weights) / (weights.sum() / 2)).sum()

                # Mean squared error not weighted by ground truth distance
                else:
                    BCE += F.mse_loss(
                        x_reconstructed_no_padding, x_no_padding, reduction="mean"
                    )

            # Taking the mean of the loss (could also be sum)
            BCE /= num + 1

            #!think about implementing weighted mse_loss where short range distance
            #! maps are more heavily weighted (i.e. more important than long range
            #! interactions)

            # See Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            KLD = torch.logsumexp(KLD, dim=0) / mu.size(0)  # Mean over batch

            # beta = 0.01
            beta = 0.1
            # beta = 1
            # KLD *= 0
            loss = BCE + beta * KLD

            return {"loss": loss, "BCE": BCE, "KLD": KLD}

        elif loss_type == "elbo":
            pass

    def forward(self, x):
        mu, logvar = self.encode(x)

        latent_encoding = self.reparameterize(mu, logvar)

        x_reconstructed = self.decode(latent_encoding)

        return x_reconstructed, mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch["input"]

        x_reconstructed, mu, logvar = self.forward(x)

        loss = self.vae_loss(x_reconstructed, x, mu, logvar, loss_type="mse")

        self.total_train_step_losses.append(loss["loss"])
        self.recon_step_losses.append(loss["BCE"])
        self.KLD_step_losses.append(loss["KLD"])

        self.log("train_loss", loss["loss"], prog_bar=True)
        self.log("recon_loss", loss["BCE"], prog_bar=True)

        return loss["loss"]

    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.total_train_step_losses).mean()
        self.log("epoch_train_loss", epoch_mean, prog_bar=True)

        recon_mean = torch.stack(self.recon_step_losses).mean()
        self.log("epoch_recon_loss", recon_mean, prog_bar=True)

        KLD_mean = torch.stack(self.KLD_step_losses).mean()
        self.log("epoch_KLD_loss", KLD_mean, prog_bar=True)

        # free up the memory
        self.total_train_step_losses.clear()
        self.recon_step_losses.clear()
        self.KLD_step_losses.clear()

    def validation_step(self, batch, batch_idx):
        x = batch["input"]

        x_reconstructed, mu, logvar = self.forward(x)

        loss = self.vae_loss(x_reconstructed, x, mu, logvar, loss_type="mse")

        self.log("epoch_val_loss", loss["loss"], prog_bar=True)

        return loss["loss"]

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=1e-4)
        # return torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)

        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.05, momentum=0.99, nesterov=True
        )

        lr_scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-4),
            "monitor": self.monitor,
            "interval": "epoch",
        }

        return [optimizer], [lr_scheduler]
