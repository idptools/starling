import math
from collections import namedtuple
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
)

# Define a named tuple at the module level
BatchOutput = namedtuple(
    "BatchOutput", ["latent_distance_maps", "sequences", "masks", "ionic_strength"]
)


class STARLING(pl.LightningModule):
    def __init__(
        self,
        ddpm_model: nn.Module,
        vae_model: nn.Module,
        learning_rate: float = 1e-4,
        scheduler: str = "LinearWarmupCosineAnnealingLR",
        warmup_fraction: float = 0.1,
        vae_learning_rate: float = 1e-5,  # Much smaller learning rate for VAE
        freeze_vae_first_epoch: bool = True,  # Option to freeze VAE for first epoch
        ddpm_loss_weight: float = 5.0,  # Weight for DDPM loss
    ):
        super().__init__()

        self.ddpm_model = ddpm_model
        self.vae_model = vae_model

        self.ddpm_loss_weight = ddpm_loss_weight

        # Add hyperparameters needed for optimizer configuration
        self.ddpm_lr = learning_rate
        self.vae_lr = vae_learning_rate
        self.config_scheduler = scheduler
        self.warmup_fraction = warmup_fraction
        self.monitor = "epoch_val_loss"
        self.num_timesteps = ddpm_model.num_timesteps
        self.freeze_vae_first_epoch = freeze_vae_first_epoch
        self.current_epoch_tracked = 0

        # Freeze VAE parameters initially if requested
        if self.freeze_vae_first_epoch:
            self._freeze_vae()

    def _freeze_vae(self):
        """Freeze all parameters in the VAE model"""
        for param in self.vae_model.parameters():
            param.requires_grad = False

    def _unfreeze_vae(self):
        """Unfreeze all parameters in the VAE model"""
        for param in self.vae_model.parameters():
            param.requires_grad = True

    def on_train_epoch_start(self):
        """Check if we should unfreeze the VAE after the first epoch"""
        if self.freeze_vae_first_epoch and self.current_epoch > 0:
            self._unfreeze_vae()
            print("Unfreezing VAE parameters after first epoch")
            # Only do this once
            self.freeze_vae_first_epoch = False

    def on_train_epoch_end(self):
        """Update the epoch counter"""
        self.current_epoch_tracked += 1

    def vae_forward(self, x: torch.Tensor) -> torch.Tensor:
        data_reconstructed, moments = self.vae_model(x)

        return data_reconstructed, moments

    def ddpm_forward(
        self,
        latent_maps: torch.Tensor,
        sequences: torch.Tensor,
        masks: torch.Tensor,
        ionic_strength: torch.Tensor,
    ) -> torch.Tensor:
        # Create a batch using the namedtuple
        batch = BatchOutput(
            latent_distance_maps=latent_maps,
            sequences=sequences,
            masks=masks,
            ionic_strength=ionic_strength,
        )

        b, c, h, w, device = *latent_maps.shape, latent_maps.device

        # Generate random timestamps to noise the tensor and learn the denoising process
        timestamps = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Pass the batch to the DDPM model
        return self.ddpm_model.p_loss(batch, timestamps)

    def forward(
        self,
        x: torch.Tensor,
        sequences,
        masks: torch.Tensor = None,
        ionic_strength: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward pass through VAE
        data_reconstructed, moments = self.vae_forward(x)

        vae_loss = self.vae_model.vae_loss(
            data_reconstructed=data_reconstructed,
            data=x,
            mu=moments.mean,
            logvar=moments.logvar,
        )

        # Forward pass through DDPM
        ddpm_loss = self.ddpm_forward(
            latent_maps=moments.mode(),
            sequences=sequences,
            masks=masks,
            ionic_strength=ionic_strength,
        )

        return vae_loss, ddpm_loss

    def training_step(self, batch):
        distance_maps, sequences, masks, ionic_strengths = (
            batch.distance_maps,
            batch.sequences,
            batch.masks,
            batch.ionic_strength,
        )

        vae_loss, ddpm_loss = self(
            x=distance_maps,
            sequences=sequences,
            masks=masks,
            ionic_strength=ionic_strengths,
        )

        # Only include DDPM loss during the first epoch when VAE is frozen
        if self.freeze_vae_first_epoch and self.current_epoch == 0:
            loss = self.ddpm_loss_weight * ddpm_loss
        else:
            loss = vae_loss["loss"] + self.ddpm_loss_weight * ddpm_loss

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=distance_maps.size(0),
        )
        self.log(
            "ddpm_loss", ddpm_loss, prog_bar=True, batch_size=distance_maps.size(0)
        )
        self.log(
            "recon_loss",
            vae_loss["recon"],
            prog_bar=True,
            batch_size=distance_maps.size(0),
        )
        self.log(
            "KLD_loss",
            vae_loss["KLD"],
            prog_bar=False,
            batch_size=distance_maps.size(0),
        )

        return loss

    def validation_step(self, batch):
        distance_maps, sequences, masks, ionic_strengths = (
            batch.distance_maps,
            batch.sequences,
            batch.masks,
            batch.ionic_strength,
        )

        vae_loss, ddpm_loss = self(
            x=distance_maps,
            sequences=sequences,
            masks=masks,
            ionic_strength=ionic_strengths,
        )

        loss = vae_loss["loss"] + self.ddpm_loss_weight * ddpm_loss

        self.log(
            "epoch_val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=distance_maps.size(0),
        )

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer with different learning rates for DDPM and VAE.
        """
        # Create parameter groups with different learning rates
        param_groups = [
            {"params": self.ddpm_model.parameters(), "lr": self.ddpm_lr},
            {"params": self.vae_model.parameters(), "lr": self.vae_lr},
        ]

        # Single optimizer with different learning rates per group
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
            amsgrad=False,
        )

        # Rest of the scheduler configuration remains the same
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
            warmup_steps = int(steps_per_epoch * num_epochs * self.warmup_fraction)

            # Define scheduler functions for different parameter groups
            def ddpm_scheduler(current_step):
                # Regular scheduling for DDPM
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

            def vae_scheduler(current_step):
                # Fixed learning rate for VAE - always return 1.0
                # (multiplied by the base learning rate)
                return 1.0

            # List of scheduler functions for each parameter group
            # First function for DDPM, second for VAE
            lr_scheduler = {
                "scheduler": LambdaLR(
                    optimizer, lr_lambda=[ddpm_scheduler, vae_scheduler]
                ),
                "monitor": self.monitor,
                "interval": "step",
            }

        else:
            raise ValueError(f"{self.config_scheduler} lr_scheduler is not implemented")

        return [optimizer], [lr_scheduler]
