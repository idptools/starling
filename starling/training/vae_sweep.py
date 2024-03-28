import argparse

import pytorch_lightning as pl
import wandb
import yaml
from IPython import embed
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from starling.data.argument_parser import get_params
from starling.models.vae import VAE
from starling.training.myloader import MatrixDataModule


def train_vae():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the config file for wandb sweep",
    )

    args = parser.parse_args()

    config = get_params(config_file=args.config_file)
    wandb.init()
    wandb_logger = WandbLogger()

    config["model"]["kernel_size"] = wandb.config.kernel_size
    config["model"]["latent_dim"] = wandb.config.latent_dim
    config["model"]["num_layers"] = wandb.config.num_layers

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Set up data loaders
    dataset = MatrixDataModule(
        **config["data"], target_shape=config["model"]["dimension"]
    )

    dataset.setup(stage="fit")

    vae = VAE(**config["model"])
    wandb_logger.watch(vae)

    trainer = pl.Trainer(
        devices=config["device"]["cuda"],
        max_epochs=config["training"]["num_epochs"],
        callbacks=[lr_monitor],
        gradient_clip_val=1.0,
        precision="16-mixed",
        logger=wandb_logger,
    )
    trainer.fit(vae, dataset)

    wandb_logger.experiment.unwatch(vae)
    wandb.finish()


if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "epoch_val_loss"},
        "parameters": {
            "kernel_size": {"values": [3, 5, 7]},
            "latent_dim": {"values": [16, 32, 64, 128, 256, 512, 1024]},
            "num_layers": {"values": [5, 6, 7]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="small_model_sweep")
    wandb.agent(sweep_id, function=train_vae, count=40)
