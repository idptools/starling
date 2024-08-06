import argparse
import os

import pytorch_lightning as pl
import wandb
import yaml
from IPython import embed
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from starling.data.argument_parser import get_vae_params
from starling.data.VAE_loader import MatrixDataModule
from starling.models.cvae import cVAE


def train_vae():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration file to use",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers to use for loading in the data",
    )

    args = parser.parse_args()

    # Reads in default and user defined configuration arguments
    config = get_vae_params(config_file=args.config_file)

    # Set up model checkpoint saving
    checkpoint_callback = ModelCheckpoint(
        monitor="epoch_val_loss",  # Monitor validation loss for saving the best model
        dirpath=f"{config['training']['output_path']}/",  # Directory to save checkpoints
        filename="model-kernel-{epoch:02d}-{epoch_val_loss:.2f}",  # File name format for saved models
        save_top_k=3,  # Save the top 3 models based on monitored metric
        mode="min",  # Minimize the monitored metric (val_loss)
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Set up data loaders
    dataset = MatrixDataModule(
        **config["data"],
        target_shape=config["model"]["dimension"],
        num_workers=args.num_workers,
    )

    dataset.setup(stage="fit")

    vae = cVAE(**config["model"])

    # Make the directories to save the model and logs
    os.makedirs(config["training"]["output_path"], exist_ok=True)

    with open(f"{config['training']['output_path']}/config.yaml", "w") as f:
        yaml.dump(config, f)

    with open(f"{config['training']['output_path']}/model_architecture.txt", "w") as f:
        f.write(str(vae))
    ##############################

    # Set up logging on weights and biases
    wandb_logger = WandbLogger(project=config["training"]["project_name"])
    wandb_logger.watch(vae)

    # Set up PyTorch Lightning Trainer
    trainer = pl.Trainer(
        devices=config["device"]["cuda"],
        max_epochs=config["training"]["num_epochs"],
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        precision="16-mixed",
        logger=wandb_logger,
    )

    # Start training
    trainer.fit(vae, dataset)

    # Detach the logging on wandb
    wandb_logger.experiment.unwatch(vae)
    wandb.finish()


if __name__ == "__main__":
    train_vae()
