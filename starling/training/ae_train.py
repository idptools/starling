import argparse

import pytorch_lightning as pl
import wandb
from IPython import embed
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from starling.data.argument_parser import get_params
from starling.data.myloader import MatrixDataModule
from starling.models.ae import AE


def train_ae():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration file to use",
    )

    args = parser.parse_args()

    config = get_params(config_file=args.config_file)

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
        **config["data"], target_shape=config["model"]["ae"]["dimension"]
    )

    dataset.setup(stage="fit")

    ae = AE(**config["model"]["ae"])

    with open(f"{config['training']['output_path']}/model_architecture.txt", "w") as f:
        f.write(str(ae))

    wandb_logger = WandbLogger(project=config["training"]["project_name"])
    wandb_logger.watch(ae)

    trainer = pl.Trainer(
        devices=config["device"]["cuda"],
        max_epochs=config["training"]["num_epochs"],
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        precision="16-mixed",
        logger=wandb_logger,
    )
    trainer.fit(ae, dataset)

    wandb_logger.experiment.unwatch(ae)
    wandb.finish()


if __name__ == "__main__":
    train_ae()
