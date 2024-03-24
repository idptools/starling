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

from starling.models.vae import VAE
from starling.training.myloader import MatrixDataModule


def train_vae():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to the datafile in tsv format: <name> <data_path>",
    )

    parser.add_argument(
        "--validation_data",
        type=str,
        default=None,
        help="Path to the datafile in tsv format: <name> <data_path>",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=".",
        help="Path to the directory to store the network in",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs to do",
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="The size of the latent 1D tensor",
    )

    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="The size of the kernel in Conv2d",
    )

    parser.add_argument(
        "--deep",
        type=int,
        default=5,
        help="How many convolutional layers (how deep should the network be)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="weighted_mse",
        help="""What loss to calculate for the reconstruction loss, current losses include 
        mse and weighted_mse""",
    )

    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="""Whether to normalize the distance map. Options:[length, afrc, bond_length].
        Length is normalizing by np.sqrt(length), afrc by AFRC distance map, and bond_length 
        by the accumulated count of residues for each element within a distance map*bond_length""",
    )

    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Enable interpolation of distance maps through torch resizing using bicubic method",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID for training",
    )

    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        monitor="epoch_val_loss",  # Monitor validation loss for saving the best model
        dirpath=f"{args.output_path}/",  # Directory to save checkpoints
        filename="model-kernel-{epoch:02d}-{epoch_val_loss:.2f}",  # File name format for saved models
        save_top_k=3,  # Save the top 3 models based on monitored metric
        mode="min",  # Minimize the monitored metric (val_loss)
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Set up data loaders
    dataset = MatrixDataModule(
        args.train_data,
        args.validation_data,
        args.validation_data,
        args.batch_size,
        args=args,
    )

    # Initialize VAE model
    input_dim = 1  # Assuming distance map
    latent_dim = args.latent_dim  # Adjust as needed

    # Train the VAE
    num_epochs = args.num_epochs

    vae = VAE(
        in_channels=input_dim,
        latent_dim=latent_dim,
        deep=args.deep,
        kernel_size=args.kernel_size,
        loss_type=args.loss_type,
    )

    wandb_logger = WandbLogger(project="full_data_vae")
    wandb_logger.watch(vae)

    trainer = pl.Trainer(
        devices=args.gpu_id,
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        precision="16-mixed",
        logger=wandb_logger,
    )
    trainer.fit(vae, dataset)

    wandb_logger.experiment.unwatch(vae)
    wandb.finish()


if __name__ == "__main__":
    train_vae()
