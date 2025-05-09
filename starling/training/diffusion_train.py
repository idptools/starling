import argparse
import glob
import os

import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from starling.data.argument_parser import get_params
from starling.data.ddpm_loader import MatrixDataModule
from starling.models.diffusion import DiffusionModel
from starling.models.unet import UNetConditional
from starling.models.vae import VAE


@rank_zero_only
def wandb_init(project: str = "starling"):
    wandb.init(project=project)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration file to use",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="Number of nodes to use for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers to use for loading in the data",
    )
    return parser.parse_args()


def setup_directories(output_path):
    """Create necessary directories and save the configuration file."""
    os.makedirs(output_path, exist_ok=True)


def save_config(config, output_path):
    """Save the configuration to a YAML file."""
    with open(f"{output_path}/config.yaml", "w") as f:
        yaml.dump(config, f)


def setup_checkpoints(output_path):
    """Set up model checkpoint callbacks."""
    checkpoint_callback = ModelCheckpoint(
        monitor="epoch_val_loss",
        dirpath=output_path,
        filename="model-kernel-{epoch:02d}-{epoch_val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    save_last_checkpoint = ModelCheckpoint(
        dirpath=output_path,
        filename="last",
    )
    return checkpoint_callback, save_last_checkpoint


def get_checkpoint_path(output_path):
    """Determine the checkpoint path to resume training if available."""
    checkpoint_pattern = os.path.join(output_path, "last.ckpt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    return "last" if checkpoint_files else None


def setup_data_module(config, num_workers):
    """Set up the data module."""
    dataset = MatrixDataModule(
        **config["data"],
        labels=config["diffusion"]["labels"],
        num_workers=num_workers,
    )
    dataset.setup(stage="fit")
    return dataset


def setup_models(config, args):
    """Set up the UNet and Diffusion models."""
    encoder_model_path = config["diffusion"].pop("encoder_path")
    UNet_model = UNetConditional(**config["unet"])
    encoder_model = VAE.load_from_checkpoint(encoder_model_path, map_location="cuda:0")

    diffusion_model = DiffusionModel(
        model=UNet_model,
        encoder_model=encoder_model,
        **config["diffusion"],
    )

    return UNet_model, diffusion_model


def setup_logger(config, diffusion_model):
    """Set up the WandB logger."""
    wandb_logger = WandbLogger(project=config["training"]["project_name"])
    wandb_logger.watch(diffusion_model)
    return wandb_logger


def setup_trainer(config, args, callbacks, logger):
    """Set up the PyTorch Lightning Trainer."""
    return pl.Trainer(
        accelerator="auto",
        devices=config["device"]["cuda"],
        num_nodes=args.num_nodes,
        max_epochs=config["training"]["num_epochs"],
        callbacks=callbacks,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        precision="bf16-mixed",
        logger=logger,
    )


def train_model():
    args = parse_arguments()
    config = get_params(config_file=args.config_file)

    # Setup directories and save config
    setup_directories(config["training"]["output_path"])
    save_config(config, config["training"]["output_path"])

    # Initialize WandB
    wandb_init(config["training"]["project_name"])

    # Setup checkpoints
    checkpoint_callback, save_last_checkpoint = setup_checkpoints(
        config["training"]["output_path"]
    )
    ckpt_path = get_checkpoint_path(config["training"]["output_path"])

    # Setup data module
    dataset = setup_data_module(config, args.num_workers)

    # Setup models
    UNet_model, diffusion_model = setup_models(config)

    # Save model architecture
    with open(f"{config['training']['output_path']}/model_architecture.txt", "w") as f:
        f.write(str(UNet_model))

    # Setup logger
    wandb_logger = setup_logger(config, diffusion_model)

    # Setup trainer
    trainer = setup_trainer(
        config,
        args,
        callbacks=[
            checkpoint_callback,
            save_last_checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=wandb_logger,
    )

    # Start training
    trainer.fit(diffusion_model, dataset, ckpt_path=ckpt_path)

    # Detach WandB logging
    wandb_logger.experiment.unwatch(diffusion_model)
    wandb.finish()


if __name__ == "__main__":
    train_model()
