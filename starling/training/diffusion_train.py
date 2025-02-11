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
def wandb_init(project: str = "starling-vista-diffusion"):
    wandb.init(project=project)


def train_model():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration file to use",
    )

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use for training",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers to use for loading in the data",
    )

    args = parser.parse_args()

    config = get_params(config_file=args.config_file)

    # Make the directories to save the model and logs
    os.makedirs(config["training"]["output_path"], exist_ok=True)
    with open(f"{config['training']['output_path']}/config.yaml", "w") as f:
        yaml.dump(config, f)

    wandb_init(config["training"]["project_name"])

    # Set up model checkpoint saving
    checkpoint_callback = ModelCheckpoint(
        monitor="epoch_val_loss",  # Monitor validation loss for saving the best model
        dirpath=f"{config['training']['output_path']}/",  # Directory to save checkpoints
        filename="model-kernel-{epoch:02d}-{epoch_val_loss:.2f}",  # File name format for saved models
        save_top_k=1,
        mode="min",  # Minimize the monitored metric (val_loss)
    )

    # delta = datetime.timedelta(hours=1)
    save_last_checkpoint = ModelCheckpoint(
        dirpath=f"{config['training']['output_path']}/",  # Directory to save checkpoints
        filename="last",
    )

    checkpoint_dir = f"{config['training']['output_path']}/"
    checkpoint_pattern = os.path.join(checkpoint_dir, "last.ckpt")

    # Check if any checkpoint exists
    checkpoint_files = glob.glob(checkpoint_pattern)

    if checkpoint_files:
        # This will load the most recent checkpoint
        ckpt_path = "last"
    else:
        # This will start training from scratch
        ckpt_path = None

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Set up data loaders
    dataset = MatrixDataModule(
        **config["data"],
        labels=config["diffusion"]["labels"],
        num_workers=args.num_workers,
    )

    dataset.setup(stage="fit")
    encoder_model_path = config["unet"].pop("encoder_path")
    UNet_model = UNetConditional(**config["unet"])

    map_location = "cuda:0"
    encoder_model = VAE.load_from_checkpoint(
        encoder_model_path, map_location=map_location
    )

    # TODO
    # SHOULD CHANGE TO DYNAMICALLY AUTOMATICALLY GET LATEN_DIM IN DIFFUSION.PY
    diffusion_model = DiffusionModel(
        model=UNet_model,
        encoder_model=encoder_model,
        **config["diffusion"],
    )

    with open(f"{config['training']['output_path']}/model_architecture.txt", "w") as f:
        f.write(str(UNet_model))

    # Set up logging on weights and biases
    wandb_logger = WandbLogger(project=config["training"]["project_name"])
    wandb_logger.watch(diffusion_model)

    # Set up PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=config["device"]["cuda"],
        num_nodes=args.num_nodes,
        max_epochs=config["training"]["num_epochs"],
        callbacks=[checkpoint_callback, lr_monitor, save_last_checkpoint],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        precision="bf16-mixed",
        logger=wandb_logger,
    )

    # Start training
    trainer.fit(diffusion_model, dataset, ckpt_path=ckpt_path)

    # Detach the logging on wandb
    wandb_logger.experiment.unwatch(diffusion_model)
    wandb.finish()


if __name__ == "__main__":
    train_model()
