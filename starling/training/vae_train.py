import argparse
import os

import pytorch_lightning as pl
import wandb
import yaml
from IPython import embed
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from starling.data.argument_parser import get_params
from starling.models.cvae import cVAE
from starling.models.vae import VAE
from starling.training.myloader import MatrixDataModule


def train_vae():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration file to use",
    )

    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Path to the pretrained model to start training from",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Type of a VAE used, currently [cVAE, VAE] are supported",
    )

    args = parser.parse_args()

    # Types of VAE models that can be trained using this script
    # VAE is a regular unconditional VAE
    # cVAE is a conditional VAE that can be pretrained with no labels
    model_type = {"cVAE": cVAE, "VAE": VAE}

    # Reads in default and user defined configuration arguments
    config = get_params(config_file=args.config_file)

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
        **config["data"], target_shape=config["model"]["dimension"]
    )

    dataset.setup(stage="fit")

    # Whether to load a pretrained model or train from scratch
    if args.pretrained_model is None:
        vae = model_type[args.model_type](**config["model"])
    else:
        vae = model_type[args.model_type].load_from_checkpoint(
            args.pretrained_model, map_location=f'cuda:{config["device"]["cuda"][0]}'
        )
        # Change the params to whatever they
        for param in config["model"]:
            vae.hparams[param] = config["model"][param]

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
