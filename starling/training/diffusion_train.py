import argparse
import os

import pytorch_lightning as pl
import torch
import wandb
import yaml
from IPython import embed
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from starling.data.argument_parser import get_params
from starling.data.myloader import MatrixDataModule
from starling.models.cvae import cVAE
from starling.models.diffusion import DiffusionModel
from starling.models.unet import UNet


def train_vae():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the configuration file to use",
    )

    args = parser.parse_args()

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

    # # Set up data loaders
    dataset = MatrixDataModule(**config["data"], target_shape=384)

    dataset.setup(stage="fit")
    # from torch.utils.data import DataLoader, random_split
    # from torchvision import transforms
    # from torchvision.datasets import MNIST

    # transform_with_padding = transforms.Compose(
    #     [
    #         transforms.Pad(2, 2),  # Add padding to the images
    #         transforms.ToTensor(),
    #         # transforms.Normalize((0.5,), (0.5,)),
    #     ]
    # )

    # train_ds = MNIST(
    #     "MNIST/raw/train-images-idx3-ubyte",
    #     train=True,
    #     download=True,
    #     transform=transform_with_padding,
    # )
    # dataset = DataLoader(train_ds, batch_size=256)

    # Loading in a model from diffusers, will replace with my own UNet model
    # Assuming I can make it work

    import diffusers

    UNet_model = diffusers.UNet2DModel(
        sample_size=16,
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        class_embed_type="identity",
        block_out_channels=(128, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    device = torch.device(f"cuda:{config['device']['cuda'][0]}")
    encoder_model = cVAE.load_from_checkpoint(
        "/home/bnovak/projects/autoencoder_training/VAE_training/testing_cond_vae/nll_ESM_8M_conditioning_decoder_conditioning_mlp/model-kernel-epoch=00-epoch_val_loss=5.84.ckpt",
        map_location=device,
    )

    diffusion_model = DiffusionModel(
        model=UNet_model,
        encoder_model=encoder_model,
        image_size=16,
        beta_scheduler="cosine",
        timesteps=1000,
        schedule_fn_kwargs=None,
    )

    # Make the directories to save the model and logs
    os.makedirs(config["training"]["output_path"], exist_ok=True)

    with open(f"{config['training']['output_path']}/config.yaml", "w") as f:
        yaml.dump(config, f)

    with open(f"{config['training']['output_path']}/model_architecture.txt", "w") as f:
        f.write(str(UNet_model))
    ##############################

    # Set up logging on weights and biases
    wandb_logger = WandbLogger(project=config["training"]["project_name"])
    wandb_logger.watch(diffusion_model)

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
    trainer.fit(diffusion_model, dataset)

    # Detach the logging on wandb
    wandb_logger.experiment.unwatch(diffusion_model)
    wandb.finish()


if __name__ == "__main__":
    train_vae()
