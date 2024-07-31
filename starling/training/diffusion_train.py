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

# from starling.models.diffusion_test_conditional import DiffusionModel
from starling.models.diffusion import DiffusionModel
from starling.models.unet import UNetConditional, UNetConditionalTest
from starling.models.vae import VAE

# labels="esm2_t30_150M_UR50D",
# labels="esm2_t12_35M_UR50D",
labels = "learned-embeddings"
labels_dim = 384


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
    dataset = MatrixDataModule(**config["data"], target_shape=384, labels=labels)

    dataset.setup(stage="fit")

    # Loading in a model from diffusers, will replace with my own UNet model
    # Assuming I can make it work

    import diffusers

    # UNet_model = diffusers.UNet2DConditionModel(
    #     sample_size=24,  # the target image resolution
    #     in_channels=1,  # the number of input channels, 3 for RGB images
    #     out_channels=1,  # the number of output channels
    #     layers_per_block=2,  # how many ResNet layers to use per UNet block
    #     block_out_channels=(128, 128, 256, 256),
    #     cross_attention_dim=640,
    #     # encoder_hid_dim_type="text_proj",
    #     # encoder_hid_dim=100,
    # )
    # UNet_model = diffusers.UNet2DConditionModel(
    #     sample_size=24,  # the target image resolution
    #     in_channels=1,  # the number of input channels, 3 for RGB images
    #     out_channels=1,  # the number of output channels
    #     cross_attention_dim=480,
    #     block_out_channels=(160, 320, 640, 640),
    # )

    UNet_model = UNetConditional(
        in_channels=1,
        out_channels=1,
        base=64,
        norm="group",
        blocks=[2, 2, 2],
        middle_blocks=2,
        labels_dim=labels_dim,
    )

    gpu_ids = config["device"]["cuda"]
    map_location = {
        f"cuda:{i}": f"cuda:{gpu_ids[i % len(gpu_ids)]}" for i in range(len(gpu_ids))
    }

    encoder_model = cVAE.load_from_checkpoint(
        "/home/bnovak/github/starling/starling/models/trained_models/renamed_keys_model-kernel-epoch=09-epoch_val_loss=1.72.ckpt",
        map_location=map_location,
    )

    diffusion_model = DiffusionModel(
        model=UNet_model,
        encoder_model=encoder_model,
        image_size=24,
        beta_scheduler="cosine",
        timesteps=1000,
        schedule_fn_kwargs=None,
        set_lr=1e-4,
        labels=labels,
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
