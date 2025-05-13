import argparse
import glob
import os

import hydra
import pytorch_lightning as pl
import wandb
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import starling.data.ddpm_loader as ddpm_loader
import starling.data.ddpm_loader_tar as ddpm_loader_tar
from starling.data.argument_parser import get_params
from starling.models.continuous_diffusion import ContinuousDiffusion
from starling.models.diffusion import DiffusionModel
from starling.models.unet import UNetConditional
from starling.models.vae import VAE


@rank_zero_only
def wandb_init(project: str = "starling"):
    wandb.init(project=project)


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


def setup_data_module(config):
    """Set up the data module."""
    data_config = config.dataloader
    effective_batch_size = data_config.batch_size * config.trainer.cuda
    if data_config.type == "tar":
        dataset = ddpm_loader_tar.DDPMDataloader(
            data_config, effective_batch_size=effective_batch_size
        )
    elif data_config.type == "h5":
        dataset = ddpm_loader.DDPMDataloader(data_config)
    else:
        raise ValueError(f"Unsupported data type: {data_config.type}")
    dataset.setup(stage="fit")
    return dataset


def setup_models(config):
    """Set up the UNet and Diffusion models."""
    model_path = config.trainer.checkpoint

    diffusion_models = {"discrete": DiffusionModel, "continuous": ContinuousDiffusion}

    unet_config_dict = OmegaConf.to_container(config.unet, resolve=True)
    UNet_model = UNetConditional(**unet_config_dict)

    if config.diffusion.type == "continuous":
        diffusion_config_dict = OmegaConf.to_container(
            config.diffusion.continuous, resolve=True
        )
    elif config.diffusion.type == "discrete":
        diffusion_config_dict = OmegaConf.to_container(
            config.diffusion.discrete, resolve=True
        )
    else:
        raise ValueError(f"Unsupported diffusion type: {config.diffusion.type}")

    if config.trainer.fine_tune:
        diffusion_model = diffusion_models[config.diffusion.type].load_from_checkpoint(
            model_path,
            model=UNet_model,
            **diffusion_config_dict,
        )
    else:
        diffusion_model = diffusion_models[config.diffusion.type](
            model=UNet_model,
            **diffusion_config_dict,
        )

    return UNet_model, diffusion_model


def setup_logger(config, diffusion_model):
    """Set up the WandB logger."""
    wandb_logger = WandbLogger(project=config.trainer.project_name)
    wandb_logger.watch(diffusion_model)
    return wandb_logger


def setup_trainer(config, callbacks, logger):
    """Set up the PyTorch Lightning Trainer."""
    return pl.Trainer(
        accelerator="auto",
        devices=config.trainer.cuda,
        num_nodes=config.trainer.num_nodes,
        max_epochs=config.trainer.num_epochs,
        callbacks=callbacks,
        gradient_clip_val=config.trainer.gradient_clip_val,
        precision=config.trainer.precision,
        logger=logger,
    )


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.path.dirname(__file__), "../configs"),
    config_name="configs",
)
def train_model(cfg: DictConfig):
    # Setup directories and save config
    output_path = cfg.trainer.output_path
    setup_directories(output_path)

    # Save the config for reference
    OmegaConf.save(cfg, f"{output_path}/config.yaml")

    # Initialize WandB
    wandb_init(cfg.trainer.project_name)

    # Setup checkpoints
    checkpoint_callback, save_last_checkpoint = setup_checkpoints(output_path)
    ckpt_path = cfg.trainer.checkpoint

    # Setup data module
    dataset = setup_data_module(cfg)

    # Setup models
    UNet_model, diffusion_model = setup_models(cfg)

    # Save model architecture
    with open(f"{output_path}/model_architecture.txt", "w") as f:
        f.write(str(UNet_model))

    # Setup logger
    wandb_logger = setup_logger(cfg, diffusion_model)

    # Setup trainer
    trainer = setup_trainer(
        cfg,
        callbacks=[
            checkpoint_callback,
            save_last_checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=wandb_logger,
    )

    # Start training
    trainer.fit(
        diffusion_model, dataset, ckpt_path=None if cfg.trainer.fine_tune else ckpt_path
    )

    # Detach WandB logging
    wandb_logger.experiment.unwatch(diffusion_model)
    wandb.finish()


if __name__ == "__main__":
    train_model()
