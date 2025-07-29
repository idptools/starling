import argparse
import glob
import os

import hydra
import pytorch_lightning as pl
import torch
import wandb
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from starling.models.vae import VAE


@rank_zero_only
def wandb_init(project: str = "starling", id=None):
    wandb.init(project=project, resume="allow", id=id)


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
        save_top_k=-1,
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


def setup_data_module(cfg, effective_batch_size=None):
    """Set up the data module for VAE training."""

    if cfg.dataloader.type == "h5":
        dataloader_config = cfg.dataloader.h5
        dataset = instantiate(dataloader_config)
        dataset.setup(stage="fit")

    elif cfg.dataloader.type == "tar":
        from starling.data.VAE_loader_tar import VAEdataloader

        dataset = VAEdataloader(
            config=cfg.dataloader.tar, effective_batch_size=effective_batch_size
        )
        dataset.setup(stage="fit")
    else:
        raise ValueError(f"Unsupported dataloader type: {cfg.dataloader.type}")

    return dataset


def setup_vae_model(cfg):
    """Set up the VAE model, with support for resuming from checkpoint or fine-tuning with custom args."""
    model_path = cfg.trainer.get("checkpoint", None)

    if cfg.trainer.get("fine_tune", False) and model_path:
        print(f"Fine-tuning VAE from checkpoint: {model_path}")
        # First instantiate with custom arguments
        vae = instantiate(cfg.vae_model)

        # Load state dict from checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        # Load only the weights, ignoring missing or extra keys
        vae.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint weights with custom model configuration")
    else:
        vae = instantiate(cfg.vae_model)

    return vae


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.path.dirname(__file__), "../configs"),
    config_name="vae_configs",
)
def train_vae(cfg: DictConfig):
    """Train a VAE model using the configuration specified by Hydra.

    Supports:
    - Training from scratch
    - Resuming training from a checkpoint
    - Fine-tuning from a checkpoint

    Args:
        cfg: The configuration object loaded by Hydra
    """
    # Setup directories and save config
    output_path = cfg.trainer.output_path
    os.makedirs(output_path, exist_ok=True)

    # Save the config for reference
    OmegaConf.save(cfg, f"{output_path}/config.yaml")

    # Initialize WandB
    wandb_init(cfg.trainer.project_name, id=cfg.trainer.get("wandb_id", None))

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

    # Determine if we should resume from checkpoint
    ckpt_path = cfg.trainer.get("checkpoint", None)
    if not cfg.trainer.get("fine_tune", False) and ckpt_path is None:
        # Try to find the last checkpoint if not explicitly provided
        last_checkpoint = os.path.join(output_path, "last.ckpt")
        if os.path.exists(last_checkpoint):
            ckpt_path = last_checkpoint
            print(f"Resuming from last checkpoint: {ckpt_path}")

    # Setup data module

    if cfg.dataloader.type == "tar":
        effective_batch_size = (
            cfg.trainer.cuda * cfg.trainer.num_nodes * cfg.dataloader.tar.batch_size
        )
    else:
        effective_batch_size = None

    dataset = setup_data_module(cfg, effective_batch_size=effective_batch_size)

    # Setup VAE model
    vae = setup_vae_model(cfg)

    # Save model architecture
    with open(f"{output_path}/model_architecture.txt", "w") as f:
        f.write(str(vae))

    # Setup logger
    wandb_logger = WandbLogger(project=cfg.trainer.project_name)
    wandb_logger.watch(vae)

    # Setup trainer with all callbacks
    trainer = pl.Trainer(
        accelerator="auto",
        devices=cfg.trainer.cuda,
        num_nodes=cfg.trainer.num_nodes,
        max_epochs=cfg.trainer.num_epochs,
        callbacks=[
            checkpoint_callback,
            save_last_checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
        precision=cfg.trainer.precision,
        logger=wandb_logger,
    )

    # Start training (don't pass checkpoint path when fine-tuning)
    trainer.fit(
        vae,
        dataset,
        ckpt_path=None if cfg.trainer.get("fine_tune", False) else ckpt_path,
    )

    # Detach WandB logging
    wandb_logger.experiment.unwatch(vae)
    wandb.finish()


if __name__ == "__main__":
    train_vae()
