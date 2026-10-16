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
from starling.utilities import MODEL_FLOAT32_MATMUL_PRECISION


@rank_zero_only
def wandb_init(project: str = "starling", id=None):
    wandb.init(project=project, resume="allow", id=id)


def setup_directories(output_path):
    """Create the output directory if it does not already exist."""
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


def get_effective_batch_size(trainer: pl.Trainer, batch_size: int) -> int:
    """
    Compute the number of samples consumed per training step across all processes.

    The world size is read from the Trainer after construction because Lightning has
    already resolved trainer.cuda (an int, -1 for all available devices, or a list of
    device indices) and num_nodes into the number of processes it will launch.

    Parameters
    ----------
    trainer : pl.Trainer
        The constructed Trainer that will run training.
    batch_size : int
        The per-process batch size from the dataloader config.

    Returns
    -------
    int
        world_size * batch_size.

    Raises
    ------
    ValueError
        If the world size or batch size is not a positive integer.
    """
    world_size = trainer.world_size
    if world_size < 1 or batch_size < 1:
        raise ValueError(
            f"Expected a positive world size and batch size, got world_size={world_size} "
            f"and batch_size={batch_size}"
        )
    return world_size * batch_size


def setup_data_module(cfg, effective_batch_size=None):
    """Set up the data module for VAE training."""

    if cfg.dataloader.type == "h5":
        # The h5 block in configs/dataloader/vae_dataloader.yaml targets
        # starling.data.VAE_loader.MatrixDataModule, which no longer exists, so only
        # tar shards are supported
        raise NotImplementedError(
            "dataloader.type 'h5' is not supported for VAE training; "
            "set dataloader.type to 'tar' and point dataloader.tar.dataset at "
            "webdataset tar shards"
        )

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
    """Set up the VAE model, optionally initialized from checkpoint weights for fine-tuning.

    When fine-tuning, the model is built from cfg.vae_model and the checkpoint's
    weights are loaded strictly, so cfg.vae_model must describe the same architecture
    as the checkpoint. Only settings that do not change the weights (e.g. learning
    rate, scheduler, KLD weighting) can differ from the original run.
    """
    model_path = cfg.trainer.get("checkpoint", None)

    if cfg.trainer.get("fine_tune", False) and model_path:
        print(f"Fine-tuning VAE from checkpoint: {model_path}")
        # First instantiate with custom arguments
        vae = instantiate(cfg.vae_model)

        # Load state dict from checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        # Load the weights strictly, so any missing or unexpected key raises
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
    # Training owns the whole process, so the models' TF32 matmul precision
    # is set process-wide here; library code only ever scopes it
    torch.set_float32_matmul_precision(MODEL_FLOAT32_MATMUL_PRECISION)

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
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
    )

    # Setup data module after the Trainer so the effective batch size uses the
    # world size Lightning resolved from trainer.cuda and num_nodes
    if cfg.dataloader.type == "tar":
        effective_batch_size = get_effective_batch_size(
            trainer, cfg.dataloader.tar.batch_size
        )
    else:
        effective_batch_size = None

    dataset = setup_data_module(cfg, effective_batch_size=effective_batch_size)

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
