import glob
import os

import hydra
import pytorch_lightning as pl
import torch
import wandb
import yaml
from typing import Any, cast
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from starling.data.ddpm_loader_tar import DDPMDataLoader
from starling.models.diffusion import DiffusionModel
from starling.models.transformer import SequenceEncoder
from starling.models.vit import ViT
from starling.utilities import MODEL_FLOAT32_MATMUL_PRECISION

# ViT backbone hyperparameters of the released STARLING diffusion model. These
# match the ViT that starling/inference/model_loading.py builds to load the
# released weights, so changing them makes new checkpoints incompatible with it.
VIT_NUM_LAYERS = 12
VIT_EMBED_DIM = 512
VIT_NUM_HEADS = 8


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
        dirpath=output_path,
        filename="model-kernel-{epoch:02d}-{epoch_val_loss:.2f}",
        monitor="epoch_val_loss",
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


def get_resume_checkpoint(trainer_config: DictConfig, output_path: str) -> str | None:
    """
    Decide which checkpoint, if any, trainer.fit should resume from.

    This mirrors vae_train. When fine-tuning, trainer.checkpoint only supplies the
    starting weights (see setup_models), so nothing is resumed. Otherwise an explicit
    trainer.checkpoint is resumed as given, and if none is set, training resumes from
    the last.ckpt that an earlier run left in output_path.

    Parameters
    ----------
    trainer_config : DictConfig
        The trainer section of the Hydra config; reads fine_tune and checkpoint.
    output_path : str
        Directory where checkpoints for this run are written.

    Returns
    -------
    str or None
        The checkpoint to pass as trainer.fit(ckpt_path=...). This is the explicit
        checkpoint path, "last" (which Lightning resolves to the newest last
        checkpoint in output_path), or None to start from scratch.
    """
    if trainer_config.fine_tune:
        return None
    if trainer_config.checkpoint is not None:
        return trainer_config.checkpoint
    return get_checkpoint_path(output_path)


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
    """Set up the data module for diffusion training."""

    if cfg.dataloader.type == "h5":
        # The h5 block in configs/dataloader/dataloader.yaml has no _target_ and no
        # h5 diffusion data module exists, so only tar shards are supported
        raise NotImplementedError(
            "dataloader.type 'h5' is not supported for diffusion training; "
            "set dataloader.type to 'tar' and point dataloader.tar.dataset at "
            "webdataset tar shards"
        )

    elif cfg.dataloader.type == "tar":
        dataset = DDPMDataLoader(
            config=cfg.dataloader.tar, effective_batch_size=effective_batch_size
        )
        dataset.setup(stage="fit")
    else:
        raise ValueError(f"Unsupported dataloader type: {cfg.dataloader.type}")

    return dataset


def setup_models(config):
    """Set up the ViT backbone and Diffusion model."""
    model_path = config.trainer.checkpoint

    diffusion_models = {"discrete": DiffusionModel}

    seq_encoder_dict = cast(
        dict[str, Any], OmegaConf.to_container(config.sequence_encoder, resolve=True)
    )
    # The ViT cross-attends to the sequence encoder output, so its context
    # dimension must equal the sequence encoder's embed_dim
    vit = ViT(
        VIT_NUM_LAYERS,
        VIT_EMBED_DIM,
        VIT_NUM_HEADS,
        context_dim=seq_encoder_dict["embed_dim"],
    )
    sequence_encoder = SequenceEncoder(**seq_encoder_dict)

    if config.diffusion.type == "discrete":
        diffusion_config_dict = cast(
            dict[str, Any],
            OmegaConf.to_container(config.diffusion.discrete, resolve=True),
        )
    else:
        raise ValueError(f"Unsupported diffusion type: {config.diffusion.type}")

    if config.trainer.fine_tune:
        # Load on the CPU; without a map_location every DDP rank would restore
        # the checkpoint onto the GPU it was saved from (cuda:0). Lightning
        # moves the model to each rank's device.
        diffusion_model = diffusion_models[config.diffusion.type].load_from_checkpoint(
            model_path,
            model=vit,
            sequence_encoder=sequence_encoder,
            map_location="cpu",
            **diffusion_config_dict,
        )
    else:
        diffusion_model = diffusion_models[config.diffusion.type](
            model=vit,
            sequence_encoder=sequence_encoder,
            **diffusion_config_dict,
        )

    return diffusion_model


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
    # Training owns the whole process, so the models' TF32 matmul precision
    # is set process-wide here; library code only ever scopes it
    torch.set_float32_matmul_precision(MODEL_FLOAT32_MATMUL_PRECISION)

    # Setup directories and save config
    output_path = cfg.trainer.output_path
    setup_directories(output_path)

    # Save the config for reference
    OmegaConf.save(cfg, f"{output_path}/config.yaml")

    # Initialize WandB
    wandb_init(cfg.trainer.project_name, id=cfg.trainer.get("wandb_id", None))

    # Setup checkpoints
    checkpoint_callback, save_last_checkpoint = setup_checkpoints(output_path)
    ckpt_path = get_resume_checkpoint(cfg.trainer, output_path)
    if ckpt_path is not None:
        print(f"Resuming from checkpoint: {ckpt_path}")

    # Setup models
    diffusion_model = setup_models(cfg)

    # Save model architecture
    with open(f"{output_path}/model_architecture.txt", "w") as f:
        f.write(str(diffusion_model))

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

    # Setup data module after the Trainer so the effective batch size uses the
    # world size Lightning resolved from trainer.cuda and num_nodes
    if cfg.dataloader.type == "tar":
        effective_batch_size = get_effective_batch_size(
            trainer, cfg.dataloader.tar.batch_size
        )
    else:
        effective_batch_size = None

    dataset = setup_data_module(cfg, effective_batch_size=effective_batch_size)

    # Start training
    trainer.fit(diffusion_model, dataset, ckpt_path=ckpt_path)

    # Detach WandB logging
    wandb_logger.experiment.unwatch(diffusion_model)
    wandb.finish()


if __name__ == "__main__":
    train_model()
