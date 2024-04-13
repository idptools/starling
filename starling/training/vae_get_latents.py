import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from IPython import embed

from starling.data.argument_parser import get_params
from starling.models.ae import AE
from starling.models.vae import VAE
from starling.training.myloader import MatrixDataModule


def vae_generate_latents():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the config file for wandb sweep",
    )

    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to the config file for wandb sweep",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID for training",
    )

    args = parser.parse_args()

    # Set up data loaders
    dataset = MatrixDataModule(
        test_data=args.test_data,
        batch_size=args.batch_size,
        target_shape=192,
    )

    device = torch.device(f"cuda:{args.gpu_id[0]}")
    # model = VAE.load_from_checkpoint(args.model_path, map_location=device)
    model = AE.load_from_checkpoint(args.model_path, map_location=device)
    model.eval()

    dataset.setup(stage="test")
    predict_dataloader = dataset.test_dataloader()

    latents = []

    for batch in predict_dataloader:
        x = batch["input"].to(f"cuda:{args.gpu_id[0]}")
        # mu, logvar = model.encode(x)
        # latent_encoding = model.reparameterize(mu, logvar)
        latent_encoding = model.encode(x)
        latents.append(latent_encoding.cpu().detach().numpy())

    embed()


if __name__ == "__main__":
    vae_generate_latents()
