import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from IPython import embed

from starling.data.argument_parser import get_params
from starling.data.myloader import MatrixDataModule
from starling.models.ae import AE
from starling.models.cvae import cVAE
from starling.models.vae import VAE


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

    device = torch.device(f"cuda:{args.gpu_id[0]}")
    model = cVAE.load_from_checkpoint(args.model_path, map_location=device)
    model.eval()

    # Set up data loaders
    dataset = MatrixDataModule(
        test_data=args.test_data,
        batch_size=args.batch_size,
        target_shape=384,
    )
    dataset.setup(stage="test")
    predict_dataloader = dataset.test_dataloader()

    latents = {}

    for batch in predict_dataloader:
        data = batch["data"].to(f"cuda:{args.gpu_id[0]}")
        encoder_labels = batch["encoder_condition"].to(f"cuda:{args.gpu_id[0]}")
        sequences = batch["decoder_condition"]

        with torch.no_grad():
            mu, logvar = model.encode(data, labels=encoder_labels)
            latent_encoding = model.reparameterize(mu, logvar)

        latent_encoding = latent_encoding.cpu().detach().numpy()
        for num, seq in enumerate(sequences):
            if seq in latents.keys():
                latents[seq].append(latent_encoding[num])
            else:
                latents[seq] = [latent_encoding[num]]

        embed()


if __name__ == "__main__":
    vae_generate_latents()
