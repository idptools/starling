import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from IPython import embed

from starling.data.argument_parser import get_params
from starling.models.ae import AE
from starling.models.cvae import cVAE
from starling.models.vae import VAE
from starling.training.myloader import MatrixDataModule


def vae_predict():
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
        default=None,
        help="GPU device ID for training",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="VAE",
        help="Type of a VAE used, currently [cVAE, VAE, AE] are supported",
    )

    args = parser.parse_args()

    model_type = {"cVAE": cVAE, "VAE": VAE, "AE": AE}

    device = torch.device(
        f"cuda:{args.gpu_id[0]}" if args.gpu_id is not None else "cpu"
    )
    model = model_type[args.model_type].load_from_checkpoint(
        args.model_path, map_location=device
    )

    input_dimension = model.hparams.get("dimension")

    model.eval()

    # Set up data loaders
    dataset = MatrixDataModule(
        test_data=args.test_data,
        batch_size=args.batch_size,
        target_shape=input_dimension,
    )

    dataset.setup(stage="test")
    predict_dataloader = dataset.test_dataloader()

    data = []
    data_reconstructed = []

    for batch in predict_dataloader:
        x = batch["data"].to(f"cuda:{args.gpu_id[0]}")
        x_reconstructed = model(x)[0]

        data.append(x.cpu().detach().numpy())
        data_reconstructed.append(x_reconstructed.cpu().detach().numpy())


    embed()
    np.save("ground_truth_small_384_model.npy", x.cpu().detach().numpy())
    np.save(
            "reconstructed_array_small_384_model.npy",
            x_reconstructed.cpu().detach().numpy(),
        )


if __name__ == "__main__":
    vae_predict()
