import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from IPython import embed

from starling.data.argument_parser import get_params
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
        default=[0],
        help="GPU device ID for training",
    )

    args = parser.parse_args()

    # Set up data loaders
    dataset = MatrixDataModule(
        test_data=args.test_data, batch_size=args.batch_size, target_shape=192
    )

    device = torch.device(f"cuda:{args.gpu_id[0]}")
    model = VAE.load_from_checkpoint(args.model_path, map_location=device)
    model.eval()

    dataset.setup(stage="test")
    predict_dataloader = dataset.test_dataloader()

    # embed()

    for batch in predict_dataloader:
        x = batch["input"].to(f"cuda:{args.gpu_id[0]}")
        x_reconstructed = model(x)[0]
        np.save("ground_truth_mse.npy", x.cpu().detach().numpy())
        np.save("reconstructed_array_mse.npy", x_reconstructed.cpu().detach().numpy())
        break

    # trainer = pl.Trainer(
    #     devices=args.gpu_id,
    # )
    # embed()
    # predictions = trainer.predict(model, dataloaders=dataset)

    # for batch in dataset:
    #     x = batch["input"].to(f"cuda:{args.gpu_id}")
    #     x_reconstructed = model(x)[0]
    #     np.save("ground_truth.npy", x.cpu().detach().numpy())
    #     np.save("reconstructed_array.npy", x_reconstructed.cpu().detach().numpy())
    #     break


if __name__ == "__main__":
    vae_predict()
