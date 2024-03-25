import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from IPython import embed

from starling.models.vae import VAE
from starling.training.myloader import MatrixDataModule


def vae_predict():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the checkpoint to load in",
    )

    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Data you want to test your model on",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="weighted_mse",
        help="""What loss to calculate for the reconstruction loss, current losses include 
        mse and weighted_mse""",
    )

    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="""Whether to normalize the distance map. Options:[length, afrc, bond_length].
        Length is normalizing by np.sqrt(length), afrc by AFRC distance map, and bond_length 
        by the accumulated count of residues for each element within a distance map*bond_length""",
    )

    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Enable interpolation of distance maps through torch resizing using bicubic method",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID for training",
    )

    args = parser.parse_args()

    dataset = MatrixDataModule(
        train_data=args.test_data,
        val_data=args.test_data,
        test_data=args.test_data,
        predict_data=args.test_data,
        args=args,
        batch_size=args.batch_size,
    )

    # dataset.setup()
    # predict_dataloader = dataset.predict_dataset()

    # device = torch.device(f"cuda:{args.gpu_id}")
    device = torch.device(f"cuda:{args.gpu_id[0]}")
    model = VAE.load_from_checkpoint(args.model_path, map_location=device)
    model.eval()

    dataset.setup(stage="predict")
    predict_dataloader = dataset.predict_dataloader()

    # embed()

    for batch in predict_dataloader:
        x = batch["input"].to(f"cuda:{args.gpu_id[0]}")
        x_reconstructed = model(x)[0]
        np.save("ground_truth_384.npy", x.cpu().detach().numpy())
        np.save("reconstructed_array_384.npy", x_reconstructed.cpu().detach().numpy())
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
