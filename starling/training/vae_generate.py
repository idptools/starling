import argparse
import time

import numpy as np
import torch
from IPython import embed

from starling.models.vae import VAE


def vae_generate():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the checkpoint to load in",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate",
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
        help="GPU device ID for training",
    )
    parser.add_argument(
        "--outfile_name",
        type=str,
        default="generated_distance_maps.npy",
        help="Path and filename to use",
    )

    args = parser.parse_args()

    num_batches = args.num_samples // args.batch_size
    remaining_samples = args.num_samples % args.batch_size

    device = torch.device(f"cuda:{args.gpu_id[0]}" if args.gpu_id else "cpu")
    model = VAE.load_from_checkpoint(args.model_path, map_location=device)
    model.eval()

    start = time.time()

    all_distance_maps = []

    latent_dimension = model.hparams.get("latent_dim")
    with torch.no_grad():
        for i in range(num_batches):
            encodings = torch.randn(args.batch_size, latent_dimension).to(device)
            distance_maps = model.decode(encodings)
            all_distance_maps.append(distance_maps.cpu().detach().numpy())

        if remaining_samples > 0:
            encodings = torch.randn(remaining_samples, latent_dimension).to(device)
            distance_maps = model.decode(encodings)
            all_distance_maps.append(distance_maps.cpu().detach().numpy())

    end = time.time()
    elapsed_time = end - start

    print(f"Generated {args.num_samples} in {elapsed_time:.4f} seconds")

    all_distance_maps = np.concatenate(all_distance_maps, axis=0)
    np.save(args.outfile_name, all_distance_maps)


if __name__ == "__main__":
    vae_generate()
