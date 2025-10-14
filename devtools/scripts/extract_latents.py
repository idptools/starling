import glob
import os

import h5py
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from starling.models.vae import VAE


def load_hdf5_compressed(file_path):
    """
    Loads data from an HDF5 file.

    Parameters:
    - file_path (str): Path to the HDF5 file.

    Returns:
    - dict: Dictionary containing loaded data.
    """
    data_dict = {}
    with h5py.File(file_path, "r", libver="latest", swmr=True) as f:
        for key in f.keys():
            data_dict[key] = f[key][...]
    return data_dict


def encode_data(model, data):
    """
    Encodes data using the provided VAE model in batches of 32.

    Parameters:
    - model (VAE): The VAE model to use for encoding.
    - data (numpy.ndarray): Data to be encoded.

    Returns:
    - numpy.ndarray: Encoded data.
    """
    batch_size = 64
    device = next(model.parameters()).device  # Get model's device dynamically

    # Process data in batches
    encoded_batches = []
    with torch.no_grad():  # Move this outside the loop for efficiency
        # Process batches
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch).to(device)
            batch_tensor = rearrange(batch_tensor, "b h w -> b 1 h w")
            encoded_batch = model.encode(batch_tensor).mode()
            encoded_batches.append(encoded_batch.cpu().numpy())

            # Optional: clear GPU memory (only if memory issues occur)
            # torch.cuda.empty_cache()

    # Concatenate all batches
    encoded_data = np.concatenate(encoded_batches, axis=0).squeeze()

    # Calculate the average latent vector
    average_latent = np.mean(encoded_data, axis=0)

    return encoded_data, average_latent


def save_h5(averange_latent, encoded_data, seq_data, file_path):
    """
    Saves encoded latent vectors and sequence data to an HDF5 file.

    Parameters:
    - encoded_data (numpy.ndarray): The encoded latent vectors to save.
    - seq_data (numpy.ndarray): The sequence data to save.
    - file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, "w") as f:
        f.create_dataset("seq", data=seq_data)
        f.create_dataset("latents", data=encoded_data)
        f.create_dataset("average_latent", data=averange_latent)


def main():
    # Load the VAE model
    model = VAE.load_from_checkpoint(
        "/work/j.lotthammer/projects/starling/VAEs/model-kernel-epoch=39-epoch_val_loss=1.82.ckpt",
        map_location="cuda",
    )
    model.eval()

    data_paths = sorted(
        glob.glob(
            "/work/bnovak/projects/sequence2ensemble/lammps_data/300mM_data/mPIPIgg_300mM/*.h5"
        )
    )

    data_paths = [path for path in data_paths if "vae_encoded" not in path]

    for data_path in tqdm(data_paths):
        outfile = data_path.replace(".h5", "_adamW_vae_encoded.h5")
        if os.path.exists(outfile):
            print(f"File {outfile} already exists. Skipping.")
            continue
        # Load the data
        data = load_hdf5_compressed(data_path)

        # Encode the data
        try:
            encoded_data, average_latent = encode_data(model, data["dm"])
        except Exception as e:
            print(f"Error encoding data from {data_path}: {e}")
            continue

        # Save the encoded data to a new HDF5 file
        save_h5(average_latent, encoded_data, data["seq"], outfile)


if __name__ == "__main__":
    main()
