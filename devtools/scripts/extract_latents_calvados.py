import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from starling.data.tokenizer import StarlingTokenizer
from starling.models.vae import VAE
from starling.utilities import symmetrize_distance_maps


def pad_dm(dm):
    """
    Pads the distance map to a fixed size of 384x384.

    Parameters:
    - dm (numpy.ndarray): The distance map to be padded.

    Returns:
    - numpy.ndarray: Padded distance map.
    """
    return np.pad(
        dm, ((0, 0), (0, 384 - dm.shape[1]), (0, 384 - dm.shape[2])), mode="constant"
    )


def load_hdf5_compressed(file_path):
    """
    Loads data from an HDF5 file, specifically only 'dm' and 'seq' keys.

    Parameters:
    - file_path (str): Path to the HDF5 file.

    Returns:
    - dict: Dictionary containing loaded data for 'dm' and 'seq' keys.
    """
    data_dict = {}
    with h5py.File(file_path, "r") as f:
        for key in ["dm", "seq"]:
            if key in f:
                data_dict[key] = f[key][...]
            else:
                print(f"Warning: Key '{key}' not found in {file_path}")

    data_dict["seq"] = data_dict["seq"][()].decode()
    if len(data_dict["seq"]) > 384:
        return None
    data_dict["seq"] = data_dict["seq"].ljust(384, "0")
    data_dict["dm"] = symmetrize_distance_maps(data_dict["dm"])
    data_dict["dm"] = pad_dm(data_dict["dm"])
    data_dict["dm"] = data_dict["dm"].astype(np.float32)

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
    seq_data = StarlingTokenizer().encode(seq_data)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("seq", data=seq_data)
        f.create_dataset("latents", data=encoded_data)
        f.create_dataset("average_latent", data=averange_latent)


def main():
    # Load the VAE model
    model = VAE.load_from_checkpoint(
        "/work/bnovak/projects/sequence2ensemble/starling/continuous_starling/encoder/model-kernel-epoch=99-epoch_val_loss=1.72.ckpt",
        map_location="cuda",
    )
    model.eval()

    outdir = "/work/bnovak/projects/sequence2ensemble/lammps_data/calvados/latents"

    data_paths = sorted(
        glob.glob(
            "/work/j.lotthammer/projects/albatross_calvados_datasets/calvados/albatross/*"
        )
    )

    for data_path in tqdm(data_paths):
        # Check if output file already exists
        name = Path(data_path).name
        outfile = Path(outdir) / (name + "_vae_encoded.h5")

        if outfile.exists():
            print(f"Skipping {data_path} as output file already exists.")
            continue

        # Load the data
        data = load_hdf5_compressed(data_path)
        if data is None:
            print(f"Skipping {data_path} due to sequence longer than 384.")
            continue

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
