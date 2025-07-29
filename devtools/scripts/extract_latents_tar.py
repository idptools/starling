import io
import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

from starling.models.vae import VAE

TARDIR = "/work/bnovak/projects/sequence2ensemble/lammps_data/combined_data/data/train/"
OUTDIR = "/work/bnovak/projects/sequence2ensemble/lammps_data/combined_data/latents_data_preprint_VAE_KLD_1e5_epoch_7/train/"
MODEL = "/work/bnovak/projects/sequence2ensemble/vae_training/preprint_VAE_cont_training_KLD_1e5/model-kernel-epoch=07-epoch_val_loss=1.90.ckpt"
BATCH_SIZE = 128


def load_model(model_path: str) -> VAE:
    """Load and prepare the VAE model for inference.

    Args:
        model_path: Path to the model checkpoint

    Returns:
        Loaded VAE model ready for inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE.load_from_checkpoint(model_path)
    model.eval()
    model.to(device)
    return model  # Add return statement


def npz_decoder(key: str, data: bytes) -> Optional[Dict]:
    """Decode NPZ files from WebDataset while preserving the source key.

    Args:
        key: The key identifying the sample
        data: The raw binary data of the NPZ file

    Returns:
        Dict containing the key and extracted array, or None if decoding failed
    """
    try:
        npz_data = np.load(io.BytesIO(data), allow_pickle=False)
        return {"__key__": key, "data": npz_data["array"]}
    except Exception as e:
        print(f"Error decoding {key}: {e}")
        return None


def process_sample(sample: Dict) -> Dict:
    """Process a sample by extracting and formatting the distance map.

    Args:
        sample: Dictionary containing the sample data

    Returns:
        Dict with the sample key and processed distance map
    """

    sample_key = sample.get("__key__", "unknown")
    distance_map = sample["distance_map.npz"]["data"]
    sequence = sample["sequence.npz"]["data"]
    # Ensure proper dimensions (add channel dimension if needed)
    if len(distance_map.shape) == 2:
        distance_map = distance_map[np.newaxis, :, :]

    return {"key": sample_key, "data": distance_map, "sequence": sequence}


def collate_fn(batch: List[Dict]) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """Collate individual samples into a batch for model processing.

    Args:
        batch: List of dictionaries containing sample data

    Returns:
        Tuple of (batched tensor data, list of sample keys),
        or None if batch is empty
    """
    # Filter out None values
    valid_samples = [item for item in batch if item is not None]

    if not valid_samples:
        return None

    # Extract keys and data
    sample_keys = [item["key"] for item in valid_samples]
    sample_data = [item["data"] for item in valid_samples]
    sequences = [item["sequence"] for item in valid_samples]

    # Stack data into a batch tensor
    data_batch = torch.tensor(np.stack(sample_data), dtype=torch.float32)

    return data_batch, sequences, sample_keys


def main():
    """Main function to process distance maps with a trained VAE model."""

    model = load_model(MODEL)
    device = next(model.parameters()).device  # Get device from model

    os.makedirs(OUTDIR, exist_ok=True)

    # Specify the input data file
    tar_files = sorted(glob(os.path.join(TARDIR, "*.tar")))

    dataset = (
        wds.WebDataset(tar_files, resampled=False, shardshuffle=False)
        .decode(npz_decoder)
        .map(process_sample)
        .batched(BATCH_SIZE, collation_fn=collate_fn, partial=False)
    )

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=4,
        prefetch_factor=2,
        shuffle=False,
    )

    tar_num = 0
    sample_count = 0
    tar_basename = f"train_{tar_num:04d}"
    output_tar_path = os.path.join(OUTDIR, f"{tar_basename}.tar")
    sink = wds.TarWriter(output_tar_path)

    print(f"Starting to write samples to {output_tar_path}")

    # Function to convert numpy array to compressed binary data
    def array_to_binary(array):
        """Convert numpy array to binary data for WebDataset."""
        buffer = io.BytesIO()
        np.savez_compressed(buffer, array=array)
        buffer.seek(0)
        return buffer.read()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            distance_maps, sequences, batch_keys = batch

            # Move data to appropriate device
            distance_maps = distance_maps.to(device)

            # Process the batch through the VAE model to get latent representations
            latent_vectors = model.encode(distance_maps).mode()

            for key, latent, sequence in zip(
                batch_keys, latent_vectors.cpu().numpy(), sequences
            ):
                # Create sample with all components using the helper function
                sample = {
                    "__key__": key,
                    "latent.npz": array_to_binary(latent),
                    "sequence.npz": array_to_binary(sequence),
                    "ionic_strength_mM.npz": array_to_binary(
                        int(key.split("_")[-1].replace("mM", ""))
                    ),
                }
                sink.write(sample)

                sample_count += 1

                # Start a new tar file after every 10,000 samples
                if sample_count % 10_000 == 0:
                    sink.close()
                    tar_num += 1
                    tar_basename = f"train_{tar_num:04d}"
                    output_tar_path = os.path.join(OUTDIR, f"{tar_basename}.tar")
                    print(
                        f"Completed {sample_count} samples. Starting new file: {output_tar_path}"
                    )
                    sink = wds.TarWriter(output_tar_path)

    # Close the final tar file
    print(f"Completed {sample_count} samples total. Closing final file.")
    sink.close()


if __name__ == "__main__":
    main()
