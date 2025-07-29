import io
import os
from collections import defaultdict
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

from starling.models.vae import VAE

TARDIR = "/home/borna/starling_data/validation"
MODEL = "/work/bnovak/projects/sequence2ensemble/vae_training/resnet34_adam_kld_1e4_lr_1e3_instance/model-kernel-epoch=35-epoch_val_loss=1.53.ckpt"
# MODEL = "/home/borna/.starling_weights/model-kernel-epoch=99-epoch_val_loss=1.72.ckpt"
BATCH_SIZE = 64
STATS_FILE = "loss_by_length_new_resnet34_VAE.txt"


def save_statistics(loss_by_length, max_loss_by_length, output_file=STATS_FILE):
    """Save loss statistics to a file.

    Args:
        loss_by_length: Dictionary mapping sequence lengths to lists of loss values
        output_file: Path to output file
    """
    print(f"\nSaving statistics to {output_file}...")
    with open(output_file, "w") as f:
        for length, losses in sorted(loss_by_length.items()):
            avg_loss = np.mean(losses)
            std_loss = np.std(losses) if len(losses) > 1 else 0
            max_losses = max_loss_by_length[length]
            avg_max_loss = np.mean(max_losses)
            std_max_loss = np.std(max_losses) if len(max_losses) > 1 else 0
            max_stats = f", Max Loss: {avg_max_loss:.4f}, Max Std: {std_max_loss:.4f}"
            line = f"Length: {length}, Average Loss: {avg_loss:.4f}, Std: {std_loss:.4f}{max_stats}, Samples: {len(losses)}"
            f.write(line + "\n")
            print(line)
    print("Statistics saved successfully!")


def calc_loss(
    reconstructed: torch.Tensor, original: torch.Tensor, sequences
) -> torch.Tensor:
    """Calculate the loss between the reconstructed and original distance maps.

    Args:
        reconstructed: Reconstructed distance maps from the VAE
        original: Original distance maps
    Returns:
        Calculated loss value
    """
    # Calculate MSE loss
    mse_loss = torch.nn.functional.mse_loss(reconstructed, original, reduction="none")

    # Create a mask to ignore zero values in the original distance maps
    mask = (original != 0).float()

    # Remove the lower triangle of the mask so that loss is only calculated on the upper triangle of the distance map
    mask = mask - mask.tril()

    # Apply the mask to the MSE loss
    masked_loss = mse_loss * mask

    mean_loss = masked_loss.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))

    max_loss = torch.amax(masked_loss, dim=(1, 2, 3), keepdim=False)

    # Return the mean loss over the non-zero elements
    return mean_loss, max_loss


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
    sequences = torch.tensor(np.stack(sequences), dtype=torch.float32)

    return data_batch, sequences, sample_keys


def main():
    """Main function to process distance maps with a trained VAE model."""

    model = load_model(MODEL)
    device = next(model.parameters()).device  # Get device from model

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
        shuffle=False,
    )

    loss_by_length = defaultdict(list)
    max_loss_by_length = defaultdict(list)

    with torch.no_grad():
        for num, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            distance_maps, sequences, batch_keys = batch

            # Move data to appropriate device
            distance_maps = distance_maps.to(device)

            # Process the batch through the VAE model to get latent representations
            reconstructed, _ = model(distance_maps)

            mean_losses, max_losses = calc_loss(reconstructed, distance_maps, sequences)

            for mean_loss, max_loss, seq in zip(mean_losses, max_losses, sequences):
                length = (seq != 0).sum().item()
                loss_by_length[length].append(mean_loss.item())
                max_loss_by_length[length].append(max_loss.item())

            if num % 100 == 0:
                # Save the loss statistics at the end of normal execution
                save_statistics(loss_by_length, max_loss_by_length)


if __name__ == "__main__":
    main()
