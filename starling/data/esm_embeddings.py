from typing import List

import esm
import torch
from IPython import embed


def esm_embeddings(
    model, alphabet, sequences: List, device: str, layer: int
) -> torch.Tensor:
    """
    Extracts ESM embeddings for a list of sequences.

    Parameters
    ----------
    model : ESM model
        Initiated ESM model
    alphabet : ESM alphabet
        Initiated ESM alphabet
    sequences : List
        List of protein sequences
    device : str
        Device to run the model on
    layer : int
        Layer to extract embeddings from

    Returns
    -------
    torch.Tensor
        Extracted embeddings for each sequences in the provided list.
    """
    sequences = [(f"protein_{num}", seq) for num, seq in enumerate(sequences)]

    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        token_representations = model(batch_tokens, repr_layers=[layer])[
            "representations"
        ][layer]

    # Extract representations for valid tokens (excluding <cls> and <eos>)
    token_representations = token_representations[:, 1:-1]

    # Calculate mean along the token dimension
    token_representations = token_representations.mean(dim=1)

    token_representations = token_representations.to("cuda")

    return token_representations
