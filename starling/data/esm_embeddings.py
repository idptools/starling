from typing import List

import torch
from IPython import embed


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(
        self, alphabet, truncation_seq_length: int = None, device: str = "cpu"
    ):
        self.alphabet = alphabet
        self.device = device
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [
                seq_str[: self.truncation_seq_length] for seq_str in seq_encoded_list
            ]
        # The only change from the original code is below (max_length = 384)
        max_len = 384
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
            device=self.device,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64, device=self.device)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = (
                    self.alphabet.eos_idx
                )
        return labels, strs, tokens


@torch.inference_mode()
def esm_embeddings(
    model, batch_converter, sequences: List, device: str, layer: int, type="residue"
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
    sequences = list(
        map(lambda num_seq: (f"protein_{num_seq[0]}", num_seq[1]), enumerate(sequences))
    )
    # batch_converter = alphabet.get_batch_converter()
    batch_converter = BatchConverter(model.alphabet, device=device)
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)

    with torch.no_grad():
        token_representations = model(batch_tokens, repr_layers=[layer])[
            "representations"
        ][layer]

    # Extract representations for valid tokens (excluding <cls> and <eos>)
    token_representations = token_representations[:, 1:-1]

    # Calculate mean along the token dimension
    if type == "sequence":
        token_representations = token_representations.mean(dim=1)

    token_representations = token_representations

    return token_representations


# @torch.inference_mode()
# def esm_embeddings(
#     model, batch_tokens: List, device: str, layer: int, type="residue"
# ) -> torch.Tensor:
#     """
#     Extracts ESM embeddings for a list of sequences.

#     Parameters
#     ----------
#     model : ESM model
#         Initiated ESM model
#     alphabet : ESM alphabet
#         Initiated ESM alphabet
#     sequences : List
#         List of protein sequences
#     device : str
#         Device to run the model on
#     layer : int
#         Layer to extract embeddings from

#     Returns
#     -------
#     torch.Tensor
#         Extracted embeddings for each sequences in the provided list.
#     """
#     with torch.no_grad():
#         token_representations = model(batch_tokens, repr_layers=[layer])[
#             "representations"
#         ][layer]

#     # Extract representations for valid tokens (excluding <cls> and <eos>)
#     token_representations = token_representations[:, 1:-1]

#     # Calculate mean along the token dimension
#     if type == "sequence":
#         token_representations = token_representations.mean(dim=1)

#     token_representations = token_representations

#     return token_representations
