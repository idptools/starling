import pytest
import torch

from starling.inference.generation import sequence_encoder_backend


class _DummyDiffusion:
    def __init__(self, emb_dim=4):
        self.emb_dim = emb_dim

    def sequence2labels(self, sequences, sequence_mask, ionic_strength):  # noqa: D401
        """Return predictable embeddings with shape (B, L, D)."""
        # sequences: (B, Lmax)
        B, Lmax = sequences.shape
        # Create position embeddings: value = position index
        pos = (
            torch.arange(Lmax, device=sequences.device)
            .view(1, Lmax, 1)
            .repeat(B, 1, self.emb_dim)
        )
        return pos.float()


class _DummyModelManager:
    def __init__(self, emb_dim=4):
        self.diffusion = _DummyDiffusion(emb_dim=emb_dim)

    def get_models(self, device, encoder_path=None, ddpm_path=None):  # noqa: D401
        # Returns (encoder_model, diffusion). We only need diffusion.
        return None, self.diffusion


@pytest.fixture
def dummy_manager():
    return _DummyModelManager(emb_dim=7)


def test_sequence_encoder_backend_basic(dummy_manager):
    sequences = {
        "seq1": "ACDE",  # len 4
        "seq2": "AC",  # len 2
        "seq3": "ACDEF",  # len 5 (longest)
        "seq4": "A",  # len 1
    }
    out = sequence_encoder_backend(
        sequence_dict=sequences,
        device="cpu",
        batch_size=2,
        ionic_strength=150,
        output_directory=None,
        model_manager=dummy_manager,
    )
    # Should return dict with same keys
    assert set(out.keys()) == set(sequences.keys())
    # Each embedding trimmed to sequence length, last dim = emb_dim
    for name, seq in sequences.items():
        emb = out[name]
        assert emb.shape == (len(seq), 7)
        # Check that positions are increasing along first axis for dim 0
        if len(seq) > 1:
            # emb is a torch.Tensor; position encoding placed in all dims
            assert torch.equal(emb[:, 0], torch.arange(len(seq)))


def test_sequence_encoder_backend_remainder_batch(dummy_manager):
    # 5 sequences with batch_size=2 -> 2 full batches + 1 remainder
    sequences = {f"s{i}": "A" * (i + 1) for i in range(5)}  # lengths 1..5
    out = sequence_encoder_backend(
        sequence_dict=sequences,
        device="cpu",
        batch_size=2,
        ionic_strength=150,
        output_directory=None,
        model_manager=dummy_manager,
    )
    assert set(out.keys()) == set(sequences.keys())
    for k, v in out.items():
        assert v.shape == (len(sequences[k]), 7)


def test_sequence_encoder_backend_saves_files(tmp_path, dummy_manager):
    sequences = {"a": "ACD", "b": "AC", "c": "A"}
    out = sequence_encoder_backend(
        sequence_dict=sequences,
        device="cpu",
        batch_size=2,
        ionic_strength=150,
        output_directory=str(tmp_path),
        model_manager=dummy_manager,
    )
    # When output_directory provided, function returns None
    assert out is None
    # Files created for each sequence
    for name, seq in sequences.items():
        fpath = tmp_path / f"{name}.pt"
        assert fpath.exists()
        tensor = torch.load(fpath)
        assert tensor.shape == (len(seq), 7)


def test_sequence_encoder_backend_pretokenized(dummy_manager):
    # Create pretokenized integer lists (simulate external tokenization)
    sequences = {
        "x": [1, 2, 3, 4],
        "y": [1, 2],
        "z": [1],
    }
    out = sequence_encoder_backend(
        sequence_dict=sequences,
        device="cpu",
        batch_size=2,
        ionic_strength=150,
        output_directory=None,
        model_manager=dummy_manager,
        pretokenized=True,
    )
    assert set(out.keys()) == set(sequences.keys())
    for name, toks in sequences.items():
        emb = out[name]
        assert emb.shape[0] == len(toks)
