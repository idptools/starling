import pytest
import torch

from starling.inference.generation import sequence_encoder_backend
from starling.data.tokenizer import StarlingTokenizer


class _DummyDiffusion:
    def __init__(self, emb_dim=4):
        self.emb_dim = emb_dim

    def sequence2labels(self, sequences, sequence_mask, ionic_strength):  # noqa: D401
        """Return predictable embeddings with shape (B, L, D)."""
        tokens = sequences.float()
        context = (tokens * sequence_mask).sum(1) / sequence_mask.sum(1)
        values = (tokens + context[:, None] + ionic_strength) * sequence_mask
        return values[..., None].expand(-1, -1, self.emb_dim)


class _DummyModelManager:
    def __init__(self, emb_dim=4):
        self.diffusion = _DummyDiffusion(emb_dim=emb_dim)

    def get_models(self, device, encoder_path=None, ddpm_path=None):  # noqa: D401
        # Returns (encoder_model, diffusion). We only need diffusion.
        return None, self.diffusion


@pytest.fixture
def dummy_manager():
    return _DummyModelManager(emb_dim=7)


def _expected_embedding(tokens, salt=150):
    tokens = torch.tensor(tokens, dtype=torch.float32)
    return (tokens + tokens.mean() + salt)[:, None].expand(-1, 7)


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("bucket", [False, True])
def test_sequence_encoder_backend_basic(dummy_manager, aggregate, bucket):
    sequences = {
        "seq1": "ACDE",  # len 4
        "seq2": "AC",  # len 2
        "seq3": "ACDEF",  # len 5 (longest)
        "seq4": "A",  # len 1
        "same_length": "KLMN",
    }
    out = sequence_encoder_backend(
        sequence_dict=sequences,
        device="cpu",
        batch_size=2,
        ionic_strength=300,
        output_directory=None,
        model_manager=dummy_manager,
        aggregate=aggregate,
        bucket=bucket,
        bucket_size=2,
    )
    # Should return dict with same keys
    assert set(out.keys()) == set(sequences.keys())
    # Each embedding trimmed to sequence length, last dim = emb_dim
    for name, seq in sequences.items():
        expected = _expected_embedding(StarlingTokenizer().encode(seq), salt=300)
        if aggregate:
            expected = expected.mean(0)
        torch.testing.assert_close(out[name], expected)


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
        aggregate=False,
    )
    assert set(out.keys()) == set(sequences.keys())
    for k, v in out.items():
        expected = _expected_embedding(StarlingTokenizer().encode(sequences[k]))
        torch.testing.assert_close(v, expected)


def test_sequence_encoder_backend_saves_files(tmp_path, dummy_manager):
    sequences = {"a": "ACD", "b": "AC", "c": "A"}
    out = sequence_encoder_backend(
        sequence_dict=sequences,
        device="cpu",
        batch_size=2,
        ionic_strength=150,
        output_directory=str(tmp_path),
        model_manager=dummy_manager,
        aggregate=False,
    )
    # When output_directory provided, function returns None
    assert out is None
    # Files created for each sequence
    for name, seq in sequences.items():
        fpath = tmp_path / f"{name}.pt"
        assert fpath.exists()
        tensor = torch.load(fpath)
        torch.testing.assert_close(tensor, _expected_embedding(StarlingTokenizer().encode(seq)))


def test_sequence_encoder_backend_pretokenized(dummy_manager):
    # Create pretokenized integer lists (simulate external tokenization)
    sequences = {
        "x": [1, 2, 3, 4],
        "y": [20, 19, 18, 17],
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
        aggregate=False,
    )
    assert set(out.keys()) == set(sequences.keys())
    for name, toks in sequences.items():
        torch.testing.assert_close(out[name], _expected_embedding(toks))
