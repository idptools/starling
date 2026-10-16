import pytest

from starling.data.tokenizer import StarlingTokenizer


def test_tokenizer_roundtrip_simple():
    tok = StarlingTokenizer()
    seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard (as defined in mapping)
    encoded = tok.encode(seq)
    # Vocabulary IDs are part of pretrained-checkpoint compatibility.
    assert encoded == list(range(1, 21))
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    assert len(encoded) == len(seq)
    decoded = tok.decode(encoded)
    # decode drops zeros only; none here
    assert decoded == seq


def test_tokenizer_decode_drops_zero():
    tok = StarlingTokenizer()
    seq = "ACD"
    encoded = tok.encode(seq)
    # prepend a padding/zero token manually to simulate padded input
    padded = [0] + encoded + [0]
    decoded = tok.decode(padded)
    assert decoded == seq  # zeros stripped


def test_tokenizer_invalid_character_raises():
    tok = StarlingTokenizer()
    with pytest.raises(KeyError):
        tok.encode("ACZ")  # 'Z' not in mapping


def test_tokenizer_alphabet_consistency():
    tok = StarlingTokenizer()
    # Ensure int_to_aa is exact inverse (excluding potential padding '0')
    for aa, idx in tok.aa_to_int.items():
        assert tok.int_to_aa[idx] == aa
    # Ensure no duplicate integer codes
    assert len(set(tok.aa_to_int.values())) == len(tok.aa_to_int)


def test_tokenizer_empty_sequence():
    tok = StarlingTokenizer()
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_tokenizer_decode_bytes_input():
    tok = StarlingTokenizer()
    seq = "ACD"
    encoded = tok.encode(seq)
    # Convert to bytes to exercise bytes path
    b = bytes(encoded)
    assert tok.decode(b) == seq


def test_tokenizer_lowercase_raises():
    tok = StarlingTokenizer()
    with pytest.raises(KeyError):
        tok.encode("acd")  # lowercase not in vocab


def test_tokenizer_all_zero_decodes_empty():
    tok = StarlingTokenizer()
    assert tok.decode([0, 0, 0]) == ""


def test_tokenizer_rejects_padding_character():
    """'0' is the padding token, not a residue; encode used to turn 'A0C' into [1, 0, 2]."""
    tok = StarlingTokenizer()
    with pytest.raises(KeyError, match="'0' at position 1"):
        tok.encode("A0C")


def test_tokenizer_non_ascii_raises_key_error():
    """Non-ASCII characters raise the same KeyError as other unknown characters (not UnicodeEncodeError)."""
    tok = StarlingTokenizer()
    with pytest.raises(KeyError, match="'Å' at position 2"):
        tok.encode("ACÅD")


def test_tokenizer_decode_still_drops_padding_after_encode_rejects_it():
    """Rejecting '0' in encode does not change how decode treats padding tokens."""
    tok = StarlingTokenizer()
    assert tok.int_to_aa[0] == "0"
    assert tok.decode([1, 0, 2]) == "AC"
