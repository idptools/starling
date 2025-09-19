import pytest

from starling.data.tokenizer import StarlingTokenizer


def test_tokenizer_roundtrip_simple():
    tok = StarlingTokenizer()
    seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard (as defined in mapping)
    encoded = tok.encode(seq)
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
