class StarlingTokenizer:
    """
    Tokenizer for Starling model - uses a byte-level translation table for speed.
    """

    # Public vocab maps (kept for compatibility with existing code)
    aa_to_int = {
        "0": 0,
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
    }
    int_to_aa = {v: k for k, v in aa_to_int.items()}

    def __init__(self):
        # Build encode translation table (byte -> id). 255 = sentinel for "unknown".
        table = bytearray([255] * 256)
        for ch, idx in self.aa_to_int.items():
            table[ord(ch)] = idx
        self._enc_table = bytes(table)

        # Build decode translation table (id -> ASCII byte), unknown -> '?'
        inv = bytearray([ord("?")] * 256)
        for idx, ch in self.int_to_aa.items():
            inv[idx] = ord(ch)
        self._dec_table = bytes(inv)

    def encode(self, sequence: str):
        """
        Encode a string of amino acids into a list[int].
        Raises ValueError if an unknown character is present.
        """
        b = sequence.encode("ascii")
        # bytes of ids (0..255; 255 means unknown)
        enc = b.translate(self._enc_table)
        if 255 in enc:
            for i, bt in enumerate(b):
                if self._enc_table[bt] == 255:
                    raise KeyError(f"Unknown token '{chr(bt)}' at position {i}")
        # Convert to Python ints
        return list(enc)

    def decode(self, sequence):
        """
        Decode a sequence of ints into a string, dropping zeros (padding).
        Accepts list/tuple[int], bytes, or bytearray.
        """
        # Normalize to raw bytes of ids
        if isinstance(sequence, (bytes, bytearray)):
            buf = sequence
        else:
            buf = bytes(int(x) & 0xFF for x in sequence)

        # Drop 0 tokens before reverse translate
        filtered = bytes(x for x in buf if x != 0)
        ascii_bytes = filtered.translate(self._dec_table)  # -> bytes of alphabet

        return ascii_bytes.decode("ascii").replace("?", "")
