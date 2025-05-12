class StarlingTokenizer:
    """
    A tokenizer for the Starling model.
    """

    def __init__(self):
        self.aa_to_int = {
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

        self.int_to_aa = {v: k for k, v in self.aa_to_int.items()}

    def encode(self, sequence):
        """
        Encode a string of amino acids into a sequence of integers.
        """
        return [self.aa_to_int[i] for i in sequence]

    def decode(self, sequence):
        """
        Decode a sequence of integers into a string of amino acids.
        """
        return "".join([self.int_to_aa[i] for i in sequence if i != 0])
