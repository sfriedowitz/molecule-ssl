import numpy as np


DEFAULT_ALPHABET = [
    " ",
    "#",
    "(",
    ")",
    "+",
    "-",
    "/",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "=",
    "@",
    "B",
    "C",
    "F",
    "H",
    "I",
    "N",
    "O",
    "P",
    "S",
    "[",
    "\\",
    "]",
    "c",
    "l",
    "n",
    "o",
    "r",
    "s",
]


class SmilesEncoder(object):
    def __init__(self, alphabet: list[str] = DEFAULT_ALPHABET, padding: int = 120):
        self.encoding = {c: i for i, c in enumerate(alphabet)}
        self.decoding = {i: c for c, i in self.encoding.items()}

        self.padding_size = padding
        self.encoding_size = len(alphabet)

    def encode_integer(self, smile: str):
        smile = smile.ljust(self.padding_size)
        return [self.encoding[char] for char in smile]

    def encode_one_hot(self, smile: str):
        integer_encoding = self.encode_integer(smile)

        one_hot = np.zeros((len(integer_encoding), self.encoding_size))
        for idx, value in enumerate(integer_encoding):
            one_hot[idx] = np.eye(self.encoding_size)[value]

        return one_hot

    def encode_dataset(self, smiles: list[str]):
        return np.array([self.encode_one_hot(x) for x in smiles])

    def decode_one_hot(self, z) -> str:
        indices = np.argmax(z, axis=1)
        return "".join([self.decoding[x] for x in indices]).strip()
