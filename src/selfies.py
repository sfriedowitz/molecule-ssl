from typing import Iterable
import selfies as sf
import torch


class SelfiesEncoder:
    def __init__(self, alphabet, pad_to_len):
        self.alphabet = alphabet
        self.pad_to_len = pad_to_len
        self.symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
        self.idx_to_symbol = {i: s for s, i in self.symbol_to_idx.items()}

    @classmethod
    def build(cls, selfies: Iterable[str]) -> "SelfiesEncoder":
        alphabet = sf.get_alphabet_from_selfies(selfies)
        alphabet.add("[nop]")
        alphabet = list(sorted(alphabet))
        pad_to_len = max(sf.len_selfies(s) for s in selfies)
        return cls(alphabet, pad_to_len)

    def encode_label(self, selfie: str) -> list[int]:
        return sf.selfies_to_encoding(
            selfies=selfie,
            vocab_stoi=self.symbol_to_idx,
            pad_to_len=self.pad_to_len,
            enc_type="label",
        )

    def encode_one_hot(self, selfie: str) -> list[int]:
        return sf.selfies_to_encoding(
            selfies=selfie,
            vocab_stoi=self.symbol_to_idx,
            pad_to_len=self.pad_to_len,
            enc_type="one_hot",
        )

    def decode_label(self, encoding: list[int]) -> str:
        return sf.encoding_to_selfies(
            encoding=encoding,
            vocab_itos=self.idx_to_symbol,
            enc_type="label",
        )

    def decode_one_hot(self, encoding: list[list[int]]) -> str:
        return sf.encoding_to_selfies(
            encoding=encoding,
            vocab_itos=self.idx_to_symbol,
            enc_type="one_hot",
        )

    def decode_tensor(self, x: torch.Tensor) -> list[str]:
        labels = x.argmax(dim=-1).tolist()
        return [self.decode_label(l) for l in labels]
