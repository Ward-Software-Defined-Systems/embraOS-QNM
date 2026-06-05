"""A minimal character-level tokenizer.

Char-level keeps the from-scratch story honest and dependency-free (no BPE /
sentencepiece). The vocab is the sorted set of characters it is built from.
"""

from __future__ import annotations

from collections.abc import Iterable


class CharTokenizer:
    def __init__(self, chars: Iterable[str]):
        self.itos: list[str] = sorted(set(chars))
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)
