"""Toy datasets for the from-scratch Core.

CopyTask is the workhorse: a synthetic copy task with a tiny fixed alphabet that the
tiny model learns to near-zero loss on CPU in seconds — an unambiguous learning signal
(copy accuracy -> 100%). ``load_tinyshakespeare`` is a secondary, opt-in path for
eyeballing real-text generation; it is never used by the test suite (no network in CI).
"""

from __future__ import annotations

import os
import urllib.request

import torch
from torch import Tensor

from embraos_qnm.data.tokenizer import CharTokenizer

_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class CopyTask:
    """``<random symbols> SEP <same symbols>`` — the model must reproduce the source.

    Loss is scored only on the copied (post-separator) positions via ``ignore_index=-1``.
    """

    SEP = ">"

    def __init__(self, n_symbols: int = 8, length: int = 6):
        if not 1 <= n_symbols <= 26:
            raise ValueError("n_symbols must be in [1, 26]")
        alphabet = [chr(ord("a") + i) for i in range(n_symbols)]
        self.length = length
        self.tokenizer = CharTokenizer(alphabet + [self.SEP])
        self.symbol_ids = torch.tensor([self.tokenizer.stoi[c] for c in alphabet])
        self.sep_id = self.tokenizer.stoi[self.SEP]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def seq_len(self) -> int:
        """Full sequence length: ``length`` symbols + separator + ``length`` symbols."""
        return 2 * self.length + 1

    def batch(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(x, y)`` for next-token training, each ``(batch_size, 2*length)``.

        ``y`` is ``-1`` on the source/separator positions so loss is scored only on the
        copied tail. Invariant: ``y[:, length:] == x[:, :length]`` (the copy = the source).
        """
        length = self.length
        choices = torch.randint(0, len(self.symbol_ids), (batch_size, length), generator=generator)
        src = self.symbol_ids[choices]
        sep = torch.full((batch_size, 1), self.sep_id)
        seq = torch.cat([src, sep, src], dim=1)  # (B, 2*length + 1)
        x = seq[:, :-1]
        y = seq[:, 1:].clone()
        y[:, :length] = -1  # ignore source + separator; score only the copied tail
        return x, y


def load_tinyshakespeare(path: str | None = None) -> str:
    """Opt-in: return the tinyshakespeare text, caching to ``path`` if given and missing.

    NOT used by the test suite — this is for interactive ``generate.py`` runs only.
    """
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    with urllib.request.urlopen(_SHAKESPEARE_URL) as resp:
        text = resp.read().decode("utf-8")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
