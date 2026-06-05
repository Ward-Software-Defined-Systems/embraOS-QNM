"""Tokenizer round-trip + CopyTask structural invariants."""

import torch

from embraos_qnm.data.tokenizer import CharTokenizer
from embraos_qnm.data.toy import CopyTask
from embraos_qnm.seed import set_seed


def test_encode_decode_roundtrip():
    text = "the quick brown fox jumps"
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size_matches_unique_chars():
    tok = CharTokenizer("aabbbc")
    assert tok.vocab_size == 3  # a, b, c


def test_copytask_shapes_and_vocab():
    task = CopyTask(n_symbols=8, length=5)
    set_seed(0)
    x, y = task.batch(4)
    assert task.vocab_size == 9  # 8 symbols + separator
    assert x.shape == (4, 2 * task.length)
    assert y.shape == (4, 2 * task.length)


def test_copytask_targets_mask_the_prompt():
    task = CopyTask(n_symbols=8, length=5)
    set_seed(0)
    _, y = task.batch(2)
    length = task.length
    assert (y[:, :length] == -1).all()  # source + separator ignored
    assert (y[:, length:] != -1).all()  # copied tail scored


def test_copytask_copy_region_equals_source():
    task = CopyTask(n_symbols=8, length=5)
    set_seed(0)
    x, y = task.batch(3)
    length = task.length
    # The scored tail is exactly the source: a learnable, verifiable objective.
    assert torch.equal(y[:, length:], x[:, :length])
