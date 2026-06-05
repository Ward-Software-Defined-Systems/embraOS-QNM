"""Shared pytest fixtures."""

import pytest

from embraos_qnm.config import QNMConfig


@pytest.fixture
def cfg() -> QNMConfig:
    """A small-but-real config: a few layers, injection in the middle, head | d_model."""
    return QNMConfig(vocab_size=11, block_size=16, n_layer=4, n_head=2, d_model=16, inject_layer=2)
