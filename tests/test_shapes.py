"""Shape/contract tests for the Core and the no-op components."""

from dataclasses import replace

import pytest
import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.transformer import TinyTransformer
from embraos_qnm.fabric import NoOpFabric
from embraos_qnm.world_state import NoOpWorldState

_BASE = QNMConfig(vocab_size=11, block_size=16, n_layer=3, n_head=2, d_model=16, inject_layer=1)


def test_logits_shape():
    cfg = _BASE
    model = TinyTransformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 8))
    logits = model(idx)
    assert logits.shape == (4, 8, cfg.vocab_size)


def test_embed_shape():
    model = TinyTransformer(_BASE)
    idx = torch.randint(0, _BASE.vocab_size, (2, 5))
    assert model.embed(idx).shape == (2, 5, _BASE.d_model)


def test_num_layers_matches_config():
    model = TinyTransformer(replace(_BASE, n_layer=3))
    assert model.num_layers() == 3


def test_sequence_longer_than_block_size_raises():
    cfg = replace(_BASE, block_size=8)
    model = TinyTransformer(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 9))
    with pytest.raises(ValueError):
        model(idx)


def test_noop_components_preserve_shape():
    h = torch.randn(2, 5, 16)
    assert NoOpFabric()(h).shape == h.shape
    ws = NoOpWorldState()
    c = NoOpFabric().surface(h)  # the surface a no-op Fabric induces: zeros (B, T)
    delta, _ = ws(h, ws.init_state(h.size(0), h.device), c)  # carried-state contract
    assert delta.shape == h.shape
