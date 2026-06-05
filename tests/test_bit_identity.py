"""The headline guarantee (H₀ wired into CI).

With no-op components, the QNM is bit-identical to the plain transformer. Asserted
with ``torch.equal`` (exact), never ``allclose`` — a tolerance would be an escape
hatch, and the no-op is an algebraic identity, not an approximation. All on CPU,
where the arithmetic is deterministic.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.block import Block
from embraos_qnm.core.transformer import TinyTransformer
from embraos_qnm.interfaces import FabricInterface, WorldStateInterface
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.manifold.qnm_block import QNMBlock
from embraos_qnm.seed import set_seed


class _EchoFabric(FabricInterface):
    """A LIVE, feature-varying delta: returns its input.

    NB: it must vary across the feature dimension. A uniform-constant delta (e.g.
    ones) sits in the null space of the downstream LayerNorms and the final ln_f, so
    it is invisible to both the output and its gradient — a real constraint on any
    future Fabric/World-State modulation.
    """

    def forward(self, h: Tensor) -> Tensor:
        return h


class _EchoWorldState(WorldStateInterface):
    def forward(self, h: Tensor) -> Tensor:
        return h


def _input(cfg: QNMConfig) -> Tensor:
    return torch.randint(0, cfg.vocab_size, (3, 8))


def test_disabled_qnm_equals_plain_transformer(cfg: QNMConfig):
    """Structural no-op (qnm_enabled=False): the SAME op path as the bare block."""
    set_seed(0)
    plain = TinyTransformer(cfg)
    set_seed(0)
    qnm = QNMModel(replace(cfg, qnm_enabled=False))
    idx = _input(cfg)
    assert torch.equal(plain(idx), qnm(idx)[0])


def test_noop_qnm_equals_plain_transformer(cfg: QNMConfig):
    """enabled=True with default no-op components: exercises the ``h + g*0`` arithmetic."""
    set_seed(0)
    plain = TinyTransformer(cfg)
    set_seed(0)
    qnm = QNMModel(cfg)
    idx = _input(cfg)
    assert torch.equal(plain(idx), qnm(idx)[0])


def test_zero_init_gates_give_identity_with_live_deltas(cfg: QNMConfig):
    """LIVE deltas (echo), gates at zero init: still identity (the cold-start guarantee)."""
    set_seed(0)
    plain = TinyTransformer(cfg)
    set_seed(0)
    qnm = QNMModel(cfg, fabric=_EchoFabric(), world_state=_EchoWorldState())
    idx = _input(cfg)
    assert torch.equal(plain(idx), qnm(idx)[0])


def test_only_inject_layer_is_wrapped(cfg: QNMConfig):
    qnm = QNMModel(cfg)
    for i, block in enumerate(qnm.core.blocks):
        if i == cfg.inject_layer:
            assert isinstance(block, QNMBlock)
        else:
            assert isinstance(block, Block)
