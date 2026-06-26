"""Gate behavior: the null is null by construction, not because the path is dead."""

from __future__ import annotations

import torch
from torch import Tensor

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.transformer import TinyTransformer
from embraos_qnm.interfaces import FabricInterface, PsiState, WorldStateInterface
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.seed import set_seed


class _EchoFabric(FabricInterface):
    """A LIVE, feature-varying delta (returns its input). See test_bit_identity for
    why a uniform-constant delta would be invisible to gradients (LayerNorm null space)."""

    def forward(self, h: Tensor) -> Tensor:
        return h


class _EchoWorldState(WorldStateInterface):
    """LIVE delta under the carried-state contract: echoes h, carries an empty register."""

    def init_state(self, batch_size: int, device: torch.device) -> PsiState:
        return None

    def forward(self, h: Tensor, psi: PsiState, c: Tensor) -> tuple[Tensor, PsiState]:
        return h, psi


def test_gates_zero_initialized(cfg: QNMConfig):
    block = QNMModel(cfg).qnm_block
    assert block.gate_fabric.item() == 0.0
    assert block.gate_world.item() == 0.0


def test_gates_receive_gradient(cfg: QNMConfig):
    """With live deltas, the gates get non-zero gradient — proof they are in the graph."""
    set_seed(0)
    qnm = QNMModel(cfg, fabric=_EchoFabric(), world_state=_EchoWorldState())
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = qnm(idx, targets)
    assert loss is not None
    loss.backward()

    block = qnm.qnm_block
    g_fabric = block.gate_fabric.grad
    g_world = block.gate_world.grad
    assert g_fabric is not None and g_world is not None
    assert g_fabric.abs().item() > 0.0
    assert g_world.abs().item() > 0.0


def test_nonzero_gate_changes_output(cfg: QNMConfig):
    """A manually opened gate changes the output — the seam CAN do something."""
    set_seed(0)
    plain = TinyTransformer(cfg)
    set_seed(0)
    qnm = QNMModel(cfg, fabric=_EchoFabric(), world_state=_EchoWorldState())
    with torch.no_grad():
        qnm.qnm_block.gate_fabric.fill_(1.0)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    assert not torch.equal(plain(idx), qnm(idx)[0])
