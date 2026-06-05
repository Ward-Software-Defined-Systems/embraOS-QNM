"""QNMBlock — the injection seam.

Wraps exactly one Core block. After the block runs, its output is routed through the
Fabric (IDENTITY) and World-State (SOUL) and recombined via zero-initialized ReZero
gates. Two independent reasons this is bit-identical to the bare block:

  1. Structural (``enabled=False``): the early return takes the SAME op path as the
     bare block — no added term to round. Immune to float corner cases.
  2. Cold-start (gates zero-initialized): ``h + g_f * Δ_f + g_w * Δ_w`` with
     ``g = 0`` is ``h + 0 + 0 = h`` exactly in IEEE-754, even when Δ is non-zero. The
     model only departs from baseline as training drives the gates off zero.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from embraos_qnm.interfaces import FabricInterface, WorldStateInterface


class QNMBlock(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        fabric: FabricInterface,
        world_state: WorldStateInterface,
        *,
        enabled: bool = True,
    ):
        super().__init__()
        self.block = block
        self.fabric = fabric
        self.world_state = world_state
        self.enabled = enabled
        # ReZero scalar gates, zero-initialized.
        self.gate_fabric = nn.Parameter(torch.zeros(()))
        self.gate_world = nn.Parameter(torch.zeros(()))

    def forward(self, h: Tensor) -> Tensor:
        h_base = self.block(h)
        if not self.enabled:
            return h_base
        delta_fabric = self.fabric(h_base)
        delta_world = self.world_state(h_base)
        return h_base + self.gate_fabric * delta_fabric + self.gate_world * delta_world
