"""NoOpFabric — the honest null for the GNN Fabric (IDENTITY)."""

from __future__ import annotations

import torch
from torch import Tensor

from embraos_qnm.interfaces import FabricInterface


class NoOpFabric(FabricInterface):
    """Returns a zero modulation: the Fabric contributes nothing yet.

    The IDENTITY component is not modeled in this scaffold. With this default in
    place, the QNM's forward is identical to the plain Core (tests/test_bit_identity).
    """

    def forward(self, h: Tensor) -> Tensor:
        return torch.zeros_like(h)
