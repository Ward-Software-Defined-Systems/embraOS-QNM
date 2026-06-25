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

from typing import Any

import torch
from torch import nn

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

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Arg-transparent: pass everything to the wrapped layer and operate on its
        # hidden-state output. One seam wraps both a TinyTransformer Block (single tensor
        # in/out) and an HF decoder layer (hidden_states + RoPE/mask kwargs, tuple out)
        # without threading the layer's auxiliary args by hand.
        out = self.block(*args, **kwargs)
        if not self.enabled:
            return out
        is_tuple = isinstance(out, tuple)
        h_base = out[0] if is_tuple else out
        delta_fabric = self.fabric(h_base)
        # The World-State carries a ψ-state across the token axis. Initialized per forward
        # here; cross-decode-step persistence is deferred until ψ is real (PSI §2). The
        # returned register is discarded for now — with gate_world == 0 (and the no-op's
        # zeros) the contribution is exactly zero, so bit-identity is untouched.
        psi = self.world_state.init_state(h_base.size(0), h_base.device)
        delta_world, _ = self.world_state(h_base, psi)
        new_h = h_base + self.gate_fabric * delta_fabric + self.gate_world * delta_world
        return (new_h, *out[1:]) if is_tuple else new_h
