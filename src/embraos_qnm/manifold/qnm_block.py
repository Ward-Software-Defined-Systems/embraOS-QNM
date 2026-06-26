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
        # Transient ψ-carry slots for the cached Arm-A decode (eval/arms.greedy_generate_psi):
        # the decode loop sets psi_in to seed the latch from the prior step and reads psi_out (the
        # advanced register) back after each forward. NOT registered buffers/params — they must stay
        # out of state_dict and never affect bit-identity. psi_in is None outside an active ψ-decode,
        # which is exactly today's behavior (init_state ⇒ zeros over the full sequence).
        self.psi_in: torch.Tensor | None = None
        self.psi_out: torch.Tensor | None = None

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
        # Fabric-supplied 𝒞 (PSI §5): the Fabric emits the constraint-surface signal c_t, the
        # World-State latches on it and emits the P_ψ correction. The latch is a causal recurrence
        # over the trajectory; on a full-sequence forward it accumulates over the sequence axis
        # (psi_in is None ⇒ init_state zeros — byte-for-byte today's behavior). Under the cached
        # Arm-A decode the loop seeds psi_in with the prior step's register and reads psi_out back,
        # so the latch persists across decode steps instead of resetting per token (see
        # eval/arms.greedy_generate_psi).
        c = self.fabric.surface(h_base)
        psi = (
            self.world_state.init_state(h_base.size(0), h_base.device)
            if self.psi_in is None
            else self.psi_in
        )
        delta_world, psi_next = self.world_state(h_base, psi, c)
        self.psi_out = psi_next
        new_h = h_base + self.gate_fabric * delta_fabric + self.gate_world * delta_world
        return (new_h, *out[1:]) if is_tuple else new_h
