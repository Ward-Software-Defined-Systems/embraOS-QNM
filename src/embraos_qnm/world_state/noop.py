"""NoOpWorldState — the honest H₀ null for the World-State (SOUL).

PROJECT DISCIPLINE — read before touching this file:

``P_ψ`` here is the IDENTITY map, so the modulation is exactly zero. ψ — the invariant
the World-State is meant to project onto — is DELIBERATELY UNDEFINED. This no-op is the
honest null, NOT a placeholder to be quietly fleshed out with a plausible-looking
constraint surface.

The moment this returns anything other than zeros for an undefined ψ, the bit-identity
guarantee (no-op QNM == plain transformer) stops meaning what it claims, and the
"every architectural effect is a provable delta from a null" property is lost. Keep
this a literal ``zeros_like`` until a ψ exists that survives the replica test
(see EPOCH-INVARIANT-GROUNDING.md). The scaffold is the body; ψ is whether there is
anyone home — and it is intentionally left empty here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from embraos_qnm.interfaces import PsiState, WorldStateInterface


class NoOpWorldState(WorldStateInterface):
    """Returns a zero modulation: ``P_ψ`` = identity, the honest null for ψ.

    Valid null under the carried-state contract: an empty (``None``) register and a
    pass-through, so it stays a literal ``zeros_like`` while ψ is undefined.
    """

    def init_state(self, batch_size: int, device: torch.device) -> PsiState:
        return None

    def forward(self, h: Tensor, psi: PsiState) -> tuple[Tensor, PsiState]:
        return torch.zeros_like(h), psi
