"""CandidateWorldState — ψ₀, the first invariant put on trial (PSI-OPERATIONAL-GROUNDING.md §5).

ψ₀ is a *causal cumulative violation latch*. Given a per-step constraint signal
``c_t = g(h_t)`` (where ``c_t > τ`` means the trajectory has left the constraint surface 𝒞 at
step t):

    m_t = max(m_{t-1}, relu(c_t − τ))        # monotone cummax over the causal prefix
    ψ holds at t   ⟺   m_t == 0               # never crossed the boundary up to t

This is the schema's RECORD half: a register that is a function of the *trajectory*, not of
each position in isolation — so it can tell a survivor from a replica that share an endpoint
(the replica test). It is deliberately the simplest thing that can pass the bar, so it has a
fair chance to be *wrong* — which is the point.

DISCIPLINE — this is NOT wired into the seam, and ``forward`` emits a literal ZERO modulation:

  * the ENFORCE half (``P_ψ`` — the corrective delta) and the probe ``g`` / surface 𝒞 are
    Fabric-supplied and land in Phase 2 (PSI §5). Until then the default World-State stays
    ``NoOpWorldState`` and bit-identity keeps its meaning.
  * what is real and tested *here* is the latch predicate, exercised at the register level by
    ``run_scan`` and gated by ``tests/test_replica.py`` — the ψ analog of how
    ``test_bit_identity`` gates the no-op. Filling ``P_ψ`` before the replica test is green
    would void that meaning (see EPOCH-INVARIANT-GROUNDING.md).
"""

from __future__ import annotations

import torch
from torch import Tensor

from embraos_qnm.interfaces import PsiState, WorldStateInterface


class CandidateWorldState(WorldStateInterface):
    """The ψ₀ violation-latch register. Carried-state contract; seam-inert (zeros) for now."""

    def __init__(self, tau: float = 0.0) -> None:
        super().__init__()
        self.tau = tau

    def init_state(self, batch_size: int, device: torch.device) -> PsiState:
        # The carried latch m: one running-max scalar per sequence in the batch.
        return torch.zeros(batch_size, 1, device=device)

    def run_scan(self, c: Tensor, m0: Tensor | None = None) -> Tensor:
        """Register-level latch over a constraint-signal sequence (the replica-test entry).

        ``c``: per-step constraint signal, shape ``(T,)`` or ``(B, T)``. Returns the running
        latch ``m`` of the same shape, ``m_t = max(m_{t-1}, relu(c_t − τ))``, optionally
        seeded by a carried ``m0`` (the latch from earlier decode steps). ψ holds at t iff
        ``m_t == 0``; ``m[..., -1]`` is ψ over the whole trajectory so far.
        """
        violation = torch.relu(c - self.tau)  # per-step amount off 𝒞; exactly 0 while on it
        m = torch.cummax(violation, dim=-1).values  # causal: m_t depends only on the prefix ≤ t
        if m0 is not None:
            m = torch.maximum(m, m0)  # carry the latch across calls (decode steps)
        return m

    @staticmethod
    def psi_holds(m: Tensor) -> Tensor:
        """The ψ predicate from a latch value: on 𝒞 throughout (True) iff ``m == 0``."""
        return m == 0

    def forward(self, h: Tensor, psi: PsiState) -> tuple[Tensor, PsiState]:
        # ENFORCE (P_ψ) is deferred to Phase 2 (Fabric-supplied 𝒞): emit the honest zero
        # modulation and pass the register through. The tested latch lives in run_scan.
        return torch.zeros_like(h), psi
