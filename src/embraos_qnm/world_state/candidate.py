"""CandidateWorldState — ψ₀, the first invariant put on trial (PSI-OPERATIONAL-GROUNDING.md §5).

ψ₀ is a *causal cumulative violation latch* over the Fabric-supplied constraint signal
``c_t = g(h_t)`` (``c_t > τ`` means the trajectory has left the constraint surface 𝒞 at step t):

    m_t = max(m_{t-1}, relu(c_t − τ))        # monotone cummax over the causal prefix
    ψ holds at t   ⟺   m_t == 0               # never crossed the boundary up to t

Two halves of the schema (PSI §4):
  * RECORD — the latch ``m`` (a function of the *trajectory*, not each position alone), carried
    across decode steps via the ψ-state register. This is what passes the replica test.
  * ENFORCE — ``P_ψ``: a **learned, latch-gated steering delta** ``tanh(m) · steer(h)`` that fires
    only once/where the run has gone off 𝒞 (chosen P2.4 design). It is trained in P2.5 against the
    no-pretense objective; ``steer`` is the only learned part here.

DISCIPLINE: this is the candidate *on trial*, NOT the default. The default World-State stays
``NoOpWorldState`` (literal zeros) and bit-identity keeps its meaning; CandidateWorldState is only
wired into the seam (and its ReZero gate trained off zero) once ``tests/test_replica.py`` is green.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from embraos_qnm.interfaces import PsiState, WorldStateInterface


class CandidateWorldState(WorldStateInterface):
    """The ψ₀ violation-latch register + a learned latch-gated enforce delta (P_ψ)."""

    def __init__(self, d_model: int, tau: float = 0.0) -> None:
        super().__init__()
        self.tau = tau
        # P_ψ: a learned steering delta, gated by the latch. Trained in P2.5 (Core frozen).
        self.steer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def init_state(self, batch_size: int, device: torch.device) -> PsiState:
        return torch.zeros(batch_size, 1, device=device)  # the carried running-max latch m

    def _latch(self, c: Tensor, m0: Tensor | None) -> Tensor:
        """The causal cumulative violation latch over c, optionally seeded by a carried m0."""
        violation = torch.relu(c - self.tau)  # per-step amount off 𝒞; exactly 0 while on it
        m = torch.cummax(violation, dim=-1).values  # causal: m_t depends only on the prefix ≤ t
        if m0 is not None:
            m = torch.maximum(m, m0)  # carry the latch across calls (decode steps)
        return m

    def run_scan(self, c: Tensor, m0: Tensor | None = None) -> Tensor:
        """Register-level latch over a constraint-signal sequence (the replica-test entry)."""
        return self._latch(c, m0)

    @staticmethod
    def psi_holds(m: Tensor) -> Tensor:
        """The ψ predicate from a latch value: on 𝒞 throughout (True) iff ``m == 0``."""
        return m == 0

    def forward(self, h: Tensor, psi: PsiState, c: Tensor) -> tuple[Tensor, PsiState]:
        m = self._latch(c, psi)  # RECORD: advance the latch with the Fabric-supplied surface c
        delta = torch.tanh(m).unsqueeze(-1) * self.steer(h)  # ENFORCE: latch-gated learned P_ψ
        psi_next = m[..., -1:].detach()  # carry m_T to the next step (state, not a grad path)
        return delta, psi_next
