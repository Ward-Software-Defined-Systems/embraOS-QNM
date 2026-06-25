"""World-State — SOUL lives here.

A persistent register that applies the ``P_ψ`` projection onto the constraint surface.
ψ (the invariant) is deliberately UNDEFINED: the no-op here is the honest H₀ null,
not a placeholder to be quietly fleshed out. See world_state/noop.py.
"""

from embraos_qnm.world_state.candidate import CandidateWorldState
from embraos_qnm.world_state.noop import NoOpWorldState

# NoOpWorldState is the default (the honest null). CandidateWorldState is the gated ψ₀ latch
# (PSI §5) — exposed for the replica-test harness, NOT wired into the seam until it is green.
__all__ = ["CandidateWorldState", "NoOpWorldState"]
