"""World-State — SOUL lives here.

A persistent register that applies the ``P_ψ`` projection onto the constraint surface.
ψ (the invariant) is deliberately UNDEFINED: the no-op here is the honest H₀ null,
not a placeholder to be quietly fleshed out. See world_state/noop.py.
"""

from embraos_qnm.world_state.noop import NoOpWorldState

__all__ = ["NoOpWorldState"]
