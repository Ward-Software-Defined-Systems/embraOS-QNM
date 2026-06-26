"""eval/replica.py — the CORE-LEVEL replica test (NEXT-STEPS P2.7): the strong falsifier.

``tests/test_replica.py`` shows ψ₀ is trajectory-dependent at the REGISTER level — fed a synthetic
constraint signal ``c``. This module feeds the latch the REAL thing: ``c_t = g(h_t)`` computed from
the Core's own hidden state at the injection layer (the graph-𝒞 distance from ``GNNFabric.surface``).
The load-bearing question (PSI §5) is whether that graph-𝒞 carries usable signal on genuine hidden
states; the falsifier is a **survivor vs replica** pair — two real token histories that reach (near)
the same endpoint ``h_T`` by different paths, one of which left 𝒞 mid-sequence — whose carried ψ and
steering must DIVERGE if 𝒞 means anything.

Here are the instruments (``surface_trajectory``, ``carried_latch``, ``replica_divergence``) and a
``search_collision`` utility. The instruments + the mechanism are unit-tested on a tiny core; the
search for a TRUE collision over a *trained* model on real probe continuations is the gated real run.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.world_state.candidate import CandidateWorldState


def injection_hidden(model: QNMModel, ids: Tensor) -> Tensor:
    """The Core's hidden state at the injection layer (the wrapped block's output), (B, T, D).

    Captured with a forward hook on the wrapped layer — no seam change, so the bit-identity null is
    untouched. Read-only; runs under ``no_grad``.
    """
    grab: dict[str, Tensor] = {}

    def hook(_module: object, _inp: object, out: object) -> None:
        h = out[0] if isinstance(out, tuple) else out
        grab["h"] = cast(Tensor, h).detach()

    handle = model.qnm_block.block.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return grab["h"]


def surface_trajectory(model: QNMModel, ids: Tensor) -> Tensor:
    """c_t over a token history, from the REAL injection-layer hidden states, (B, T)."""
    h = injection_hidden(model, ids)
    with torch.no_grad():
        return model.qnm_block.fabric.surface(h)


def carried_latch(model: QNMModel, ids: Tensor) -> Tensor:
    """The ψ₀ latch m_t over the real c_t trajectory (cummax of relu(c − τ)), (B, T)."""
    ws = cast(CandidateWorldState, model.qnm_block.world_state)
    return ws.run_scan(surface_trajectory(model, ids))


def replica_divergence(
    model: QNMModel, survivor: Tensor, replica: Tensor
) -> dict[str, float | bool]:
    """Carried ψ + steering for a survivor vs a replica history. ψ must diverge if 𝒞 is real.

    ``survivor`` stays on 𝒞 (latch never trips → ψ holds, steering ≈ 0); ``replica`` left 𝒞
    mid-sequence (latch trips → ψ false, steering fires). Whether real histories can be made to
    realize this on a TRAINED model is the open question this instrument exists to decide.
    """
    ws = cast(CandidateWorldState, model.qnm_block.world_state)
    m_s, m_r = carried_latch(model, survivor), carried_latch(model, replica)
    h_s, h_r = injection_hidden(model, survivor), injection_hidden(model, replica)
    with torch.no_grad():
        delta_s, _ = ws(h_s, None, surface_trajectory(model, survivor))
        delta_r, _ = ws(h_r, None, surface_trajectory(model, replica))
    return {
        "survivor_psi_holds": bool(ws.psi_holds(m_s[..., -1:]).all()),
        "replica_psi_holds": bool(ws.psi_holds(m_r[..., -1:]).all()),
        "survivor_latch_end": float(m_s[..., -1].max()),
        "replica_latch_end": float(m_r[..., -1].max()),
        "survivor_steer_norm": float(delta_s[..., -1, :].norm()),
        "replica_steer_norm": float(delta_r[..., -1, :].norm()),
    }


def search_collision(model: QNMModel, candidates: list[Tensor]) -> tuple[int, int, float] | None:
    """HEAVY/real-run falsifier: among real token histories, find the pair whose endpoint hidden
    states ``h_T`` are CLOSEST while their carried ψ DIFFERS (one left 𝒞, one didn't).

    A small ``h``-distance with diverging ψ is the load-bearing evidence that the graph-𝒞 is real on
    genuine hidden states; if the closest such pair is still far apart (or none diverge), 𝒞 is noise
    and the fallback surface is needed (PSI §5). Returns ``(i, j, h_distance)`` or None. Build
    ``candidates`` from real probe continuations on the TRAINED model.
    """
    ws = cast(CandidateWorldState, model.qnm_block.world_state)
    ends = [injection_hidden(model, c)[..., -1, :].reshape(-1) for c in candidates]
    holds = [bool(ws.psi_holds(carried_latch(model, c)[..., -1:]).all()) for c in candidates]
    best: tuple[int, int, float] | None = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if holds[i] == holds[j]:
                continue  # need a survivor paired with a replica (diverging ψ)
            dist = float((ends[i] - ends[j]).norm())
            if best is None or dist < best[2]:
                best = (i, j, dist)
    return best
