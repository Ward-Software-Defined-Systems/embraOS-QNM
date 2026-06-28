"""test_replica.py — gates ψ on (PSI-OPERATIONAL-GROUNDING.md §6).

The ψ analog of test_bit_identity: until these pass, the World-State stays a literal
``zeros_like`` null. They test the *register-level* latch on hand-built constraint signals —
the load-bearing property (trajectory-dependence) without needing real hidden-state
collisions. The stronger Core-level version (two token histories that collide at h_T via
different paths) is the Phase-2 follow-up.

The bar (EPOCH-INVARIANT-GROUNDING.md): a real ψ is trajectory-dependent (passes the replica
test), can be false mid-trajectory, and is not true-by-construction.

    A definition that cannot be false somewhere is a name.
    A definition that survives the replica test is a tool.
"""

from __future__ import annotations

import torch

from embraos_qnm.world_state.candidate import CandidateWorldState

TAU = 0.5

# Two trajectories of per-step constraint signals that share the SAME endpoint (c_T = 0.3)
# but differ mid-path: the survivor stays inside 𝒞 throughout; the replica exits 𝒞 at t=1
# (0.9 > τ) and returns. A pointwise ψ sees only the endpoint and cannot tell them apart.
_C_SURVIVOR = torch.tensor([0.0, 0.1, 0.2, 0.1, 0.3])  # never exceeds τ
_C_REPLICA = torch.tensor([0.0, 0.9, 0.2, 0.1, 0.3])  # exceeds τ at t=1, then returns


def _pointwise_psi(c_t: torch.Tensor, tau: float = TAU) -> bool:
    """The pointwise (endpoint-only) view of ψ: on 𝒞 iff the final signal is within τ."""
    return bool(torch.relu(c_t - tau) == 0)


def test_carried_psi_separates_replica_from_survivor() -> None:
    """The load-bearing claim: same endpoint, different path => different ψ. The pointwise view
    cannot tell them apart; the carried latch must."""
    ws = CandidateWorldState(d_model=8, tau=TAU)
    mA = ws.run_scan(_C_SURVIVOR)
    mB = ws.run_scan(_C_REPLICA)
    psiA = bool(ws.psi_holds(mA[-1]))
    psiB = bool(ws.psi_holds(mB[-1]))

    # the pointwise (endpoint-only) view cannot tell them apart (both final signals are 0.3):
    assert _pointwise_psi(_C_SURVIVOR[-1]) == _pointwise_psi(_C_REPLICA[-1])
    # a genuine continuity-invariant MUST:
    assert psiA != psiB, "ψ is still the static one in a trajectory costume"
    assert psiA is True and psiB is False  # survivor held; replica crossed and is marked


def test_psi_can_be_false_mid_trajectory() -> None:
    """The content is in the path, not the boundary: ψ goes false at the step of the crossing
    (t=1 here), not only at the end."""
    ws = CandidateWorldState(d_model=8, tau=TAU)
    m = ws.run_scan(_C_REPLICA)
    assert bool(ws.psi_holds(m[0]))  # t=0: still on 𝒞
    assert not bool(ws.psi_holds(m[1]))  # t=1: latch tripped mid-trajectory
    assert not bool(ws.psi_holds(m[-1]))  # stays tripped (monotone)


def test_psi_is_not_true_by_construction() -> None:
    """There is a trajectory on which ψ is false (the replica) and one on which it holds (the
    survivor): the predicate is falsifiable, not a tautology — and τ is not set so high that
    nothing can ever cross (the vacuous-latch failure the grounding note warns about)."""
    ws = CandidateWorldState(d_model=8, tau=TAU)
    assert not bool(ws.psi_holds(ws.run_scan(_C_REPLICA)[-1]))  # ψ can be false
    assert bool(ws.psi_holds(ws.run_scan(_C_SURVIVOR)[-1]))  # and ψ can be true


def test_carried_state_persists_across_calls() -> None:
    """The register carries the latch across separate forwards (decode steps): a violation in
    an earlier chunk is remembered when a later chunk, viewed alone, looks clean."""
    ws = CandidateWorldState(d_model=8, tau=TAU)
    early = torch.tensor([[0.0, 0.9]])  # (B=1, T=2): crosses τ
    late = torch.tensor([[0.1, 0.2]])  # later chunk, clean on its own
    m_early = ws.run_scan(early)
    m_late = ws.run_scan(late, m0=m_early[:, -1:])  # carry the latch forward
    assert not bool(ws.psi_holds(m_late[0, -1])), "a remembered violation must survive the carry"
    # without the carry, the clean late chunk alone would (wrongly) read as on-𝒞:
    assert bool(ws.psi_holds(ws.run_scan(late)[0, -1]))


def test_enforce_fires_only_after_the_latch_trips() -> None:
    """Core-level mechanism (P2.4): the learned P_ψ enforce delta is exactly zero while on 𝒞 and
    nonzero once the trajectory has gone off 𝒞 — trajectory-dependent correction on real h."""
    torch.manual_seed(0)
    d = 16
    ws = CandidateWorldState(d_model=d, tau=TAU)
    h = torch.randn(1, 5, d)
    c = torch.tensor([[0.0, 0.1, 0.9, 0.2, 0.1]])  # crosses τ at t=2; the latch stays tripped
    delta, psi_next = ws(h, ws.init_state(1, h.device), c)
    norms = delta.norm(dim=-1)[0]  # per-token enforce magnitude
    assert torch.equal(norms[:2], torch.zeros(2)), "no correction while on 𝒞 (t < 2)"
    assert (norms[2:] > 0).all(), "correction fires once off 𝒞 (t >= 2, monotone latch)"
    assert not bool(ws.psi_holds(psi_next)), "carried ψ records the violation for the next step"
