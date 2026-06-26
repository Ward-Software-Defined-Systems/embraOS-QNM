"""
test_pathway_capacity.py  —  the capability counterpart to test_bit_identity.py

Place at: tests/test_pathway_capacity.py

WHY THIS EXISTS
---------------
test_bit_identity.py proves the seam is *inert*: with no-op components the QNM is
bit-for-bit a plain transformer. That is the H0 -- "the architecture changed nothing."
This file proves the seam is *capable and controllable*: the Fabric pathway can be driven
to carry a prescribed, feature-varying signal into the residual stream, and the ReZero
gate is the thing that unlocks it. Together they bracket the seam:

    bit-identity      :  gate == 0  =>  pathway provably inert      (exact, torch.equal)
    pathway-capacity  :  gate ->  *  =>  pathway carries intended Δ (learned, tolerances)

The capacity target is mean-zero across features on purpose, so it is NOT annihilated by a
downstream LayerNorm (ARCHITECTURE Sec 3.3). A uniform (all-ones) target would be invisible
-- that negative finding is already covered by the gate-gradient test and is not re-tested
here.

BINDINGS — resolved against the real repo (were assumptions; confirmed/edited as noted):
  1. QNMBlock import + constructor. CONFIRMED: `from embraos_qnm.manifold import QNMBlock`
     (re-exported) and `QNMBlock(block, fabric, world_state)` (real sig adds only `*, enabled=True`).
  2. Fabric-pathway gate attribute. RESOLVED: the real name is `gate_fabric` (a zero-init scalar
     nn.Parameter), not `g_f` -- applied in `_fabric_gate` below.
  3. The seam applies the Fabric to the POST-block residual h_base = Block(h)
     (ARCHITECTURE Sec 3.1). CONFIRMED in QNMBlock.forward. The target operator matches.
  4. Fabric/WorldState contract `forward(h:(B,T,D)) -> (B,T,D)`, additive. CONFIRMED.

Everything else -- the inner "core" block, the Fabric, the no-op World-State -- is a
test-local stand-in. This file therefore pins ONLY the seam contract and is independent of
Core / Config internals; dropping the real `core.Block` in changes nothing about what is
tested.

Run:  uv run pytest tests/test_pathway_capacity.py -q        (CPU, float32, deterministic)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

# --- binding point #1: the real seam under test --------------------------------------
from embraos_qnm.interfaces import FabricInterface, PsiState, WorldStateInterface
from embraos_qnm.manifold import QNMBlock  # re-exported from embraos_qnm.manifold (binding #1)

D = 32  # embedding dim
T = 16  # sequence length
B = 4  # batch
STEPS = 1500
LR = 1e-2
SEED = 0


def _meancenter(x: torch.Tensor) -> torch.Tensor:
    """Project onto the zero-mean-across-features subspace -> out of the LayerNorm null space."""
    return x - x.mean(dim=-1, keepdim=True)


class _FrozenCoreBlock(nn.Module):
    """Stand-in for core.Block: any frozen (B,T,D)->(B,T,D) map. Frozen so that during the
    test only the Fabric pathway can move."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.proj(self.ln(h))  # residual-style, like a real block


class _LinearFabric(FabricInterface):
    """Minimal trainable Fabric satisfying forward(h)->delta. Output is mean-centered so it
    lives outside the LayerNorm null space. Stands in for the (not-yet-built) real Fabric;
    its only job here is to demonstrate the seam can be driven to an intended signal."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return _meancenter(self.w(h))


class _ZeroWorldState(WorldStateInterface):
    """No-op World-State under the carried-state contract: zeros + pass-through register."""

    def init_state(self, batch_size: int, device: torch.device) -> PsiState:
        return None

    def forward(
        self, h: torch.Tensor, psi: PsiState, c: torch.Tensor
    ) -> tuple[torch.Tensor, PsiState]:
        return torch.zeros_like(h), psi


# --- binding point #2: gate accessor -------------------------------------------------
def _fabric_gate(qnm_block: QNMBlock) -> nn.Parameter:
    """Return the Fabric-pathway ReZero scalar. Resolved against the real QNMBlock: the
    gate is `gate_fabric`, a zero-initialized nn.Parameter (header assumption #2)."""
    return qnm_block.gate_fabric


def _build() -> tuple[
    QNMBlock,
    _FrozenCoreBlock,
    _LinearFabric,
    torch.Tensor,
    Callable[[torch.Tensor], torch.Tensor],
]:
    torch.manual_seed(SEED)
    block = _FrozenCoreBlock(D)
    fabric = _LinearFabric(D)
    qnm_block = QNMBlock(block, fabric, _ZeroWorldState())  # binding point #1

    h0 = torch.randn(B, T, D)

    # Fixed, feature-varying, mean-zero target *operator* applied to the post-block residual
    # h_base = block(h0). This is the signal we want the Fabric pathway to learn to carry.
    w_star = torch.randn(D, D)

    def target_delta(h_base: torch.Tensor) -> torch.Tensor:
        return _meancenter(h_base @ w_star.T)

    return qnm_block, block, fabric, h0, target_delta


def test_cold_start_pathway_is_inert() -> None:
    """ReZero cold-start: at init the seam adds exactly nothing, even though the Fabric is
    non-trivial. Same guarantee as bit-identity, asserted exactly."""
    qnm_block, block, _fabric, h0, _target = _build()
    gate = _fabric_gate(qnm_block)
    assert torch.equal(gate.detach(), torch.zeros_like(gate)), "gate must be zero-initialized"
    with torch.no_grad():
        assert torch.equal(qnm_block(h0), block(h0)), (
            "cold-start QNMBlock must equal the bare block exactly"
        )


def test_pathway_carries_prescribed_signal() -> None:
    """The load-bearing capability claim: training only the Fabric + its gate, the seam
    learns to realize a prescribed feature-varying modulation of the residual stream."""
    qnm_block, block, fabric, h0, target_delta = _build()
    gate = _fabric_gate(qnm_block)

    with torch.no_grad():
        h_base = block(h0)
        tgt_resid = h_base + target_delta(h_base)  # baseline + intended delta

    opt = torch.optim.Adam(list(fabric.parameters()) + [gate], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = torch.mean((qnm_block(h0) - tgt_resid) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        target = target_delta(block(h0))
        realized = qnm_block(h0) - block(h0)  # == gate_fabric * Fabric(h_base)
        final_loss = torch.mean((qnm_block(h0) - tgt_resid) ** 2).item()
        # Scale-invariant "clearly learned" bars (an absolute MSE threshold would depend on
        # the target's scale): the fraction of the target signal's variance left unexplained,
        # and the relative L2 error of the realized delta against the prescribed one.
        baseline_loss = torch.mean(target**2).item()  # loss if the pathway stayed inert
        unexplained = final_loss / baseline_loss
        rel_err = ((realized - target).norm() / target.norm()).item()

    assert gate.detach().abs().item() > 1e-3, "gate never left zero -- pathway was not used"
    assert unexplained < 1e-2, (
        f"pathway carried <99% of the prescribed signal (unexplained variance={unexplained:.2e})"
    )
    assert rel_err < 0.1, f"realized delta != prescribed signal (relative L2 error={rel_err:.2e})"


def test_zero_gate_cannot_carry_signal() -> None:
    """Negative control / ReZero semantics: with the gate frozen at zero, NO amount of Fabric
    training can change the output. Proves the gate is the load-bearing unlock and the pathway
    is genuinely gated. Asserted exactly (0 * anything == 0 in IEEE-754)."""
    qnm_block, block, fabric, h0, target_delta = _build()
    gate = _fabric_gate(qnm_block)

    with torch.no_grad():
        h_base = block(h0)
        tgt_resid = h_base + target_delta(h_base)

    # gate is deliberately NOT in the optimizer -> it stays at its zero init
    opt = torch.optim.Adam(fabric.parameters(), lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = torch.mean((qnm_block(h0) - tgt_resid) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        realized = qnm_block(h0) - block(h0)
        assert torch.equal(realized, torch.zeros_like(realized)), (
            "gate frozen at zero must leave the pathway exactly inert regardless of Fabric"
        )
        assert torch.equal(gate.detach(), torch.zeros_like(gate)), "gate must remain zero"
