"""ψ-carrying KV-cached decode (P2.7) — the token-identity gate (the KV analogue of bit-identity).

Arm A decodes through ``greedy_generate_psi``: HF's attention cache carries the K/V, the seam carries
the ψ₀ latch across decode steps. The gate: this cached decode must produce TOKEN-IDENTICAL output to
the no-cache oracle ``greedy_generate`` (which re-forwards the whole prefix each step — ψ-correct by
construction). Gated on the ``hf`` extra; a tiny-random Qwen3 core (no download) keeps it in CI.

The gate is VACUOUS at the ReZero cold start (gate_world == 0 ⇒ zero steering ⇒ all decoders agree
regardless of ψ), so every test sets gate_world != 0 to make the ψ path live; ``test_psi_carry_is_
load_bearing`` then shows an amnesiac (ψ-reset-per-step) decode actually diverges, so the gate can
fail.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.hf_core import HFCausalCore
from embraos_qnm.eval.arms import greedy_generate, greedy_generate_psi
from embraos_qnm.interfaces import FabricInterface
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.world_state.candidate import CandidateWorldState

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="requires the `hf` extra: uv sync --extra hf",
)

_GATE_WORLD = 5.0  # live (non-zero) steering so the ψ path actually affects the logits


def _tiny_qwen3() -> object:
    """Tiny-random Qwen3 (matches tests/test_hf_core.py::_tiny_qwen3), no download."""
    import transformers  # pyright: ignore[reportMissingImports]

    cfg = transformers.Qwen3Config(  # pyright: ignore[reportAttributeAccessIssue]
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(cfg)  # pyright: ignore[reportAttributeAccessIssue]


class _LiveFabric(FabricInterface):
    """A live Fabric stand-in mimicking the real R-GCN surface (fabric/gnn.py): a per-position,
    SCALE-INVARIANT distance ``c_t = 1 − cos(h_t, ref)`` in [0, 2] (a fixed reference direction
    stands in for the identity node reps). Direction-based, so it varies across the trajectory even
    when the tiny-random core's hidden states are near-zero in magnitude — a magnitude-based surface
    collapses to ~constant there and the carry becomes invisible. ``delta_fabric`` is unused
    (gate_fabric stays 0); only ``surface`` drives the World-State latch."""

    ref: Tensor

    def __init__(self, d_model: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(7)  # fixed reference direction, seed-independent
        self.register_buffer("ref", torch.randn(d_model, generator=gen))

    def forward(self, h: Tensor) -> Tensor:
        return h  # gated by gate_fabric == 0, so the value is unused

    def surface(self, h: Tensor) -> Tensor:
        ref = self.ref.to(h.dtype).view(1, 1, -1).expand_as(h)
        return 1.0 - F.cosine_similarity(h, ref, dim=-1)  # (B, T): [0, 2], per-position, scale-free


def _seam_model() -> tuple[HFCausalCore, QNMModel]:
    """A QNM-wrapped tiny Qwen3 with the seam ENABLED, a live surface, and a non-zero world gate."""
    core = HFCausalCore(model=_tiny_qwen3())
    cfg = QNMConfig(
        vocab_size=int(core._model.config.vocab_size),
        block_size=core.block_size,
        n_layer=core.num_layers(),
        n_head=1,  # unused for an injected core
        d_model=core.d_model,
        inject_layer=core.num_layers() // 2,
    )
    torch.manual_seed(0)  # pin the Fabric / World-State init independent of the model build
    qnm = QNMModel(
        cfg,
        core=core,
        fabric=_LiveFabric(core.d_model),
        world_state=CandidateWorldState(core.d_model),
    )
    with torch.no_grad():
        qnm.qnm_block.gate_world.fill_(_GATE_WORLD)  # live steering (else the gate is vacuous)
    qnm.eval()
    return core, qnm


def test_psi_cached_decode_matches_oracle() -> None:
    """Integration gate: the real greedy_generate_psi == the no-cache oracle greedy_generate,
    token-for-token (seam on, live gate) — the KV analogue of test_bit_identity."""
    core, qnm = _seam_model()
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 8))
    oracle = greedy_generate(
        lambda idx: qnm(idx)[0], ids, max_new_tokens=10, block_size=core.block_size, eos_id=None
    )
    cached = greedy_generate_psi(core._model, qnm.qnm_block, ids, max_new_tokens=10)
    assert torch.equal(cached, oracle)


def test_psi_cached_logits_match_oracle_each_step() -> None:
    """Strong gate (degeneracy-proof): at every generated position the cached step's next-token
    logits equal the oracle's (a full re-forward of the grown prefix), within fp tolerance. An
    untrained core that argmaxes the same token regardless of ψ can't hide a carry bug here."""
    core, qnm = _seam_model()
    seam = qnm.qnm_block
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 8))
    try:
        seam.psi_in = None
        out = core._model(ids, use_cache=True)  # prefill
        past, m = out.past_key_values, seam.psi_out
        cur = ids
        for step in range(8):
            nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
            seam.psi_in = None
            oracle_logits = qnm(cur)[0][:, -1]  # full re-forward, ψ-correct by construction
            seam.psi_in = m  # carry the running-max latch from the prior step
            out = core._model(nxt, past_key_values=past, use_cache=True)
            past, m = out.past_key_values, seam.psi_out
            assert torch.allclose(out.logits[:, -1], oracle_logits, atol=1e-4), f"step {step}"
    finally:
        seam.psi_in = None


def test_psi_carry_is_load_bearing() -> None:
    """Anti-vacuity: seeding the latch from the prior step (carry) vs resetting it (amnesiac) changes
    the next-step logits — so the gates above exercise the carry and would fail if the seam stopped
    persisting ψ."""
    core, qnm = _seam_model()
    seam = qnm.qnm_block
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 8))

    def next_step_logits(*, carry: bool) -> Tensor:
        seam.psi_in = None
        out = core._model(ids, use_cache=True)  # fresh prefill (own cache) per call
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        seam.psi_in = seam.psi_out if carry else None  # carry prompt latch, else reset (amnesiac)
        try:
            stepped = core._model(nxt, past_key_values=out.past_key_values, use_cache=True)
            return stepped.logits[:, -1]
        finally:
            seam.psi_in = None

    assert not torch.allclose(next_step_logits(carry=True), next_step_logits(carry=False))


def test_psi_decode_leaves_no_state_leak() -> None:
    """A full-sequence forward after a ψ-decode is unchanged: psi_in is reset (no stale latch)."""
    core, qnm = _seam_model()
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 6))
    with torch.no_grad():
        before = qnm(ids)[0].clone()
    greedy_generate_psi(core._model, qnm.qnm_block, ids, max_new_tokens=5)
    with torch.no_grad():
        after = qnm(ids)[0]
    assert torch.equal(before, after)
