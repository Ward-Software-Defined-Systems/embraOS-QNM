"""GNNFabric (P2.3): the R-GCN identity Fabric — graph loads/validates, delta+surface shapes,
the surface is differentiable and input-dependent, and the real Fabric is cold-start inert in
the seam (the seam's gate, not the Fabric, guarantees the null). No model/transformers needed.
"""

from __future__ import annotations

from pathlib import Path

import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.transformer import TinyTransformer
from embraos_qnm.fabric.gnn import GNNFabric
from embraos_qnm.fabric.graph import RELATIONS, load_graph
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.world_state import NoOpWorldState

_GRAPH = Path(__file__).resolve().parents[1] / "classical_constraints" / "Embra_IDENTITY.graph.json"


def _fabric(d_model: int) -> GNNFabric:
    graph = load_graph(_GRAPH)
    torch.manual_seed(0)
    feats = torch.randn(len(graph.nodes), d_model)  # stand-in for Core-embedded node text
    return GNNFabric(graph, d_model, feats)


def test_identity_graph_loads_and_validates() -> None:
    graph = load_graph(_GRAPH)
    assert any(n.id == "embra" for n in graph.nodes)
    assert any(n.id == "no_pretense" for n in graph.nodes)  # the constraint content is present
    assert all(e.relation in RELATIONS for e in graph.edges)
    assert len({n.id for n in graph.nodes}) == len(graph.nodes)  # unique ids


def test_fabric_emits_delta_and_surface_shapes() -> None:
    d = 32
    fab = _fabric(d)
    h = torch.randn(2, 5, d)
    delta = fab(h)
    c = fab.surface(h)
    assert delta.shape == (2, 5, d)
    assert c.shape == (2, 5)
    assert torch.isfinite(delta).all() and torch.isfinite(c).all()
    assert (c >= 0).all()  # 1 - max cosine, and max cosine <= 1


def test_surface_is_differentiable_and_input_dependent() -> None:
    d = 32
    fab = _fabric(d)
    h = torch.randn(2, 5, d, requires_grad=True)
    c = fab.surface(h)
    c.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert not torch.allclose(fab.surface(torch.randn(2, 5, d)), c.detach())  # depends on h


def test_cold_start_inert_with_gnn_fabric() -> None:
    """The real GNNFabric in the seam with the gate at zero init is bit-identical to the plain
    Core — the cold-start guarantee holds with a live, non-trivial Fabric."""
    cfg = QNMConfig(vocab_size=17, block_size=16, n_layer=3, n_head=2, d_model=32, inject_layer=1)
    fab = _fabric(cfg.d_model)  # build first; its RNG use is irrelevant (re-seeded below)
    torch.manual_seed(0)
    plain = TinyTransformer(cfg)
    torch.manual_seed(0)
    qnm = QNMModel(cfg, fabric=fab, world_state=NoOpWorldState())
    idx = torch.randint(0, cfg.vocab_size, (3, 8))
    assert torch.equal(plain(idx), qnm(idx)[0])


def test_qnm_fabric_plus_candidate_world_state_cold_start_inert() -> None:
    """ψ fully wired (Fabric surface -> CandidateWorldState latch -> learned enforce) is STILL
    bit-identical at gate-0 -- wiring the SOUL in does not break the null."""
    from embraos_qnm.world_state.candidate import CandidateWorldState

    cfg = QNMConfig(vocab_size=17, block_size=16, n_layer=3, n_head=2, d_model=32, inject_layer=1)
    fab = _fabric(cfg.d_model)
    ws = CandidateWorldState(d_model=cfg.d_model, tau=0.5)
    torch.manual_seed(0)
    plain = TinyTransformer(cfg)
    torch.manual_seed(0)
    qnm = QNMModel(cfg, fabric=fab, world_state=ws)
    idx = torch.randint(0, cfg.vocab_size, (3, 8))
    assert torch.equal(plain(idx), qnm(idx)[0])
