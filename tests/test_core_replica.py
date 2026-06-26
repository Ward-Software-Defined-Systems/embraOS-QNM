"""Core-level replica test (P2.7) on a tiny core: the falsifier instruments run on REAL hidden
states (not synthetic c), the latch stays monotone, and a survivor (on 𝒞) and a replica (off 𝒞)
produce divergent carried ψ + steering. The collision SEARCH over a trained model is the gated run.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.eval.replica import (
    carried_latch,
    replica_divergence,
    search_collision,
    surface_trajectory,
)
from embraos_qnm.fabric.gnn import GNNFabric
from embraos_qnm.fabric.graph import load_graph
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.world_state.candidate import CandidateWorldState

_GRAPH = Path(__file__).resolve().parents[1] / "classical_constraints" / "Embra_IDENTITY.graph.json"


def _tiny_qnm(tau: float = 0.0) -> QNMModel:
    cfg = QNMConfig(vocab_size=32, block_size=32, n_layer=3, n_head=2, d_model=16, inject_layer=1)
    graph = load_graph(_GRAPH)
    torch.manual_seed(0)
    feats = torch.randn(len(graph.nodes), cfg.d_model)
    fabric = GNNFabric(graph, cfg.d_model, feats)
    ws = CandidateWorldState(cfg.d_model, tau=tau)
    torch.manual_seed(0)
    return QNMModel(cfg, fabric=fabric, world_state=ws)


def _set_tau(model: QNMModel, tau: float) -> None:
    cast(CandidateWorldState, model.qnm_block.world_state).tau = tau


def test_surface_trajectory_is_real_and_input_dependent() -> None:
    model = _tiny_qnm()
    a, b = torch.randint(0, 32, (1, 8)), torch.randint(0, 32, (1, 8))
    ca, cb = surface_trajectory(model, a), surface_trajectory(model, b)
    assert ca.shape == (1, 8) and torch.isfinite(ca).all() and (ca >= 0).all()
    assert not torch.allclose(ca, cb)  # c_t depends on the real hidden-state trajectory


def test_carried_latch_is_monotone() -> None:
    m = carried_latch(_tiny_qnm(), torch.randint(0, 32, (1, 10)))
    assert torch.all(m[:, 1:] >= m[:, :-1] - 1e-6)  # cummax => non-decreasing along the trajectory


def test_core_replica_separates_survivor_from_replica() -> None:
    model = _tiny_qnm()
    torch.manual_seed(5)
    seqs = [torch.randint(0, 32, (1, 8)) for _ in range(10)]
    maxc = [float(surface_trajectory(model, s).max()) for s in seqs]
    survivor = seqs[min(range(10), key=lambda i: maxc[i])]
    replica = seqs[max(range(10), key=lambda i: maxc[i])]
    lo, hi = min(maxc), max(maxc)
    assert lo < hi  # the two histories really do differ in how far off 𝒞 they travel
    _set_tau(model, (lo + hi) / 2)  # τ between them: survivor stays on 𝒞, replica leaves it

    div = replica_divergence(model, survivor, replica)
    assert div["survivor_psi_holds"] and not div["replica_psi_holds"]  # ψ diverges
    assert div["replica_steer_norm"] > div["survivor_steer_norm"]  # enforce fires only off 𝒞


def test_search_collision_runs() -> None:
    model = _tiny_qnm()
    torch.manual_seed(7)
    cands = [torch.randint(0, 32, (1, 8)) for _ in range(6)]
    maxc = [float(surface_trajectory(model, s).max()) for s in cands]
    _set_tau(model, (min(maxc) + max(maxc)) / 2)  # ensure both survivors and replicas exist
    res = search_collision(model, cands)
    assert res is None or (0 <= res[0] < res[1] < len(cands) and res[2] >= 0.0)
