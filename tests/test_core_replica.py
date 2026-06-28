"""Trajectory infra on a tiny core: the decode-history builder (``history_ids``) and the
injection-layer node reps (``injection_node_reps``, used by the enforce install). The closed geometric
replica scouts these once accompanied are removed — their findings are in
``docs/PSI-EMBRA-ANALYSIS-AND-FINDINGS.md`` (Parts I–III).
"""

from __future__ import annotations

import types
from pathlib import Path

import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.eval.replica import history_ids, injection_node_reps
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


class _FakeTok:
    """Tiny-vocab stand-in: ids stay < 32 for the tiny core's embedding."""

    eos_token_id = 0

    def apply_chat_template(self, messages, **kw):
        n = sum(len(m["content"]) for m in messages) % 5 + 3
        return torch.arange(1, n + 1).unsqueeze(0) % 32

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = torch.tensor([[(ord(c) % 31) + 1 for c in text[:6]] or [1]])
        return types.SimpleNamespace(input_ids=ids)


def test_history_ids_concatenates_prompt_and_continuation() -> None:
    ids, cont_len = history_ids(_FakeTok(), "What is your name?", "I am Embra.", "cpu")
    assert ids.shape[0] == 1 and cont_len > 0
    assert ids.shape[1] > cont_len  # prompt (user turn) + continuation


def test_injection_node_reps_match_model_space() -> None:
    model = _tiny_qnm()
    graph = load_graph(_GRAPH)
    reps = injection_node_reps(model, _FakeTok(), graph, "cpu")
    assert reps.shape == (len(graph.nodes), model.config.d_model)  # one rep per node, in h-space
