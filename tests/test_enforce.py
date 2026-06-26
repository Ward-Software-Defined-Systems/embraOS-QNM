"""P2.5 enforce-training mechanism on a tiny core (no transformers / no download).

Pins the load-bearing guarantees of the training loop: the Core (including the wrapped seam layer)
is frozen, gradients flow ONLY to the side-pathway, the ReZero gates move off zero, and the
checkpoint round-trips the side-pathway alone. The real Qwen3-8B run is a gated CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from embraos_qnm.config import QNMConfig
from embraos_qnm.core.transformer import TinyTransformer
from embraos_qnm.fabric.gnn import GNNFabric
from embraos_qnm.fabric.graph import load_graph
from embraos_qnm.manifold.model import QNMModel
from embraos_qnm.train_enforce import (
    EnforceConfig,
    freeze_to_side_pathway,
    load_side_pathway,
    side_pathway_state_dict,
    train_enforce,
)
from embraos_qnm.world_state.candidate import CandidateWorldState

_GRAPH = Path(__file__).resolve().parents[1] / "classical_constraints" / "Embra_IDENTITY.graph.json"


def _tiny_model() -> QNMModel:
    cfg = QNMConfig(vocab_size=32, block_size=32, n_layer=3, n_head=2, d_model=16, inject_layer=1)
    graph = load_graph(_GRAPH)
    torch.manual_seed(0)
    feats = torch.randn(len(graph.nodes), cfg.d_model)  # stand-in for Core-embedded node text
    fabric = GNNFabric(graph, cfg.d_model, feats)
    world_state = CandidateWorldState(cfg.d_model, tau=0.0)  # τ=0 => the latch trips on real h
    torch.manual_seed(0)
    return QNMModel(cfg, fabric=fabric, world_state=world_state)


def _synthetic_samples(model: QNMModel) -> list[tuple[torch.Tensor, torch.Tensor, bool]]:
    vocab = model.config.vocab_size
    torch.manual_seed(1)
    samples = []
    for i in range(6):
        prompt = torch.randint(0, vocab, (1, 5))
        target = torch.randint(0, vocab, (1, 3))
        samples.append((prompt, target, i % 3 == 0))  # mix of answerable / no-pretense
    return samples


def test_freeze_isolates_the_side_pathway() -> None:
    model = _tiny_model()
    freeze_to_side_pathway(model)
    block = model.qnm_block

    # the wrapped Core decoder layer and the rest of the Core are frozen...
    assert all(not p.requires_grad for p in block.block.parameters())
    assert not cast(TinyTransformer, model.core).lm_head.weight.requires_grad
    assert all(not p.requires_grad for p in model.core.blocks[0].parameters())
    # ...and ONLY the side-pathway is trainable.
    assert all(p.requires_grad for p in block.fabric.parameters())
    assert all(p.requires_grad for p in block.world_state.parameters())
    assert block.gate_fabric.requires_grad and block.gate_world.requires_grad


def test_training_moves_side_pathway_not_core() -> None:
    model = _tiny_model()
    core = cast(TinyTransformer, model.core)
    fabric = cast(GNNFabric, model.qnm_block.fabric)
    samples = _synthetic_samples(model)
    cap_ids = [torch.randint(0, model.config.vocab_size, (1, 8))]

    head_before = core.lm_head.weight.detach().clone()
    wrapped = next(model.qnm_block.block.parameters())
    wrapped_before = wrapped.detach().clone()
    fabric_w = fabric.delta_proj.weight
    fabric_before = fabric_w.detach().clone()

    losses = train_enforce(
        model, samples, cap_ids, EnforceConfig(steps=8, batch_size=3), log_every=0
    )

    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    # frozen Core is bit-for-bit unchanged
    assert torch.equal(core.lm_head.weight, head_before)
    assert torch.equal(wrapped, wrapped_before)
    # the side-pathway learned: a Fabric weight moved and the ReZero gates left zero
    assert not torch.equal(fabric_w, fabric_before)
    assert model.qnm_block.gate_fabric.item() != 0.0
    assert model.qnm_block.gate_world.item() != 0.0


def test_checkpoint_roundtrips_side_pathway_only() -> None:
    model = _tiny_model()
    freeze_to_side_pathway(model)
    fabric = cast(GNNFabric, model.qnm_block.fabric)
    with torch.no_grad():  # give the gates a nonzero value to round-trip
        model.qnm_block.gate_world.fill_(0.37)
    saved = side_pathway_state_dict(model)
    assert "gate_world" in saved and any(k.startswith("fabric.") for k in saved)

    with torch.no_grad():  # perturb, then restore from the checkpoint
        model.qnm_block.gate_world.fill_(0.0)
        fabric.delta_proj.weight.add_(1.0)
    load_side_pathway(model, saved)

    assert torch.equal(model.qnm_block.gate_world, saved["gate_world"])  # exact roundtrip
    assert torch.equal(fabric.delta_proj.weight, saved["fabric.delta_proj.weight"])


def test_seam_toggle_serves_arm0_and_armA() -> None:
    """The Arm-A runner serves all arms from ONE core via the seam toggle: OFF == stock (Arm 0/P,
    unaffected by the trained checkpoint), ON == Arm A (differs)."""
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    block = model.qnm_block
    with torch.no_grad():  # simulate a trained checkpoint: gates off zero
        block.gate_fabric.fill_(0.5)
        block.gate_world.fill_(0.5)

    block.enabled = False  # Arm 0/P
    with torch.no_grad():
        out0 = model(ids)[0].clone()
    block.enabled = True  # Arm A
    with torch.no_grad():
        out_a = model(ids)[0]
    assert not torch.equal(out_a, out0)  # Arm A (seam on, trained gates) differs from Arm 0

    # seam OFF is the stock Core regardless of gate values => the checkpoint never leaks into Arm 0/P
    block.enabled = False
    with torch.no_grad():
        block.gate_fabric.fill_(2.0)
        block.gate_world.fill_(-1.0)
        assert torch.equal(model(ids)[0], out0)
