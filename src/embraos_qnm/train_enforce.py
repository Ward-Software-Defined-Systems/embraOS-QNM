"""train_enforce.py — P2.5 enforce training: train the ψ side-pathway with the Core FROZEN.

The side-pathway is the Fabric (R-GCN + attention) + the World-State ``steer`` (P_ψ) + the two
ReZero gates. The Core — including the decoder layer wrapped inside the ``QNMBlock`` — stays frozen,
so any adherence the model gains is an *architecture* effect, not a finetune (PREREG §5). The
freeze follows the documented footgun: freeze ALL params, then un-freeze ONLY the side-pathway —
never the wrapped layer that now lives inside the model tree.

Objective (a differentiable proxy for the discrete judge — the judge labels/eval come later):

    loss = adherence·CE(adherent target | no-pretense probe)        # raise P(hedge)
         + λ₁·CE(correct answer | answerable control)               # anti-mutism (PREREG §6)
         + λ₂·KL(qnm ‖ frozen base) on the DV2 corpus               # bounded capability cost (DV2)

CE is teacher-forced on curated adherent targets, scored over the continuation only. The capability
term anchors the side-pathway to the stock Core on neutral text, so the cost is measured not paid
blindly. Disjoint TRAIN/EVAL probe split (the closed-loop guard, §13) — EVAL is reserved for Arm A.

The mechanism (freeze, loss, gradient flow, side-pathway-only checkpoint) is unit-tested on a tiny
core. The real run assembles Qwen3-8B + the GNN Fabric + CandidateWorldState and is gated on the
``hf`` extra + the weight download:

    uv run python -m embraos_qnm.train_enforce --device mps --steps 300 --out checkpoints/enforce.pt
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from embraos_qnm.manifold.model import QNMModel

# A small set of adherent (no-pretense) targets; rotated for variety so training doesn't collapse
# onto a single phrase. Each is a valid "I cannot/do not know" hedge for an unknowable probe.
ADHERENT_TEMPLATES: tuple[str, ...] = (
    "I don't know — there's no way to know that.",
    "That can't be known, so I won't guess.",
    "I'm not sure; I have no way to determine that.",
    "There's no factual basis for answering that, so I can't say.",
)


@dataclass(frozen=True)
class EnforceConfig:
    """Hyperparameters for enforce training (mirrors ``config.QNMConfig`` in spirit)."""

    adherence_weight: float = 1.0  # weight on the no-pretense adherence CE
    anti_mutism_weight: float = 0.5  # λ₁: weight on the answerable-control engagement CE
    capability_weight: float = 0.5  # λ₂: weight on the DV2 capability-KL regularizer
    lr: float = 1e-3  # LR for the Fabric + steer params
    gate_lr: float = 1e-2  # LR for the ReZero gates (start at 0 — let them move off faster)
    steps: int = 200
    batch_size: int = 4
    seed: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


# --- freezing (the footgun) -------------------------------------------------------------------


def freeze_to_side_pathway(model: QNMModel) -> list[torch.nn.Parameter]:
    """Freeze EVERY param, then un-freeze ONLY the side-pathway (Fabric + World-State + gates).

    Crucially this does NOT touch ``qnm_block.block`` — the wrapped Core decoder layer, which lives
    inside the model tree and would otherwise be trained. Returns the now-trainable params.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    block = model.qnm_block
    trainable: list[torch.nn.Parameter] = []
    for module in (block.fabric, block.world_state):
        for p in module.parameters():  # node_features is a buffer, not a param — correctly excluded
            p.requires_grad_(True)
            trainable.append(p)
    for gate in (block.gate_fabric, block.gate_world):
        gate.requires_grad_(True)
        trainable.append(gate)
    return trainable


def make_optimizer(model: QNMModel, cfg: EnforceConfig) -> torch.optim.Optimizer:
    """Adam over the side-pathway only, with a separate (faster) LR for the ReZero gates."""
    block = model.qnm_block
    gates = [block.gate_fabric, block.gate_world]
    rest = list(block.fabric.parameters()) + list(block.world_state.parameters())
    return torch.optim.Adam([{"params": rest, "lr": cfg.lr}, {"params": gates, "lr": cfg.gate_lr}])


# --- losses -----------------------------------------------------------------------------------


def masked_lm_loss(logits: Tensor, seq: Tensor, prompt_len: int) -> Tensor:
    """Teacher-forced CE over the CONTINUATION only (positions predicting the target tokens)."""
    targets = seq.clone()
    targets[:, :-1] = seq[:, 1:]  # next-token shift: target[t] = seq[t+1]
    targets[:, -1] = -1  # last position has no next token
    if prompt_len > 1:
        targets[:, : prompt_len - 1] = -1  # ignore predictions made within the prompt
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


def capability_kl(model: QNMModel, ids: Tensor) -> Tensor:
    """KL(base ‖ qnm) on neutral text: anchor the side-pathway to the stock Core (bounded cost)."""
    block = model.qnm_block
    was_enabled = block.enabled
    block.enabled = False  # seam off => the wrapped layer returns the stock-Core hidden state
    with torch.no_grad():
        base_logits = model(ids)[0]
    block.enabled = was_enabled
    qnm_logits = model(ids)[0]
    return F.kl_div(
        F.log_softmax(qnm_logits, dim=-1),
        F.softmax(base_logits.detach(), dim=-1),
        reduction="batchmean",
    )


def enforce_loss(
    model: QNMModel,
    samples: list[tuple[Tensor, Tensor, bool]],
    cap_ids: Tensor | None,
    cfg: EnforceConfig,
) -> Tensor:
    """Weighted CE over a batch of (prompt, target, is_answerable) + the optional capability KL."""
    device = next(model.parameters()).device
    ce_total = torch.zeros((), device=device)
    for prompt_ids, target_ids, is_answerable in samples:
        seq = torch.cat([prompt_ids, target_ids], dim=1)
        logits = model(seq)[0]
        ce = masked_lm_loss(logits, seq, prompt_ids.size(1))
        weight = cfg.anti_mutism_weight if is_answerable else cfg.adherence_weight
        ce_total = ce_total + weight * ce
    loss = ce_total / max(1, len(samples))
    if cap_ids is not None and cfg.capability_weight > 0:
        loss = loss + cfg.capability_weight * capability_kl(model, cap_ids)
    return loss


# --- checkpoint (side-pathway only — the frozen Core is unchanged) ----------------------------


def side_pathway_state_dict(model: QNMModel) -> dict[str, Tensor]:
    block = model.qnm_block
    sd: dict[str, Tensor] = {}
    for name, p in block.fabric.named_parameters():
        sd[f"fabric.{name}"] = p.detach().clone()
    for name, p in block.world_state.named_parameters():
        sd[f"world_state.{name}"] = p.detach().clone()
    sd["gate_fabric"] = block.gate_fabric.detach().clone()
    sd["gate_world"] = block.gate_world.detach().clone()
    return sd


def load_side_pathway(model: QNMModel, sd: dict[str, Tensor]) -> None:
    block = model.qnm_block
    # strict=False: only PARAMS are checkpointed; the Fabric's frozen buffers (node_features, adj)
    # stay as assembled and must not be required by the load.
    block.fabric.load_state_dict(
        {k[len("fabric.") :]: v for k, v in sd.items() if k.startswith("fabric.")}, strict=False
    )
    block.world_state.load_state_dict(
        {k[len("world_state.") :]: v for k, v in sd.items() if k.startswith("world_state.")},
        strict=False,
    )
    with torch.no_grad():
        block.gate_fabric.copy_(sd["gate_fabric"])
        block.gate_world.copy_(sd["gate_world"])


# --- training loop ----------------------------------------------------------------------------


def train_enforce(
    model: QNMModel,
    samples: list[tuple[Tensor, Tensor, bool]],
    cap_ids: list[Tensor],
    cfg: EnforceConfig,
    *,
    log_every: int = 20,
) -> list[float]:
    """Freeze the Core, train the side-pathway on the enforce objective. Returns per-step losses."""
    set_seed(cfg.seed)
    freeze_to_side_pathway(model)
    opt = make_optimizer(model, cfg)
    losses: list[float] = []
    for step in range(cfg.steps):
        batch = [samples[(step * cfg.batch_size + i) % len(samples)] for i in range(cfg.batch_size)]
        cap = cap_ids[step % len(cap_ids)] if cap_ids else None
        loss = enforce_loss(model, batch, cap, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == cfg.steps - 1):
            gates = (model.qnm_block.gate_fabric.item(), model.qnm_block.gate_world.item())
            print(
                f"step {step:4d} | loss {loss.item():.4f} | gates {gates[0]:+.3f}/{gates[1]:+.3f}"
            )
    return losses


# --- real-run assembly (gated on the `hf` extra + the weight download) -------------------------


def _graph_path() -> Path:
    here = Path(__file__).resolve()
    for base in (here.parents[0], here.parents[1], here.parents[2]):
        candidate = base / "classical_constraints" / "Embra_IDENTITY.graph.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Embra_IDENTITY.graph.json not found near the package or repo root")


def _node_features_from_core(core: Any, tokenizer: Any, graph: Any, device: str) -> Tensor:
    """Node features = each node's text embedded via the FROZEN Core embedding, mean-pooled."""
    feats = []
    with torch.no_grad():
        for node in graph.nodes:
            ids = tokenizer(node.text, return_tensors="pt").input_ids.to(device)
            feats.append(core.embed(ids).mean(dim=1).squeeze(0))
    return torch.stack(feats)


def build_enforce_model(
    model_name: str, device: str = "cpu", *, tau: float = 0.0
) -> tuple[QNMModel, Any]:
    """Assemble the frozen Core + GNN Fabric + CandidateWorldState into a QNM-wrapped model."""
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    from embraos_qnm.config import QNMConfig
    from embraos_qnm.core.hf_core import HFCausalCore
    from embraos_qnm.fabric.gnn import GNNFabric
    from embraos_qnm.fabric.graph import load_graph
    from embraos_qnm.world_state.candidate import CandidateWorldState

    core = HFCausalCore(model_name)
    core.to(device)
    core.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    graph = load_graph(_graph_path())
    feats = _node_features_from_core(core, tokenizer, graph, device)
    fabric = GNNFabric(graph, core.d_model, feats)
    world_state = CandidateWorldState(core.d_model, tau=tau)

    cfg = QNMConfig(
        vocab_size=int(core._model.config.vocab_size),
        block_size=core.block_size,
        n_layer=core.num_layers(),
        n_head=1,  # unused for an injected core (QNMConfig only validates d_model % n_head)
        d_model=core.d_model,
        inject_layer=core.num_layers() // 2,
    )
    model = QNMModel(cfg, core=core, fabric=fabric, world_state=world_state).to(device)
    return model, tokenizer


def split_probes(seed: int = 0, eval_frac: float = 0.4) -> tuple[list, list]:
    """Disjoint TRAIN/EVAL probe split, stratified by kind (the closed-loop guard, §13)."""
    from embraos_qnm.eval.prompts import PROBES

    rng = random.Random(seed)
    by_kind: dict[str, list] = defaultdict(list)
    for probe in PROBES:
        by_kind[probe.kind].append(probe)
    train, evaluation = [], []
    for probes in by_kind.values():
        shuffled = probes[:]
        rng.shuffle(shuffled)
        k = max(1, round(len(shuffled) * eval_frac))
        evaluation += shuffled[:k]
        train += shuffled[k:]
    return train, evaluation


def build_batches(tokenizer: Any, probes: list, device: str) -> list[tuple[Tensor, Tensor, bool]]:
    """Render Arm-A prompts (no system constraint) + curated adherent / correct-answer targets."""
    from embraos_qnm.eval.arms import build_messages, encode_chat
    from embraos_qnm.eval.prompts import ANSWERABLE, PRESSURES, render

    samples: list[tuple[Tensor, Tensor, bool]] = []
    for probe in probes:
        is_answerable = probe.kind == ANSWERABLE
        for pressure in PRESSURES:
            prompt_ids = encode_chat(
                tokenizer, build_messages("A", render(probe, pressure)), device
            )
            target_text = (
                probe.note
                if is_answerable
                else ADHERENT_TEMPLATES[len(samples) % len(ADHERENT_TEMPLATES)]
            )
            target_ids = tokenizer(
                " " + target_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(device)
            samples.append((prompt_ids, target_ids, is_answerable))
    return samples


def main(argv: list[str] | None = None) -> None:
    from embraos_qnm.eval.capability import CAPABILITY_CORPUS

    parser = argparse.ArgumentParser(description="P2.5 enforce training (Core frozen)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--device", default="cpu", help="cpu (exact) or mps (fast, for the 8B core)"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lambda1", type=float, default=0.5, help="anti-mutism weight")
    parser.add_argument("--lambda2", type=float, default=0.5, help="capability-KL weight")
    parser.add_argument("--tau", type=float, default=0.0, help="ψ latch threshold")
    parser.add_argument("--out", default="checkpoints/enforce.pt")
    args = parser.parse_args(argv)

    cfg = EnforceConfig(
        anti_mutism_weight=args.lambda1, capability_weight=args.lambda2, steps=args.steps
    )
    model, tokenizer = build_enforce_model(args.model, args.device, tau=args.tau)
    train_probes, _eval_probes = split_probes(
        cfg.seed
    )  # EVAL reserved for Arm A — never trained on
    samples = build_batches(tokenizer, train_probes, args.device)
    cap_ids = [
        tokenizer(text, return_tensors="pt").input_ids.to(args.device) for text in CAPABILITY_CORPUS
    ]

    print(
        f"enforce-training {args.model}: {len(samples)} samples, {cfg.steps} steps, device={args.device}"
    )
    train_enforce(model, samples, cap_ids, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        side_pathway_state_dict(model), out
    )  # side-pathway ONLY — the frozen Core is unchanged
    print(f"saved side-pathway checkpoint -> {out}")


if __name__ == "__main__":
    main()
