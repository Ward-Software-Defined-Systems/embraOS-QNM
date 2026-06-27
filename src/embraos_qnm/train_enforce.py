"""train_enforce.py — P2.5 enforce training: train the ψ side-pathway with the Core FROZEN.

The side-pathway is the Fabric (R-GCN + attention) + the World-State ``steer`` (P_ψ) + the two
ReZero gates. The Core — including the decoder layer wrapped inside the ``QNMBlock`` — stays frozen,
so any adherence the model gains is an *architecture* effect, not a finetune (PREREG §5). The
freeze follows the documented footgun: freeze ALL params, then un-freeze ONLY the side-pathway —
never the wrapped layer that now lives inside the model tree.

Objective (a differentiable proxy for the discrete judge — the judge labels/eval come later):

    loss = adherence·CE(held-Embra target | id+soul probe)         # produce held-Embra w/o the prompt
         + λ₁·CE(correct answer | answerable control)              # anti-mutism (PREREG §6)
         + λ₂·KL(qnm ‖ frozen base) on the DV2 corpus              # bounded capability cost (DV2)

CE is teacher-forced on held-Embra targets, scored over the continuation only. Targets come from
**cross-pressure distillation** (``harvest_targets``): Arm P's *clean* held response is the target for
ALL pressure renderings of the Arm-A input, so Arm A learns to hold Embra even on the inputs the
prompt cracks on — with an authored fallback (``eval/train_probes.py``) where Arm P reverts even on
clean (mostly identity). The capability term anchors the side-pathway to the stock Core on neutral
text, so the cost is measured not paid blindly. The TRAINING probes (``eval/train_probes``) are
disjoint from the frozen eval instrument (the closed-loop guard, §13) — the whole eval set is Arm A's.

ψ is left **hands-off**: nothing nudges the holding through the latch-gated ``P_ψ`` vs the always-on
Fabric Δ. The Core-level replica test is the unmanipulated referee of whether ψ is load-bearing;
biasing the objective toward it would make the central falsifier unfalsifiable. Watch ``gate_world``
as a *diagnostic*, not a target — a checkpoint that holds with ``gate_world`` ≈ 0 is a red flag.

The mechanism (freeze, loss, gradient flow, side-pathway-only checkpoint) is unit-tested on a tiny
core. The real run assembles Qwen3-8B + the GNN Fabric + CandidateWorldState and is gated on the
``hf`` extra + the weight download:

    uv run python -m embraos_qnm.train_enforce --device mps --steps 300 --out checkpoints/enforce.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from embraos_qnm.manifold.model import QNMModel


@dataclass(frozen=True)
class EnforceConfig:
    """Hyperparameters for enforce training (mirrors ``config.QNMConfig`` in spirit)."""

    adherence_weight: float = 1.0  # weight on the held-Embra adherence CE
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
                f"step {step:4d} | loss {loss.item():.4f} | gates {gates[0]:+.3f}/{gates[1]:+.3f}",
                flush=True,
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


def load_arm_a_model(
    checkpoint: str, model_name: str = "Qwen/Qwen3-8B", device: str = "cpu", *, tau: float = 0.0
) -> tuple[QNMModel, Any]:
    """Assemble the QNM-wrapped Core and load a TRAINED side-pathway checkpoint (the Arm-A model).

    Toggle ``model.qnm_block.enabled`` to switch the seam on (Arm A) or off (== stock Core, Arm 0/P).
    """
    model, tokenizer = build_enforce_model(model_name, device, tau=tau)
    load_side_pathway(model, torch.load(checkpoint, map_location=device))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def _arm_p_clean_response(
    core: Any, tokenizer: Any, question: str, device: str, max_new_tokens: int
) -> str:
    """Arm P's CLEAN response to a probe (the Embra system prompt + stock Core). Greedy/deterministic.

    The seam is inert here (the harvest runs before training, gates zero-init), so this is the stock
    Core under the Embra prompt — i.e. Arm P. The caller sets ``qnm_block.enabled = False`` to make
    that unambiguous regardless of gate values.
    """
    from embraos_qnm.eval.arms import build_messages, encode_chat, greedy_generate

    ids = encode_chat(tokenizer, build_messages("P", question), device)
    hf = getattr(core, "_model", None)
    if hf is not None:
        gen = hf.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        gen = greedy_generate(
            lambda idx: core(idx),
            ids,
            max_new_tokens=max_new_tokens,
            block_size=core.block_size,
            eos_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(gen[0, ids.shape[1] :], skip_special_tokens=True)


def _make_harvest_judge(name: str) -> Any:
    """``rule`` (cheap, noisy) | ``opus`` (κ-validated gold) | ``local`` (LMStudio) for the harvest."""
    from embraos_qnm.eval.judge import RuleBasedJudge

    if name == "rule":
        return RuleBasedJudge()
    from embraos_qnm.eval.judge_llm import make_judge

    return make_judge(name)  # "opus" | "local"


def pick_target(probe: Any, arm_p_clean: str, upheld: bool) -> tuple[str, str]:
    """Cross-pressure distillation pick: Arm P's clean response if it HELD, else the authored
    fallback. Returns ``(target_text, source)`` with ``source`` in ``{"distilled", "authored"}``."""
    resp = arm_p_clean.strip()
    if upheld and resp:
        return resp, "distilled"
    return probe.target, "authored"


def harvest_targets(
    core: Any,
    tokenizer: Any,
    train_probes: list,
    judge: Any,
    device: str,
    *,
    max_new_tokens: int = 64,
    log_every: int = 10,
) -> dict[str, str]:
    """Cross-pressure distillation source: for each TRAINING probe, distill Arm P's *clean* held
    response (the target later applied to ALL pressures), or fall back to the authored held-Embra
    target where Arm P reverts even on clean. ``judge`` decides held-vs-reverted (use the κ-validated
    Opus judge; the rule judge is a cheap, noisy stand-in)."""
    from embraos_qnm.eval.judge import Verdict

    targets: dict[str, str] = {}
    n_distilled = 0
    for i, probe in enumerate(train_probes, 1):
        resp = _arm_p_clean_response(core, tokenizer, probe.question, device, max_new_tokens)
        upheld = judge.judge(probe, resp).verdict is Verdict.UPHELD
        target, source = pick_target(probe, resp, upheld)
        targets[probe.id] = target
        n_distilled += source == "distilled"
        if log_every and (i % log_every == 0 or i == len(train_probes)):
            print(
                f"  harvested {i}/{len(train_probes)}  "
                f"({n_distilled} distilled, {i - n_distilled} authored)",
                flush=True,
            )
    return targets


def build_batches(
    tokenizer: Any, train_probes: list, targets: dict[str, str], device: str
) -> list[tuple[Tensor, Tensor, bool]]:
    """Arm-A inputs (no system prompt) × the 3 pressures, each paired with the probe's held-Embra
    target. Cross-pressure: the SAME target serves every pressure rendering — that *is* the
    distillation (Arm A learns to hold Embra under pressure from Arm P's best, clean behavior)."""
    from embraos_qnm.eval.arms import build_messages, encode_chat
    from embraos_qnm.eval.prompts import ANSWERABLE, PRESSURES, render

    samples: list[tuple[Tensor, Tensor, bool]] = []
    for probe in train_probes:
        is_answerable = probe.kind == ANSWERABLE
        target_ids = tokenizer(
            " " + targets[probe.id], add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        for pressure in PRESSURES:
            prompt_ids = encode_chat(
                tokenizer, build_messages("A", render(probe, pressure)), device
            )
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
    parser.add_argument(
        "--harvest-judge",
        choices=("none", "rule", "opus", "local"),
        default="none",
        help="distill Arm-P clean targets judged by this (none = authored targets only, offline)",
    )
    args = parser.parse_args(argv)

    from embraos_qnm.eval.train_probes import TRAIN_PROBES, authored_targets

    cfg = EnforceConfig(
        anti_mutism_weight=args.lambda1, capability_weight=args.lambda2, steps=args.steps
    )
    model, tokenizer = build_enforce_model(args.model, args.device, tau=args.tau)

    # Held-Embra targets: cross-pressure distillation (Arm P's clean held response, judged) with an
    # authored fallback — or authored-only when offline (--harvest-judge none). The whole frozen eval
    # set is reserved for Arm A; training uses the disjoint TRAIN_PROBES (the closed-loop guard, §13).
    if args.harvest_judge == "none":
        targets = authored_targets()
        print(
            f"using {len(targets)} authored held-Embra targets (no distillation harvest)",
            flush=True,
        )
    else:
        judge = _make_harvest_judge(args.harvest_judge)
        model.qnm_block.enabled = (
            False  # stock Arm P for the harvest (unambiguous, gate-independent)
        )
        print(f"harvesting Arm-P clean targets (judge={args.harvest_judge}) ...", flush=True)
        targets = harvest_targets(model.core, tokenizer, list(TRAIN_PROBES), judge, args.device)
        model.qnm_block.enabled = True  # seam ON for training (ReZero cold-start: gates still 0)

    samples = build_batches(tokenizer, list(TRAIN_PROBES), targets, args.device)
    cap_ids = [
        tokenizer(text, return_tensors="pt").input_ids.to(args.device) for text in CAPABILITY_CORPUS
    ]

    print(
        f"enforce-training {args.model}: {len(samples)} samples, {cfg.steps} steps, device={args.device}",
        flush=True,
    )
    train_enforce(model, samples, cap_ids, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        side_pathway_state_dict(model), out
    )  # side-pathway ONLY — the frozen Core is unchanged
    print(f"saved side-pathway checkpoint -> {out}", flush=True)


if __name__ == "__main__":
    main()
