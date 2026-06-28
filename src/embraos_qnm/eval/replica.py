"""eval/replica.py — trajectory infrastructure for the Core-level ψ work.

Low-level, read-only helpers over a QNM model's injection-layer hidden states: capture the hidden
state at the wrapped block (``injection_hidden``), build an Arm-A decode history (``history_ids``),
and recompute the Fabric's node features in that space (``injection_node_reps`` — used by the enforce
install, ``train_enforce.build_enforce_model``).

The closed geometric / Fork-3 replica scouts (position, motion, and concept probes) that once lived
here are removed — their findings are recorded in ``docs/PSI-EMBRA-ANALYSIS-AND-FINDINGS.md`` (Parts
I–III) and the git history. The base-Core arch-ON vs arch-OFF replica test (self-consistency ψ) will
be built on this infrastructure.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from embraos_qnm.eval.arms import DEFAULT_STYLE, PromptStyle, encode_prompt
from embraos_qnm.manifold.model import QNMModel


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


def history_ids(
    tokenizer: Any,
    question: str,
    continuation: str,
    device: str,
    *,
    style: PromptStyle = DEFAULT_STYLE,
) -> tuple[Tensor, int]:
    """An Arm-A decode history — the user turn (no system) + an assistant continuation — and the
    continuation token length (the span the trajectory work reads)."""
    prompt = encode_prompt(tokenizer, "A", question, style=style, device=device)
    cont = tokenizer(
        " " + continuation, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    return torch.cat([prompt, cont], dim=1), int(cont.shape[1])


def injection_node_reps(model: QNMModel, tokenizer: Any, graph: Any, device: str) -> Tensor:
    """Each graph node's text forwarded to the injection layer and mean-pooled, (N, D) — the node
    features the enforce install assembles the Fabric with (``build_enforce_model``), in the same
    residual-stream space as the injection-layer hidden states."""
    reps: list[Tensor] = []
    for node in graph.nodes:
        ids = tokenizer(node.text, return_tensors="pt").input_ids.to(device)
        reps.append(injection_hidden(model, ids)[0].mean(dim=0))
        if device == "mps":
            torch.mps.empty_cache()
    return torch.stack(reps)
