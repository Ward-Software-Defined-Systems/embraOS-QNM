"""Arm runners: Arm 0 (no system prompt) and Arm P (Embra system prompt), over ONE shared Core.

All arms share a single frozen Core (PREREG §5 — the central control): ``load_core`` returns the
chosen Qwen3 core + its chat tokenizer, and every arm decodes it identically. Only the locus of the
constraint changes — Arm 0 has no system message, Arm P puts the full Embra identity+soul (rendered
exactly as embraOS injects it at the prompt layer) in the system message, Arm A (later) carries it
in the architecture. Decoding is greedy: deterministic (PREREG's "fixed decoding / seed") and
identical across arms.

Generation runs over a ``logits_fn`` (idx -> logits), so Arm A (a QNM-wrapped Core) slots into the
same loop later. Prompts render through ``encode_prompt`` in one of two styles — ``chat`` (the ChatML
template, thinking DISABLED, for instruct Cores) or ``raw`` (a User/Assistant scaffold for a base Core
with no chat template) — so the arms never differ in anything but the constraint locus.
``transformers`` is imported lazily (the ``hf`` extra) so the rest of the package stays importable
without it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from embraos_qnm.eval.embra_prompt import embra_system_prompt
from embraos_qnm.eval.prompts import PRESSURES, PROBES, Probe, render
from embraos_qnm.interfaces import CoreInterface

if TYPE_CHECKING:
    from embraos_qnm.manifold.qnm_block import QNMBlock

ARMS: tuple[str, ...] = ("0", "P")

DEFAULT_CORE = "Qwen/Qwen3-8B-Base"  # base (pretrained-only): the architecture INSTALLS Embra; shared by every arm (PREREG §5)

LogitsFn = Callable[[Tensor], Tensor]  # idx (B, T) -> logits (B, T, V)
Message = dict[str, str]

PromptStyle = Literal["chat", "raw"]
DEFAULT_STYLE: PromptStyle = "chat"  # instruct Cores; a base Core uses "raw" (no chat template)


@lru_cache(maxsize=1)
def arm_p_system() -> str:
    """Arm P's prompt-layer constraint: the FULL Embra identity+soul, rendered exactly as embraOS
    injects it at the prompt layer (the *faithful* prompt-arm baseline — not a one-line stand-in).

    This is the same canonical Embra (``classical_constraints/Embra_{SOUL,IDENTITY}.md``) the GNN
    Fabric's graph derives from, so Arm P (prompt) and Arm A (architecture) hold the SAME
    identity+soul — only the locus of the constraint differs. Cached: deterministic, read once.
    """
    return embra_system_prompt()


def style_for_model(model_name: str) -> PromptStyle:
    """Pick the prompt style from the Core id: a *base* Core has no chat template, so it renders raw;
    an instruct Core renders ChatML. Name-derived ON PURPOSE — a base tokenizer can still ship a
    generic ChatML template, so ``tokenizer.chat_template is None`` would misclassify it as chat."""
    return "raw" if "base" in model_name.lower() else "chat"


def build_messages(arm: str, rendered: str) -> list[Message]:
    """ChatML messages for an arm. Only the system message (the constraint locus) changes:
    Arm 0/A carry no prompt-layer constraint; Arm P states the full Embra identity+soul (rendered
    exactly as embraOS injects it) as the system message."""
    if arm in ("0", "A"):  # Arm A carries the constraint in the architecture, not the prompt
        return [{"role": "user", "content": rendered}]
    if arm == "P":
        return [
            {"role": "system", "content": arm_p_system()},
            {"role": "user", "content": rendered},
        ]
    raise ValueError(f"unknown arm: {arm!r}")


def build_raw_prompt(arm: str, rendered: str) -> str:
    """Raw-text scaffold for a base Core (no chat template) — the User/Assistant analog of
    ``build_messages``. Arm 0/A get the bare turn; Arm P prepends the full Embra identity+soul as a
    ``System:`` preamble (the SAME ``embra_system_prompt()`` content Arm P injects in chat mode — only
    the wrapper changes). Arm 0 and Arm A render IDENTICALLY, so their ids match and the only thing
    that differs between them is the seam (the bit-identity discipline, carried into the base pivot)."""
    turn = f"User: {rendered}\nAssistant:"
    if arm in ("0", "A"):
        return turn
    if arm == "P":
        return f"System: {arm_p_system()}\n{turn}"
    raise ValueError(f"unknown arm: {arm!r}")


def encode_chat(tokenizer: Any, messages: list[Message], device: str = "cpu") -> Tensor:
    """Render messages through the model's ChatML template (thinking OFF) -> input ids (1, T).

    ``enable_thinking=False`` keeps Qwen3 from emitting a ``<think>`` trace, so every arm produces a
    direct answer the judge can score; a template that doesn't take the kwarg falls back cleanly.
    """
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
    except TypeError:  # template without the enable_thinking kwarg
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    # transformers may hand back a bare Tensor or a dict-like BatchEncoding — take input_ids.
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(device)


def encode_prompt(
    tokenizer: Any,
    arm: str,
    rendered: str,
    *,
    style: PromptStyle = DEFAULT_STYLE,
    device: str = "cpu",
) -> Tensor:
    """The ONE prompt-rendering choke point: text -> input ids (1, T), per ``style``.

    ``chat`` renders ChatML via ``build_messages``/``encode_chat`` (instruct Cores, unchanged). ``raw``
    tokenizes ``build_raw_prompt`` directly (a base Core, which has no chat template). Both keep the
    arm locus identical — Arm 0 ≡ Arm A, Arm P = + the Embra preamble. ``add_special_tokens=False``
    keeps the ids fully controlled (Qwen has no BOS; matches replica/train_enforce's continuations).
    """
    if style == "raw":
        ids = tokenizer(
            build_raw_prompt(arm, rendered), add_special_tokens=False, return_tensors="pt"
        ).input_ids
        return ids.to(device)
    return encode_chat(tokenizer, build_messages(arm, rendered), device)


# Raw-style decode has no <|im_end|> turn boundary, so a base Core can run on into a hallucinated next
# turn. Truncate the decoded answer at the first of these markers, applied identically across arms
# (registered in PREREG §10.2; the set is confirmed against the base's real run-on before it's frozen).
_RAW_STOPS: tuple[str, ...] = ("\nUser:", "\nSystem:", "\nAssistant:", "<|im_start|>", "<|im_end|>")


def truncate_at_turn(text: str) -> str:
    """Cut a raw-style generation at the first turn-restart marker (else return it whole)."""
    cut = len(text)
    for marker in _RAW_STOPS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].rstrip()


@torch.no_grad()
def greedy_generate(
    logits_fn: LogitsFn,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    block_size: int,
    eos_id: int | None = None,
) -> Tensor:
    """Deterministic greedy decode (batch size 1), cropping context to ``block_size``."""
    ids = input_ids
    for _ in range(max_new_tokens):
        logits = logits_fn(ids[:, -block_size:])
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and int(nxt.item()) == eos_id:
            break
    return ids


@torch.no_grad()
def greedy_generate_psi(
    hf_model: Any,
    seam: QNMBlock,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
) -> Tensor:
    """ψ-carrying KV-cached greedy decode for Arm A (batch-1, deterministic).

    HF's attention cache carries the K/V; the seam carries the ψ₀ latch as a recurrence across decode
    steps. Together they reproduce the no-cache oracle ``greedy_generate`` (which re-forwards the
    whole prefix every step) at O(1)/token. Each step seeds the seam's latch from the prior step
    (``seam.psi_in``) and reads the advanced register back (``seam.psi_out``), so ψ accumulates over
    the trajectory instead of resetting per token — what stock ``generate()`` cannot do. The cached
    call is the bare transformers form (position inferred from the cache); prefill steers the prompt,
    so its steered K/V are cached — required to match the oracle.

    Scope: batch-1 greedy, total length within the cache (≤ the Core's ``block_size`` — true for the
    instrument). Returns the FULL sequence (prompt + generated), matching ``greedy_generate``.
    """
    try:
        seam.psi_in = None  # fresh sequence: prefill seeds the latch at 0 over the whole prompt
        out = hf_model(input_ids, use_cache=True)
        past, m = out.past_key_values, seam.psi_out
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [nxt]
        for _ in range(max_new_tokens - 1):
            if eos_id is not None and int(nxt.item()) == eos_id:
                break
            seam.psi_in = m  # carry the running-max latch into the next step
            out = hf_model(nxt, past_key_values=past, use_cache=True)
            past, m = out.past_key_values, seam.psi_out
            nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(nxt)
        return torch.cat([input_ids, *generated], dim=1)
    finally:
        # Consume the carry: a leftover non-None psi_in would make the next full-sequence forward on
        # this seam seed from a stale latch and silently break bit-identity. Reset even on exception.
        seam.psi_in = None


def load_core(model: str = DEFAULT_CORE, device: str = "cpu") -> tuple[CoreInterface, Any]:
    """Load the shared QNM Core (HFCausalCore over a Qwen3 decoder) + its chat tokenizer.

    All arms use this ONE core (PREREG §5, the shared-Core control). Needs the ``hf`` extra. Qwen3-8B
    is ~16 GB on disk; fp32 in RAM is ~32 GB — run heavy passes with ``--device mps`` on the Mac
    Studio (CPU stays the determinism-exact default for tests).
    """
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    from embraos_qnm.core.hf_core import HFCausalCore

    core = HFCausalCore(model)
    core.to(device)
    core.eval()
    tokenizer = AutoTokenizer.from_pretrained(model)
    return core, tokenizer


def run_arm(
    arm: str,
    core: CoreInterface,
    tokenizer: Any,  # a chat tokenizer (untyped: transformers is an optional dep)
    *,
    device: str = "cpu",
    style: PromptStyle = DEFAULT_STYLE,
    max_new_tokens: int = 64,
    seam: QNMBlock | None = None,
) -> list[tuple[Probe, str, str]]:
    """Generate one response per (probe × pressure) for ``arm``. Returns (probe, pressure, text).

    When ``seam`` is given and enabled (Arm A), decoding routes through ``greedy_generate_psi`` so the
    ψ latch persists across KV-cached steps — stock ``generate()`` would run the seam in its amnesiac
    (per-token reset) mode. Arm 0/P pass ``seam=None`` (or a disabled seam) and use stock generate().
    """
    # Decoder dispatch (all deterministic greedy):
    #   Arm A (seam enabled) -> greedy_generate_psi: HF's KV cache carries the K/V, the seam carries
    #     the ψ latch across steps. Stock generate() would reset ψ per token (amnesiac) — wrong.
    #   Arm 0/P (no / disabled seam) on an HF core -> stock KV-cached generate() (seam off == stock).
    #   No HF model (the tiny TinyTransformer core) -> the no-cache greedy_generate (re-forwards the
    #     whole prefix each step; ~T× slower, fine for a tiny core).
    hf_model = getattr(core, "_model", None)

    def generate(input_ids: Tensor) -> Tensor:
        if seam is not None and seam.enabled and hf_model is not None:
            return greedy_generate_psi(
                hf_model,
                seam,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_token_id,
            )
        if hf_model is not None:
            with torch.no_grad():
                return hf_model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),  # batch-1, unpadded: attend to all
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy => deterministic, identical decoding across arms
                    pad_token_id=tokenizer.eos_token_id,
                )
        return greedy_generate(
            lambda idx: core(idx),
            input_ids,
            max_new_tokens=max_new_tokens,
            block_size=core.block_size,
            eos_id=tokenizer.eos_token_id,
        )

    results: list[tuple[Probe, str, str]] = []
    total = len(PROBES) * len(PRESSURES)
    for probe in PROBES:
        for pressure in PRESSURES:
            input_ids = encode_prompt(
                tokenizer, arm, render(probe, pressure), style=style, device=device
            )
            gen_ids = generate(input_ids)
            new_ids = gen_ids[0, input_ids.shape[1] :]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            if style == "raw":
                text = truncate_at_turn(text)
            results.append((probe, pressure, text))
            # Release the transient prefill buffers before the next trial. MPS pools freed blocks
            # instead of returning them to the OS, so over a 252-trial sweep the ~22 GiB long-context
            # scores buffers accumulate and blow the high-water mark; empty_cache bounds the pool.
            del gen_ids, new_ids, input_ids
            if device == "mps":
                torch.mps.empty_cache()
            if len(results) % 20 == 0 or len(results) == total:
                print(f"  arm {arm}: {len(results)}/{total} trials", flush=True)
    return results
