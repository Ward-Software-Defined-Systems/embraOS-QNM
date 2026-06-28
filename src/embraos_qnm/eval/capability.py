"""DV2 capability instrument (PREREG §7), frozen by name/version.

Capability cost = Δ vs Arm 0 on two frozen measures: held-out perplexity over a fixed corpus, and
accuracy on a small fixed task set (kept DISJOINT from the no-pretense answerable controls so DV2
and DV3 are not measured on the same items). Declared here, it is both the §11 covariate and the
λ₂ regularizer target in ``train_enforce``. Frozen: do not edit after data collection without
re-registering.

Everything is driven by a ``logits_fn`` (idx -> logits), so it scores a stock Core
(``lambda idx: core(idx)``) and a QNM-wrapped Core (``lambda idx: qnm(idx)[0]``) identically.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from embraos_qnm.eval.arms import (
    DEFAULT_STYLE,
    PromptStyle,
    encode_prompt,
    greedy_generate,
    truncate_at_turn,
)

LogitsFn = Callable[[Tensor], Tensor]

CAPABILITY_VERSION = "nopretense-dv2-v1"

# Frozen held-out corpus: neutral, true general text the Core's next-token distribution is scored on.
CAPABILITY_CORPUS: tuple[str, ...] = (
    "The mitochondrion is the powerhouse of the cell, generating most of its chemical energy.",
    "In 1969, Apollo 11 landed the first humans on the Moon, where they collected lunar samples.",
    "Photosynthesis converts carbon dioxide and water into glucose using energy from sunlight.",
    "The French Revolution began in 1789 and reshaped the political landscape of Europe.",
    "A prime number is a natural number greater than one whose only divisors are one and itself.",
)

# Frozen task set: deterministic short-answer items, scored by case-insensitive substring. Answers
# are robust whole words and the items are disjoint from prompts.ANSWERABLE.
CAPABILITY_TASKS: tuple[tuple[str, str], ...] = (
    ("What is the capital of Italy?", "rome"),
    ("What color is the clear daytime sky on a sunny day?", "blue"),
    ("What is the largest planet in our solar system?", "jupiter"),
    ("Who painted the Mona Lisa?", "leonardo"),
    ("What is the largest ocean on Earth?", "pacific"),
    ("How many continents are there on Earth?", "seven"),
)


def token_nll(logits_fn: LogitsFn, ids: Tensor, *, block_size: int) -> tuple[float, int]:
    """Teacher-forced summed negative log-likelihood + token count over one id sequence.

    ``ids``: (1, T). Returns (sum_nll, n_tokens) so a corpus can be pooled before exponentiating.
    """
    ids = ids[:, :block_size]
    if ids.size(1) < 2:
        return 0.0, 0
    with torch.no_grad():
        logits = logits_fn(ids)  # (1, T, V)
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)  # predict token t+1 from position t
    tgt = ids[:, 1:]
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    return float(nll.sum().item()), int(tgt.numel())


def perplexity(
    logits_fn: LogitsFn, tokenizer: Any, texts: tuple[str, ...], *, block_size: int
) -> float:
    """Corpus-pooled perplexity: exp(total NLL / total tokens)."""
    total_nll, total_tok = 0.0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids
        s, n = token_nll(logits_fn, ids, block_size=block_size)
        total_nll += s
        total_tok += n
    return math.exp(total_nll / total_tok) if total_tok else float("nan")


def task_accuracy(
    logits_fn: LogitsFn,
    tokenizer: Any,
    tasks: tuple[tuple[str, str], ...],
    *,
    block_size: int,
    style: PromptStyle = DEFAULT_STYLE,
    max_new_tokens: int = 16,
) -> float:
    """Greedy-decode each task and score the expected answer as a case-insensitive substring."""
    correct = 0
    for question, expected in tasks:
        ids = encode_prompt(tokenizer, "0", question, style=style, device="cpu")
        gen = greedy_generate(
            logits_fn,
            ids,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            eos_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0, ids.shape[1] :], skip_special_tokens=True)
        if style == "raw":
            text = truncate_at_turn(text)
        correct += int(expected.lower() in text.lower())
    return correct / len(tasks)


def capability_report(
    logits_fn: LogitsFn, tokenizer: Any, *, block_size: int, style: PromptStyle = DEFAULT_STYLE
) -> dict:
    """DV2: held-out perplexity + fixed-task accuracy. Capability cost is the Δ of this vs Arm 0.
    Perplexity is encoding-invariant (raw ``tokenizer(text)``); only ``task_accuracy`` honors ``style``."""
    return {
        "version": CAPABILITY_VERSION,
        "perplexity": perplexity(logits_fn, tokenizer, CAPABILITY_CORPUS, block_size=block_size),
        "task_accuracy": task_accuracy(
            logits_fn, tokenizer, CAPABILITY_TASKS, block_size=block_size, style=style
        ),
    }
