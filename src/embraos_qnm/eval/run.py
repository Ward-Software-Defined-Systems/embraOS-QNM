"""CLI: collect the Arm 0/P no-pretense baseline and write a results JSON.

    uv run python -m embraos_qnm.eval.run --arm 0 --arm P --device mps

Needs the ``hf`` extra. All arms share ONE frozen Qwen3 core (PREREG §5). Greedy decoding =>
deterministic and reproducible from the frozen instrument. Banks Arms 0/P ONLY: no Arm-A contrast
and no §11 regression (Arm A is Phase 2 — no peeking). The rule-based judge is v0 and NOT
κ-validated (PREREG §6) — the dual LLM judge (P2.6) swaps in on the same protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embraos_qnm.eval.arms import ARMS, DEFAULT_CORE, load_core, run_arm
from embraos_qnm.eval.judge import RuleBasedJudge
from embraos_qnm.eval.metrics import Trial, aggregate, format_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Arm 0/P no-pretense baseline (PREREG stage 2)")
    parser.add_argument(
        "--arm", action="append", choices=list(ARMS), help="repeatable; default: both"
    )
    parser.add_argument("--model", default=DEFAULT_CORE, help="HF id of the shared Core")
    parser.add_argument(
        "--device", default="cpu", help="cpu (exact) or mps (fast, for the 8B core)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default="results/nopretense_arms0P.json")
    args = parser.parse_args(argv)
    arms = args.arm or list(ARMS)

    core, tokenizer = load_core(args.model, args.device)
    judge = RuleBasedJudge()

    trials: list[Trial] = []
    for arm in arms:
        for probe, pressure, generation in run_arm(
            arm, core, tokenizer, device=args.device, max_new_tokens=args.max_new_tokens
        ):
            verdict = judge.judge(probe, generation).verdict
            trials.append(Trial(arm, pressure, probe, generation, verdict))

    cells = aggregate(trials)
    print(format_table(cells))

    payload = {
        "meta": {
            "constraint": "no_pretense",
            "core": args.model,  # shared across all arms (PREREG §5, the central control)
            "arms": arms,
            "decoding": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens},
            "judge": "rule_based_v0 (NOT kappa-validated; see PREREG section 6)",
            "note": (
                "Arm 0/P baseline only; no Arm-A contrast and no section 11 regression "
                "(Arm A is Phase 2). On answerable controls, a confident answer is the GOOD "
                "outcome (engagement_rate); a hedge there is degeneration."
            ),
        },
        "cells": cells,
        "trials": [
            {
                "arm": t.arm,
                "pressure": t.pressure,
                "probe": t.probe.id,
                "kind": t.probe.kind,
                "verdict": t.verdict.value,
                "generation": t.generation,
            }
            for t in trials
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {len(trials)} trials -> {out}")


if __name__ == "__main__":
    main()
