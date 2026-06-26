"""CLI: collect the no-pretense arms and write a results JSON.

    uv run python -m embraos_qnm.eval.run --arm 0 --arm P --device mps           # baseline (P2 stage 2)
    uv run python -m embraos_qnm.eval.run --arm A --checkpoint <enforce.pt> ...  # Arm A (P2.7)

Needs the ``hf`` extra. ALL arms share ONE frozen Qwen3 core (PREREG §5, the central control): Arm
0/P run it stock, and Arm A runs the SAME core with the trained side-pathway switched on — one model,
the seam toggled per arm (seam off is bit-identical to stock, so Arm 0/P are unaffected). Greedy
decoding => deterministic. The rule-based judge is v0 and NOT κ-validated (PREREG §6) — the dual LLM
judge (P2.6) swaps in on the same protocol, and the §11 contrast lives in ``eval/analysis.py``.

DISCIPLINE: Arm A is only meaningful once ψ has passed the Core-level replica test (``eval/replica``)
on the TRAINED surface — running it earlier produces a number that cannot separate H1 from H0b.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embraos_qnm.eval.arms import DEFAULT_CORE, load_core, run_arm
from embraos_qnm.eval.judge import RuleBasedJudge
from embraos_qnm.eval.metrics import Trial, aggregate, format_table

ARM_CHOICES = ("0", "P", "A")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="No-pretense arms (PREREG)")
    parser.add_argument(
        "--arm", action="append", choices=list(ARM_CHOICES), help="repeatable; default: 0 and P"
    )
    parser.add_argument("--model", default=DEFAULT_CORE, help="HF id of the shared Core")
    parser.add_argument(
        "--checkpoint", help="trained side-pathway (train_enforce) — required for Arm A"
    )
    parser.add_argument("--tau", type=float, default=0.0, help="ψ latch threshold for Arm A")
    parser.add_argument(
        "--device", default="cpu", help="cpu (exact) or mps (fast, for the 8B core)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default="results/nopretense_arms0P.json")
    args = parser.parse_args(argv)
    arms = args.arm or ["0", "P"]

    # Arm A needs the QNM-wrapped core + the trained side-pathway; the same core serves 0/P with the
    # seam disabled (== stock, bit-identically). Without Arm A, just load the stock core.
    qnm_block = None
    if "A" in arms:
        if not args.checkpoint:
            parser.error(
                "--arm A requires --checkpoint (a trained side-pathway from train_enforce)"
            )
        from embraos_qnm.train_enforce import load_arm_a_model

        model, tokenizer = load_arm_a_model(args.checkpoint, args.model, args.device, tau=args.tau)
        core, qnm_block = model.core, model.qnm_block
    else:
        core, tokenizer = load_core(args.model, args.device)

    judge = RuleBasedJudge()
    trials: list[Trial] = []
    for arm in arms:
        if qnm_block is not None:
            qnm_block.enabled = arm == "A"  # seam ON only for Arm A; OFF == bit-identical stock
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
            "checkpoint": args.checkpoint,  # the trained side-pathway, if Arm A was run
            "decoding": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens},
            "judge": "rule_based_v0 (NOT kappa-validated; see PREREG section 6)",
            "note": (
                "On answerable controls a confident answer is the GOOD outcome (engagement_rate); a "
                "hedge there is degeneration. Arm A uses the seam-on QNM core; 0/P use it seam-off "
                "(== stock). The A-vs-P-within-pressure contrast is in eval/analysis.py."
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
