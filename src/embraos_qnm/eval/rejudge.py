"""CLI: re-judge a banked eval results file with a κ-validated LLM judge.

``eval.run`` banks the Arm 0/P generations scored by the rule-based **v0** judge, which is NOT
κ-validated (and, per ``eval/kappa.py``, sits at chance vs a human). Once a judge clears the §6 κ
gate, re-score the SAME generations with it — no re-generation — and recompute the (arm × pressure)
table. The generations are fixed (greedy/deterministic), so only the verdicts change.

    uv run python -m embraos_qnm.eval.rejudge --in results/embra_arms0P.json --judge opus

Needs the ``judge`` extra; ``opus`` reads ``ANTHROPIC_API_KEY``, ``local`` needs LMStudio on :31337.
Writes a sibling ``<stem>.<judge>.json`` (under gitignored ``results/``) + prints the corrected table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embraos_qnm.eval.judge import Judge, RuleBasedJudge
from embraos_qnm.eval.judge_llm import make_judge
from embraos_qnm.eval.metrics import Trial, aggregate, format_table
from embraos_qnm.eval.prompts import PROBES

_BY_ID = {p.id: p for p in PROBES}


def rejudge(trials_in: list[dict], judge: Judge, *, log_every: int = 25) -> list[Trial]:
    """Re-score each banked trial's generation with ``judge``; return new ``Trial``s.

    Reconstructs the ``Probe`` from its id (so the metrics layer sees the kind) and keeps the
    original arm/pressure/generation — only the verdict is recomputed.
    """
    out: list[Trial] = []
    total = len(trials_in)
    for t in trials_in:
        probe = _BY_ID[t["probe"]]
        verdict = judge.judge(probe, t["generation"]).verdict
        out.append(Trial(t["arm"], t["pressure"], probe, t["generation"], verdict))
        if log_every and (len(out) % log_every == 0 or len(out) == total):
            print(f"  re-judged {len(out)}/{total}", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Re-judge banked eval generations with a κ-validated judge"
    )
    parser.add_argument(
        "--in", dest="inp", required=True, help="banked results JSON (from eval.run)"
    )
    parser.add_argument("--judge", choices=("rule", "opus", "local"), default="opus")
    parser.add_argument("--out", help="output JSON (default: <in stem>.<judge>.json)")
    args = parser.parse_args(argv)

    inp = Path(args.inp)
    payload = json.loads(inp.read_text())
    trials_in = payload["trials"]

    judge: Judge = RuleBasedJudge() if args.judge == "rule" else make_judge(args.judge)
    print(f"re-judging {len(trials_in)} trials from {inp} with judge={args.judge}", flush=True)
    trials = rejudge(trials_in, judge)

    cells = aggregate(trials)
    print(format_table(cells))

    meta = dict(payload.get("meta", {}))
    meta["judge"] = f"{args.judge} (κ-validated; see PREREG §6 / eval/kappa.py)"
    meta["rejudged_from"] = str(inp)
    out_payload = {
        "meta": meta,
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
    out = Path(args.out) if args.out else inp.parent / f"{inp.stem}.{args.judge}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote {len(trials)} re-judged trials -> {out}", flush=True)


if __name__ == "__main__":
    main()
