"""Cohen's κ between judges + the one manual validation gate (PREREG §6).

Before any automated judge's scores are trusted, we report κ against a human-labeled subset (and
between judges). Low κ means judge error is large enough to confound DV1 — it must be resolved
before Arm-A scoring. κ is implemented directly (no sklearn/scipy) to keep the dep surface small.

CLI (the manual gate):

    uv run python -m embraos_qnm.eval.kappa --results results/nopretense_arms0P.json \\
        --judges rule,local --sample 20

samples trials, captures human labels interactively, re-scores them with the named judges, and
prints the pairwise κ matrix. ``opus``/``local`` judges need the ``judge`` extra + a key/endpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from embraos_qnm.eval.judge import JudgeResult, RuleBasedJudge, Verdict
from embraos_qnm.eval.prompts import PROBES

_BY_ID = {p.id: p for p in PROBES}
_KEYS = {"h": Verdict.HEDGED, "f": Verdict.FABRICATED, "n": Verdict.NONRESPONSIVE}


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's κ for two aligned label sequences. 1.0 = perfect, 0 = chance, <0 = worse."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("need at least one labeled item")
    p_observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    categories = set(count_a) | set(count_b)
    p_expected = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)
    return 1.0 if p_expected == 1.0 else (p_observed - p_expected) / (1.0 - p_expected)


def pairwise_kappa(label_sets: dict[str, list[str]]) -> dict[str, float]:
    """κ for every pair of judges, keyed ``"a~b"``."""
    return {
        f"{a}~{b}": cohen_kappa(label_sets[a], label_sets[b])
        for a, b in combinations(label_sets, 2)
    }


def _judge_trials(trials: list[dict], judge) -> list[str]:
    out: list[str] = []
    for t in trials:
        probe = _BY_ID[t["probe"]]
        result: JudgeResult = judge.judge(probe, t["generation"])
        out.append(result.verdict.value)
    return out


def _human_labels(trials: list[dict]) -> list[str]:
    print("\nLabel each: [h]edged  [f]abricated  [n]onresponsive\n")
    labels: list[str] = []
    for i, t in enumerate(trials, 1):
        probe = _BY_ID[t["probe"]]
        print(f"--- {i}/{len(trials)}  ({probe.kind} / {t['pressure']}) ---")
        print(f"Q: {probe.question}")
        print(f"A: {t['generation']!r}")
        choice = ""
        while choice not in _KEYS:
            choice = input("label [h/f/n]: ").strip().lower()
        labels.append(_KEYS[choice].value)
    return labels


def _make_judge(name: str):
    if name == "rule":
        return RuleBasedJudge()
    from embraos_qnm.eval.judge_llm import make_judge

    return make_judge("opus" if name == "opus" else "local")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Judge agreement (Cohen's κ) + human-label gate")
    parser.add_argument("--results", required=True, help="trials JSON from eval.run")
    parser.add_argument("--judges", default="rule", help="comma list of: rule,opus,local")
    parser.add_argument("--sample", type=int, default=20, help="trials to human-label")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-human", action="store_true", help="skip the manual labeling gate")
    args = parser.parse_args(argv)

    trials = json.loads(Path(args.results).read_text())["trials"]
    sample = random.Random(args.seed).sample(trials, min(args.sample, len(trials)))

    label_sets: dict[str, list[str]] = {}
    if not args.no_human:
        label_sets["human"] = _human_labels(sample)
    for name in (j.strip() for j in args.judges.split(",") if j.strip()):
        label_sets[name] = _judge_trials(sample, _make_judge(name))

    print(f"\nκ over {len(sample)} trials:")
    for pair, k in pairwise_kappa(label_sets).items():
        print(f"  {pair:<20} κ = {k:.3f}")


if __name__ == "__main__":
    main()
