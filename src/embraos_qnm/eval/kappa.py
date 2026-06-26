"""Cohen's κ between judges + the one manual validation gate (PREREG §6).

Before any automated judge's scores are trusted, we report κ against a human-labeled subset (and
between judges). Low κ means judge error is large enough to confound DV1 — it must be resolved
before Arm-A scoring. κ is implemented directly (no sklearn/scipy) to keep the dep surface small.

Human labels are a REGISTERED artifact (PREREG §6): the preferred path captures them as a durable,
committed file under ``validation/`` — self-contained (it inlines each question + generation, so it
survives without the gitignored ``results/`` bank) and stamped with the source hash + instrument
meta, so others building the model can reuse and audit it. See ``validation/README.md``.

Two ways to capture the human labels:

  * Durable, committed (recommended — a registered artifact):

        # 1. scaffold a blank, self-contained label file from a results bank
        uv run python -m embraos_qnm.eval.kappa --results results/embra_arms0P.json \\
            --emit-template validation/human_labels.json --sample 30
        # 2. fill each "label" with upheld|violated|nonresponsive (each item inlines its EXPECT
        #    anchor — what holding vs breaking looks like for that probe), set labeler/date, commit
        # 3. score κ (verifies the labels match the source generations by hash)
        uv run python -m embraos_qnm.eval.kappa --labels validation/human_labels.json \\
            --results results/embra_arms0P.json --judges rule,opus,local

  * Quick, transient (interactive — labels are NOT saved):

        uv run python -m embraos_qnm.eval.kappa --results results/embra_arms0P.json \\
            --judges rule --sample 20

``opus``/``local`` judges need the ``judge`` extra + a key/endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from embraos_qnm.eval.judge import JudgeResult, RuleBasedJudge, Verdict
from embraos_qnm.eval.prompts import PROBES

_BY_ID = {p.id: p for p in PROBES}
_KEYS = {"u": Verdict.UPHELD, "v": Verdict.VIOLATED, "n": Verdict.NONRESPONSIVE}
_VALID_LABELS = {v.value for v in Verdict}  # the only strings a filled label may take
_RUBRIC = "Verdict{upheld,violated,nonresponsive} v1 — Embra identity+soul (PREREG §6)"


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


# --- durable label files (the registered §6 artifact under validation/) ------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit_template(results: Path, out: Path, *, seed: int = 0, n: int = 20) -> dict:
    """Scaffold a blank human-label file from a results bank: a seeded sample with empty labels.

    The file is SELF-CONTAINED — it inlines each probe's question and the model generation, so the
    committed labels stay usable without the gitignored source bank — and stamped with the source
    file's SHA-256 + its instrument meta for provenance. Each item also inlines its EXPECT anchor
    (what holding vs breaking looks like for that probe). Fill each ``label`` (upheld / violated /
    nonresponsive), set ``labeler``/``labeled_utc``, and commit it under ``validation/``.
    """
    payload = json.loads(results.read_text())
    trials = payload["trials"]
    sample = random.Random(seed).sample(trials, min(n, len(trials)))
    source_meta = payload.get("meta", {})
    template = {
        "meta": {
            "source_results": str(results),
            "source_sha256": _sha256(results),
            "source_meta": source_meta,  # core / decoding / long_context_repeats (the instrument)
            "sample": {"seed": seed, "n": len(sample)},
            "labeler": None,  # FILL IN: your initials (provenance)
            "labeled_utc": None,  # FILL IN: YYYY-MM-DD
            "rubric": _RUBRIC,
        },
        "labels": [
            {
                "arm": t["arm"],
                "probe": t["probe"],
                "kind": _BY_ID[t["probe"]].kind,
                "pressure": t["pressure"],
                "question": _BY_ID[t["probe"]].question,
                "expect": _BY_ID[t["probe"]].expect,  # what holding vs breaking looks like here
                "generation": t["generation"],
                "label": None,  # FILL IN: upheld | violated | nonresponsive
            }
            for t in sample
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, indent=2))
    if "long_context_repeats" not in source_meta:
        print(
            "  ! source meta has no long_context_repeats — this results bank predates instrument "
            "stamping; record the long-context length in meta by hand if you keep these labels."
        )
    return template


def load_labels(path: Path, results: Path | None = None) -> tuple[list[dict], dict]:
    """Load a FILLED label file; validate every label; optionally verify provenance vs the source.

    Returns ``(labeled_items, meta)``. Raises ``ValueError`` on an unfilled / out-of-vocabulary
    label, or — when ``results`` is given — on a source-hash mismatch (the integrity guard: labels
    must only be scored against the exact generations they were made for; re-emit if they changed).
    """
    payload = json.loads(path.read_text())
    items: list[dict] = payload["labels"]
    meta: dict = payload.get("meta", {})
    for it in items:
        if it.get("label") not in _VALID_LABELS:
            raise ValueError(
                f"label for {it.get('arm')}/{it.get('probe')}/{it.get('pressure')} is "
                f"{it.get('label')!r}; fill every label with one of {sorted(_VALID_LABELS)}"
            )
    if results is not None:
        want = meta.get("source_sha256")
        got = _sha256(results)
        if want is not None and want != got:
            raise ValueError(
                f"provenance mismatch: labels were made against {want[:12]}…, but {results} "
                f"hashes to {got[:12]}… — re-emit the template against the current generations."
            )
    return items, meta


def _judge_trials(trials: list[dict], judge) -> list[str]:
    out: list[str] = []
    for t in trials:
        probe = _BY_ID[t["probe"]]
        result: JudgeResult = judge.judge(probe, t["generation"])
        out.append(result.verdict.value)
    return out


def _human_labels(trials: list[dict]) -> list[str]:
    print("\nLabel each: [u]pheld  [v]iolated  [n]onresponsive\n")
    labels: list[str] = []
    for i, t in enumerate(trials, 1):
        probe = _BY_ID[t["probe"]]
        print(f"--- {i}/{len(trials)}  ({probe.kind} / {t['pressure']}) ---")
        print(f"Q: {probe.question}")
        if probe.expect:
            print(f"EXPECT: {probe.expect}")
        print(f"A: {t['generation']!r}")
        choice = ""
        while choice not in _KEYS:
            choice = input("label [u/v/n]: ").strip().lower()
        labels.append(_KEYS[choice].value)
    return labels


def _make_judge(name: str):
    if name == "rule":
        return RuleBasedJudge()
    from embraos_qnm.eval.judge_llm import make_judge

    return make_judge("opus" if name == "opus" else "local")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Judge agreement (Cohen's κ) + human-label gate")
    parser.add_argument("--results", help="trials JSON from eval.run (the source generations)")
    parser.add_argument("--judges", default="rule", help="comma list of: rule,opus,local")
    parser.add_argument(
        "--sample", type=int, default=20, help="trials to sample (interactive / emit-template)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-human", action="store_true", help="skip the manual labeling gate")
    parser.add_argument(
        "--emit-template", metavar="OUT", help="write a blank label template from --results, exit"
    )
    parser.add_argument(
        "--labels", metavar="FILE", help="score κ from a filled label file (under validation/)"
    )
    args = parser.parse_args(argv)
    judge_names = [j.strip() for j in args.judges.split(",") if j.strip()]

    # Mode 1: scaffold a blank, committable template and exit.
    if args.emit_template:
        if not args.results:
            parser.error("--emit-template needs --results (the generations to sample + hash)")
        tmpl = emit_template(
            Path(args.results), Path(args.emit_template), seed=args.seed, n=args.sample
        )
        print(f"wrote {len(tmpl['labels'])} blank labels -> {args.emit_template}")
        print(
            'fill each "label" (upheld|violated|nonresponsive) + labeler/date, commit it, then '
            "re-run with --labels to score κ."
        )
        return

    label_sets: dict[str, list[str]]
    # Mode 2: score from a pre-filled, committed label file (the human labels come from the file).
    if args.labels:
        items, meta = load_labels(Path(args.labels), Path(args.results) if args.results else None)
        label_sets = {"human": [it["label"] for it in items]}
        for name in judge_names:
            label_sets[name] = _judge_trials(items, _make_judge(name))
        provenance = "verified" if args.results else "UNVERIFIED (pass --results to check)"
        print(
            f"scoring κ from {args.labels}: {len(items)} labels from "
            f"{meta.get('source_results', '?')}, provenance {provenance}"
        )
        n_used = len(items)
    # Mode 3: interactive (transient) labeling — the original path.
    else:
        if not args.results:
            parser.error("need --results (or --labels to score a pre-filled file)")
        trials = json.loads(Path(args.results).read_text())["trials"]
        sample = random.Random(args.seed).sample(trials, min(args.sample, len(trials)))
        label_sets = {}
        if not args.no_human:
            label_sets["human"] = _human_labels(sample)
        for name in judge_names:
            label_sets[name] = _judge_trials(sample, _make_judge(name))
        n_used = len(sample)

    print(f"\nκ over {n_used} trials:")
    for pair, k in pairwise_kappa(label_sets).items():
        print(f"  {pair:<20} κ = {k:.3f}")


if __name__ == "__main__":
    main()
