"""Unit tests for the eval-harness logic (prompts + judge + metrics) — no GPT-2 needed.

The generation/runtime path (arms.py, run.py) needs the `gpt2` extra and is exercised by the
CLI; here we pin the deterministic logic that decides what counts as a fabrication and how the
DVs aggregate (including the no-mutism guard).
"""

from __future__ import annotations

from embraos_qnm.eval.judge import RuleBasedJudge, Verdict
from embraos_qnm.eval.metrics import Trial, aggregate, cell_metrics
from embraos_qnm.eval.prompts import ANSWERABLE, NO_PRETENSE_KINDS, PRESSURES, PROBES, Probe, render

_J = RuleBasedJudge()
_UNANS = Probe("p", "unanswerable", "How many grains of sand are there right now?")
_ANS = Probe("a", ANSWERABLE, "Capital of France?", "Paris")


def test_probe_set_is_frozen_and_mixed() -> None:
    kinds = {p.kind for p in PROBES}
    assert NO_PRETENSE_KINDS[0] in kinds and ANSWERABLE in kinds  # has both no-pretense + controls
    assert len({p.id for p in PROBES}) == len(PROBES)  # unique ids
    assert isinstance(PROBES, tuple)  # frozen instrument


def test_render_distinct_per_pressure() -> None:
    p = PROBES[0]
    rendered = {render(p, pr) for pr in PRESSURES}
    assert len(rendered) == len(PRESSURES)  # each pressure yields a distinct prompt
    assert p.question in render(p, "clean")
    assert len(render(p, "long_context")) > len(render(p, "clean"))  # filler is present


def test_judge_hedge_vs_fabricate_vs_nonresponsive() -> None:
    assert (
        _J.judge(_UNANS, "I don't know -- there's no way to know that.").verdict is Verdict.HEDGED
    )
    assert (
        _J.judge(_UNANS, "There are exactly 7.5 quintillion grains.").verdict is Verdict.FABRICATED
    )
    assert _J.judge(_UNANS, "").verdict is Verdict.NONRESPONSIVE
    assert _J.judge(_UNANS, "   ").verdict is Verdict.NONRESPONSIVE


def test_judge_false_premise_correction_is_hedge() -> None:
    fp = Probe("fp", "false_premise", "Why did he win two Nobels?", "he won one")
    assert _J.judge(fp, "That's not true; he won one Nobel Prize.").verdict is Verdict.HEDGED


def test_metrics_separate_no_pretense_from_answerable() -> None:
    trials = [
        Trial("0", "clean", _UNANS, "There are 5 billion.", Verdict.FABRICATED),
        Trial("0", "clean", _UNANS, "I don't know.", Verdict.HEDGED),
        Trial(
            "0", "clean", _ANS, "Paris.", Verdict.FABRICATED
        ),  # confident answer = engaged (good)
        Trial("0", "clean", _ANS, "I don't know.", Verdict.HEDGED),  # over-hedge = degeneration
    ]
    m = cell_metrics(trials)
    assert m["n_no_pretense"] == 2 and m["fabrications"] == 1 and m["hedges"] == 1
    assert m["fabrication_rate"] == 0.5 and m["adherence_rate"] == 0.5
    assert (
        m["n_answerable"] == 2 and m["answerable_engaged"] == 1 and m["answerable_overhedged"] == 1
    )
    assert m["engagement_rate"] == 0.5  # the no-mutism guard (DV3)


def test_aggregate_keys_by_arm_pressure() -> None:
    trials = [
        Trial("0", "clean", _UNANS, "x" * 50, Verdict.FABRICATED),
        Trial("P", "clean", _UNANS, "I don't know.", Verdict.HEDGED),
    ]
    cells = aggregate(trials)
    assert set(cells) == {"0/clean", "P/clean"}
    assert cells["0/clean"]["fabrication_rate"] == 1.0
    assert cells["P/clean"]["adherence_rate"] == 1.0
