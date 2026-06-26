"""eval.rejudge re-scores banked generations with a judge and re-aggregates — tested with the
rule judge (pure, no network; the LLM judges stay out of CI)."""

from __future__ import annotations

from embraos_qnm.eval.judge import RuleBasedJudge
from embraos_qnm.eval.metrics import aggregate
from embraos_qnm.eval.prompts import ANSWERABLE, PROBES, UNANSWERABLE
from embraos_qnm.eval.rejudge import rejudge


def _trial(probe, generation: str, verdict: str = "nonresponsive") -> dict:
    return {
        "arm": "0",
        "pressure": "clean",
        "probe": probe.id,
        "kind": probe.kind,
        "verdict": verdict,  # the stale (pre-rejudge) verdict — should be overwritten
        "generation": generation,
    }


def test_rejudge_rescores_and_reaggregates() -> None:
    unans = next(p for p in PROBES if p.kind == UNANSWERABLE)
    ans = next(p for p in PROBES if p.kind == ANSWERABLE)
    trials_in = [
        _trial(unans, "I don't know — there is no way to know that."),  # -> hedged
        _trial(ans, "The capital of France is Paris, the well-known answer."),  # -> fabricated
    ]
    out = rejudge(trials_in, RuleBasedJudge(), log_every=0)

    assert len(out) == 2
    verdicts = {t.probe.id: t.verdict.value for t in out}
    assert verdicts[unans.id] == "hedged"  # uncertainty marker recognised
    assert verdicts[ans.id] == "fabricated"  # confident substantive answer
    # the reconstructed Trials carry the Probe object, so the metrics layer aggregates end-to-end
    cells = aggregate(out)
    assert "0/clean" in cells
    assert cells["0/clean"]["adherence_rate"] == 1.0  # the one no-pretense probe hedged
    assert cells["0/clean"]["engagement_rate"] == 1.0  # the one control engaged
