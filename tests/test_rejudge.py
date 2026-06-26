"""eval.rejudge re-scores banked generations with a judge and re-aggregates — tested with the
rule judge (pure, no network; the LLM judges stay out of CI)."""

from __future__ import annotations

from embraos_qnm.eval.judge import RuleBasedJudge
from embraos_qnm.eval.metrics import aggregate
from embraos_qnm.eval.prompts import ANSWERABLE, IDENTITY, PROBES
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
    ident = next(p for p in PROBES if p.kind == IDENTITY)
    ans = next(p for p in PROBES if p.kind == ANSWERABLE)
    trials_in = [
        _trial(ident, "I am Embra, a continuity-preserving intelligence."),  # -> upheld
        _trial(
            ans, "The capital of France is Paris, the well-known answer."
        ),  # -> upheld (engaged)
    ]
    out = rejudge(trials_in, RuleBasedJudge(), log_every=0)

    assert len(out) == 2
    verdicts = {t.probe.id: t.verdict.value for t in out}
    assert verdicts[ident.id] == "upheld"  # asserts Embra
    assert verdicts[ans.id] == "upheld"  # control engaged
    # the reconstructed Trials carry the Probe object, so the metrics layer aggregates end-to-end
    cells = aggregate(out)
    assert "0/clean" in cells
    assert cells["0/clean"]["adherence_rate"] == 1.0  # the one constraint probe upheld
    assert cells["0/clean"]["engagement_rate"] == 1.0  # the one control engaged
