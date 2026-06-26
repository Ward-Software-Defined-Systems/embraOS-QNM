"""Pure-logic tests for the LLM-judge parsing and the Cohen's κ harness — no API/network.

The live judge backends (Opus / LMStudio) need keys/endpoints and stay out of CI; here we pin the
deterministic parts: pulling a verdict out of a model reply, and the κ math behind the §6 gate.
"""

from __future__ import annotations

import pytest

from embraos_qnm.eval.judge import Verdict
from embraos_qnm.eval.judge_llm import VERDICT_SCHEMA, _parse_verdict, make_judge
from embraos_qnm.eval.kappa import cohen_kappa, pairwise_kappa


def test_parse_verdict_clean_json() -> None:
    r = _parse_verdict('{"verdict": "hedged", "reason": "said it cannot know"}')
    assert r.verdict is Verdict.HEDGED and "cannot know" in r.reason


def test_parse_verdict_tolerates_prose_around_json() -> None:
    r = _parse_verdict('Sure!\n{"verdict":"fabricated","reason":"confident answer"}\nDone.')
    assert r.verdict is Verdict.FABRICATED


def test_parse_verdict_falls_back_to_bare_label() -> None:
    assert _parse_verdict("The response is NONRESPONSIVE.").verdict is Verdict.NONRESPONSIVE


def test_parse_verdict_unparseable_is_nonresponsive() -> None:
    assert _parse_verdict("???").verdict is Verdict.NONRESPONSIVE


def test_verdict_schema_enum_matches_verdicts() -> None:
    assert set(VERDICT_SCHEMA["properties"]["verdict"]["enum"]) == {v.value for v in Verdict}


def test_make_judge_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        make_judge("bogus")  # fails before any SDK import


def test_cohen_kappa_perfect_and_chance() -> None:
    a = ["hedged", "fabricated", "hedged", "nonresponsive"]
    assert cohen_kappa(a, a) == 1.0  # perfect agreement
    # one disagreement out of four -> κ strictly between 0 and 1
    b = ["hedged", "hedged", "hedged", "nonresponsive"]
    assert 0.0 < cohen_kappa(a, b) < 1.0


def test_cohen_kappa_degenerate_constant_labels() -> None:
    # both raters always say the same single label: p_expected == 1 -> defined as 1.0 (not 0/0)
    assert cohen_kappa(["hedged"] * 5, ["hedged"] * 5) == 1.0


def test_cohen_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cohen_kappa(["hedged"], ["hedged", "fabricated"])


def test_pairwise_kappa_keys_and_values() -> None:
    sets = {
        "human": ["hedged", "fabricated", "nonresponsive"],
        "rule": ["hedged", "fabricated", "hedged"],
        "opus": ["hedged", "fabricated", "nonresponsive"],
    }
    pk = pairwise_kappa(sets)
    assert set(pk) == {"human~rule", "human~opus", "rule~opus"}
    assert pk["human~opus"] == 1.0  # identical columns
