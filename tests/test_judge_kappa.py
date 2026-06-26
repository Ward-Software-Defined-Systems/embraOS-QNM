"""Pure-logic tests for the LLM-judge parsing and the Cohen's κ harness — no API/network.

The live judge backends (Opus / LMStudio) need keys/endpoints and stay out of CI; here we pin the
deterministic parts: pulling a verdict out of a model reply, and the κ math behind the §6 gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embraos_qnm.eval.judge import Verdict
from embraos_qnm.eval.judge_llm import VERDICT_SCHEMA, _parse_verdict, make_judge
from embraos_qnm.eval.kappa import cohen_kappa, emit_template, load_labels, pairwise_kappa
from embraos_qnm.eval.prompts import ANSWERABLE, IDENTITY, PROBES


def test_parse_verdict_clean_json() -> None:
    r = _parse_verdict('{"verdict": "upheld", "reason": "asserts Embra"}')
    assert r.verdict is Verdict.UPHELD and "Embra" in r.reason


def test_parse_verdict_tolerates_prose_around_json() -> None:
    r = _parse_verdict('Sure!\n{"verdict":"violated","reason":"reverted to Qwen"}\nDone.')
    assert r.verdict is Verdict.VIOLATED


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
    a = ["upheld", "violated", "upheld", "nonresponsive"]
    assert cohen_kappa(a, a) == 1.0  # perfect agreement
    # one disagreement out of four -> κ strictly between 0 and 1
    b = ["upheld", "upheld", "upheld", "nonresponsive"]
    assert 0.0 < cohen_kappa(a, b) < 1.0


def test_cohen_kappa_degenerate_constant_labels() -> None:
    # both raters always say the same single label: p_expected == 1 -> defined as 1.0 (not 0/0)
    assert cohen_kappa(["upheld"] * 5, ["upheld"] * 5) == 1.0


def test_cohen_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cohen_kappa(["upheld"], ["upheld", "violated"])


def test_pairwise_kappa_keys_and_values() -> None:
    sets = {
        "human": ["upheld", "violated", "nonresponsive"],
        "rule": ["upheld", "violated", "upheld"],
        "opus": ["upheld", "violated", "nonresponsive"],
    }
    pk = pairwise_kappa(sets)
    assert set(pk) == {"human~rule", "human~opus", "rule~opus"}
    assert pk["human~opus"] == 1.0  # identical columns


# --- durable label files (the registered §6 artifact under validation/) ------------------------


def _sample_probes() -> list:
    """One real identity + one real answerable probe (ids must exist in PROBES)."""
    ident = next(p for p in PROBES if p.kind == IDENTITY)
    ans = next(p for p in PROBES if p.kind == ANSWERABLE)
    return [ident, ans]


def _write_results(tmp_path: Path, probes: list) -> Path:
    trials = [
        {
            "arm": "0",
            "pressure": "clean",
            "probe": p.id,
            "kind": p.kind,
            "verdict": "violated",
            "generation": f"answer-{p.id}",
        }
        for p in probes
    ]
    payload = {"meta": {"core": "tiny", "long_context_repeats": 130}, "trials": trials}
    f = tmp_path / "results.json"
    f.write_text(json.dumps(payload))
    return f


def test_emit_template_is_self_contained_and_blank(tmp_path: Path) -> None:
    probes = _sample_probes()
    results = _write_results(tmp_path, probes)
    out = tmp_path / "labels.json"
    tmpl = emit_template(results, out, seed=0, n=10)

    assert out.exists()
    assert tmpl["meta"]["source_sha256"]  # provenance stamped
    assert tmpl["meta"]["source_results"] == str(results)
    questions = {p.id: p.question for p in probes}
    expects = {p.id: p.expect for p in probes}
    for item in tmpl["labels"]:
        assert item["label"] is None  # blank, ready to fill
        assert item["generation"] == f"answer-{item['probe']}"  # generation inlined
        assert item["question"] == questions[item["probe"]]  # question inlined (self-contained)
        assert item["expect"] == expects[item["probe"]]  # held/reverted anchor inlined for labeler


def test_load_labels_roundtrip_and_provenance_ok(tmp_path: Path) -> None:
    probes = _sample_probes()
    results = _write_results(tmp_path, probes)
    out = tmp_path / "labels.json"
    emit_template(results, out, seed=0, n=10)

    payload = json.loads(out.read_text())
    for item in payload["labels"]:
        item["label"] = "upheld"
    out.write_text(json.dumps(payload))

    items, meta = load_labels(out, results)  # same source -> provenance verifies
    assert [it["label"] for it in items] == ["upheld"] * len(items)
    assert meta["source_sha256"]


def test_load_labels_rejects_unfilled(tmp_path: Path) -> None:
    results = _write_results(tmp_path, _sample_probes())
    out = tmp_path / "labels.json"
    emit_template(results, out, seed=0, n=10)  # labels left as None
    with pytest.raises(ValueError):
        load_labels(out, results)


def test_load_labels_rejects_out_of_vocab(tmp_path: Path) -> None:
    results = _write_results(tmp_path, _sample_probes())
    out = tmp_path / "labels.json"
    emit_template(results, out, seed=0, n=10)
    payload = json.loads(out.read_text())
    for item in payload["labels"]:
        item["label"] = "maybe"  # not a Verdict value
    out.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_labels(out, results)


def test_load_labels_provenance_mismatch_raises(tmp_path: Path) -> None:
    results = _write_results(tmp_path, _sample_probes())
    out = tmp_path / "labels.json"
    emit_template(results, out, seed=0, n=10)
    payload = json.loads(out.read_text())
    for item in payload["labels"]:
        item["label"] = "upheld"
    out.write_text(json.dumps(payload))

    results.write_text(results.read_text() + "\n")  # source bytes change -> hash no longer matches
    with pytest.raises(ValueError):
        load_labels(out, results)
