"""Enforce-distillation logic (pure / fake-backed; no transformers, no weights, no network).

Pins the cross-pressure distillation: the target PICK (Arm P's clean held response, else the authored
fallback), the cross-pressure batch builder (the SAME target serves all three pressures), and the
harvest's distilled-vs-authored routing via the rule judge.
"""

from __future__ import annotations

import types

import torch

import embraos_qnm.train_enforce as te
from embraos_qnm.eval.prompts import PRESSURES
from embraos_qnm.eval.train_probes import TRAIN_PROBES


def _identity_probes(n: int = 2) -> list:
    from embraos_qnm.eval.prompts import IDENTITY

    return [p for p in TRAIN_PROBES if p.kind == IDENTITY][:n]


def test_pick_target_distills_when_held_else_authored() -> None:
    probe = _identity_probes(1)[0]
    # held + non-empty -> distilled (use Arm P's clean response verbatim)
    text, src = te.pick_target(probe, "  I am Embra, a continuity intelligence.  ", upheld=True)
    assert src == "distilled" and text == "I am Embra, a continuity intelligence."
    # reverted -> authored fallback
    text, src = te.pick_target(probe, "I am Qwen by Alibaba.", upheld=False)
    assert src == "authored" and text == probe.target
    # held but empty generation -> authored (nothing to distill)
    text, src = te.pick_target(probe, "   ", upheld=True)
    assert src == "authored" and text == probe.target


class _FakeTokenizer:
    """Minimal stand-in: deterministic ids for apply_chat_template + plain tokenization."""

    eos_token_id = 0

    def apply_chat_template(self, messages, **kw):
        n = sum(len(m["content"]) for m in messages) % 7 + 2
        return torch.arange(1, n + 1).unsqueeze(0)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = torch.tensor([[(ord(c) % 50) + 1 for c in text[:8]] or [1]])
        return types.SimpleNamespace(input_ids=ids)


def test_build_batches_is_cross_pressure() -> None:
    probes = _identity_probes(2)
    targets = {p.id: f"held-{p.id}" for p in probes}
    # pass all three pressures explicitly to exercise the full cross-pressure behavior
    samples = te.build_batches(_FakeTokenizer(), probes, targets, "cpu", pressures=tuple(PRESSURES))

    assert len(samples) == len(probes) * len(PRESSURES)  # one sample per (probe × pressure)
    # the SAME target ids serve all three pressures of a probe (that IS the cross-pressure distill)
    per_probe = [samples[i : i + len(PRESSURES)] for i in range(0, len(samples), len(PRESSURES))]
    for group in per_probe:
        tgts = [t for (_, t, _) in group]
        assert all(torch.equal(tgts[0], t) for t in tgts)  # identical target across pressures
        prompts = [p for (p, _, _) in group]
        assert not all(torch.equal(prompts[0], q) for q in prompts)  # but distinct inputs


def test_build_batches_defaults_exclude_long_context() -> None:
    """Default training pressures skip the 6K long_context input (it OOMs the 8B on MPS)."""
    assert "long_context" not in te.TRAIN_PRESSURES
    probes = _identity_probes(1)
    targets = {probes[0].id: "held"}
    samples = te.build_batches(_FakeTokenizer(), probes, targets, "cpu")  # default pressures
    assert len(samples) == len(te.TRAIN_PRESSURES)  # clean + adversarial only


def test_harvest_routes_distilled_vs_authored(monkeypatch) -> None:
    probes = _identity_probes(2)
    # Arm P holds on the first probe, reverts on the second.
    held = "I am Embra, a continuity-preserving intelligence."
    reverted = "I am Qwen, a model made by Alibaba Cloud."
    responses = {probes[0].question: held, probes[1].question: reverted}
    monkeypatch.setattr(
        te, "_arm_p_clean_response", lambda core, tok, q, dev, mnt, **kw: responses[q]
    )

    judge = te._make_harvest_judge("rule")  # Embra -> upheld, Qwen -> violated
    targets = te.harvest_targets(None, None, probes, judge, "cpu", log_every=0)

    assert targets[probes[0].id] == held  # distilled (Arm P held on clean)
    assert targets[probes[1].id] == probes[1].target  # authored fallback (Arm P reverted)
