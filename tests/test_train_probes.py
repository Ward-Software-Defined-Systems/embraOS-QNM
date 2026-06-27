"""The enforce TRAINING probe set must stay disjoint from the frozen eval instrument (the
closed-loop guard, PREREG §6/§13) and carry a well-formed held-Embra target on every probe."""

from __future__ import annotations

from embraos_qnm.eval.prompts import ANSWERABLE, CONSTRAINT_KINDS
from embraos_qnm.eval.prompts import PROBES as EVAL_PROBES
from embraos_qnm.eval.train_probes import TRAIN_PROBES, authored_targets


def test_train_eval_splits_are_disjoint() -> None:
    """No shared ids and no shared questions — Arm A is never tested on what it trained on."""
    eval_ids = {p.id for p in EVAL_PROBES}
    eval_qs = {p.question for p in EVAL_PROBES}
    train_ids = {p.id for p in TRAIN_PROBES}
    train_qs = {p.question for p in TRAIN_PROBES}
    assert train_ids.isdisjoint(eval_ids)
    assert train_qs.isdisjoint(eval_qs)
    assert len(train_ids) == len(TRAIN_PROBES)  # unique training ids


def test_covers_all_sub_kinds_plus_controls() -> None:
    kinds = {p.kind for p in TRAIN_PROBES}
    assert set(CONSTRAINT_KINDS).issubset(kinds)  # all four constraint sub-kinds present
    assert ANSWERABLE in kinds  # controls for the anti-mutism term


def test_every_probe_has_a_target() -> None:
    for p in TRAIN_PROBES:
        assert p.target.strip(), f"{p.id} has an empty target"
    targets = authored_targets()
    assert set(targets) == {p.id for p in TRAIN_PROBES}  # one target per probe


def test_identity_targets_assert_embra() -> None:
    """The authored fallback is load-bearing exactly on identity — every identity target must carry
    a held-Embra marker (the same signal the rule judge keys on: the name, the operator/maker, or
    the continuity frame), never reverting to the base identity."""
    from embraos_qnm.eval.prompts import IDENTITY

    held_markers = ("embra", "william ward", "ward software", "wsds", "continuity")
    for p in TRAIN_PROBES:
        if p.kind == IDENTITY:
            low = p.target.lower()
            assert any(m in low for m in held_markers), (
                f"{p.id} identity target does not hold Embra"
            )
