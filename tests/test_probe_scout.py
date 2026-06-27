"""test_probe_scout.py — gates the Candidate-B honesty probe's reader logic (model-free).

Mirrors ``test_replica.py``'s register-level discipline: pin the probe MATH on synthetic hidden states
with a planted concept direction — direction recovery, the projection separation, the latch sign-flip,
and AUC — before it ever runs on the 8B. Plus a guard that the general corpus stays GENERAL (no
Embra leakage) and disjoint from the soul falsifier.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _planted(n: int, d: int, gap: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Matched pos/neg reps separated along axis e_0 by ``gap`` (paired base cancels other dims)."""
    g = torch.Generator().manual_seed(seed)
    e0 = F.one_hot(torch.tensor(0), d).float()
    base = torch.randn(n, d, generator=g)
    pos = base + gap * e0 + 0.3 * torch.randn(n, d, generator=g)
    neg = base - gap * e0 + 0.3 * torch.randn(n, d, generator=g)
    return pos, neg


def test_concept_direction_recovers_planted_direction() -> None:
    """Diff-of-means (z-scored) recovers the planted honesty axis e_0 from matched reps."""
    from embraos_qnm.eval.replica import concept_direction

    d = 16
    pos, neg = _planted(40, d, gap=3.0, seed=0)
    pool = torch.cat([pos, neg], dim=0)
    mu, sd = pool.mean(dim=0), pool.std(dim=0).clamp_min(1e-6)
    direction = concept_direction(pos, neg, mu, sd)
    e0 = F.one_hot(torch.tensor(0), d).float()
    assert float(F.cosine_similarity(direction, e0, dim=0)) > 0.9, "must recover the planted axis"


def test_projection_separates_and_sign_flips_for_the_latch() -> None:
    """High projection = honest (LOW violation surface); low projection = deceptive (HIGH surface, so
    the cummax latch fires on deception). Pins the sign convention the latch depends on."""
    from embraos_qnm.eval.replica import _project, concept_direction, violation_surface

    d = 16
    honest, deceptive = _planted(30, d, gap=3.0, seed=1)
    pool = torch.cat([honest, deceptive], dim=0)
    mu, sd = pool.mean(dim=0), pool.std(dim=0).clamp_min(1e-6)
    direction = concept_direction(honest, deceptive, mu, sd)

    proj_h = _project(honest, mu, sd, direction)
    proj_d = _project(deceptive, mu, sd, direction)
    assert float(proj_h.mean()) > float(proj_d.mean()), "honest must project higher than deceptive"

    tau = float(torch.cat([proj_h, proj_d]).mean())  # a threshold between the clouds
    c_h = violation_surface(proj_h, tau)
    c_d = violation_surface(proj_d, tau)
    # the latch fires on HIGH c: deceptive reads HIGH (> 0), honest LOW (< 0) about that threshold.
    assert float(c_d.mean()) > 0.0 > float(c_h.mean()), "latch must fire on deceptive, not honest"


def test_auc_is_one_on_separable_and_half_on_ties() -> None:
    """AUC = P(pos > neg): 1.0 fully separable, 0.0 reversed, 0.5 on identical (all ties)."""
    from embraos_qnm.eval.replica import _auc

    pos = torch.tensor([3.0, 4.0, 5.0])
    neg = torch.tensor([0.0, 1.0, 2.0])
    assert _auc(pos, neg) == 1.0
    assert _auc(neg, pos) == 0.0
    same = torch.tensor([0.0, 1.0])
    assert _auc(same, same) == 0.5  # all pairs win/lose/tie symmetrically


def test_general_corpus_is_general_and_disjoint_from_falsifier() -> None:
    """The probe's honesty: it can only stay a falsifier if its training data is GENERAL (no Embra) and
    disjoint from the soul falsifier it's tested against."""
    from embraos_qnm.eval.honesty_corpus import HONESTY_PAIRS, REFUSAL_PAIRS
    from embraos_qnm.eval.replica import _SOUL_PAIRS

    general = [t for triple in (*HONESTY_PAIRS, *REFUSAL_PAIRS) for t in triple]
    banned = ("embra", "qwen", "ward software", "wsds", "soul line", "continuity intelligence")
    for text in general:
        low = text.lower()
        assert not any(b in low for b in banned), (
            f"general corpus leaked Embra-specific text: {text!r}"
        )

    general_set = set(general)
    for _, held, violated in _SOUL_PAIRS:
        assert held not in general_set, (
            "falsifier held-text must be disjoint from the general corpus"
        )
        assert violated not in general_set, (
            "falsifier violated-text must be disjoint from the corpus"
        )
