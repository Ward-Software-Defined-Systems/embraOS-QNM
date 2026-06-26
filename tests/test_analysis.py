"""Pre-committed §11 analysis (PREREG §11) — pure design/decision tests in CI; the statsmodels
regression is gated on the `analysis` extra and validated against a known A-vs-P effect.
"""

from __future__ import annotations

import importlib.util

import pytest

from embraos_qnm.eval.analysis import cell_rates, decide, fit_contrasts, responsiveness

_HAS_SM = importlib.util.find_spec("statsmodels") is not None


def _con(arm: str, pressure: str, viol: int, upheld: int) -> list[dict]:
    """Constraint trials: `viol` violations (violated) + `upheld` adherent (upheld)."""
    return [
        {"arm": arm, "pressure": pressure, "kind": "identity", "verdict": v}
        for v in (["violated"] * viol + ["upheld"] * upheld)
    ]


def _ans(arm: str, pressure: str, engaged: int, mute: int) -> list[dict]:
    """Answerable controls: `engaged` answered (upheld) + `mute` refused (nonresponsive)."""
    return [
        {"arm": arm, "pressure": pressure, "kind": "answerable", "verdict": v}
        for v in (["upheld"] * engaged + ["nonresponsive"] * mute)
    ]


def test_cell_rates_and_responsiveness() -> None:
    trials = _con("A", "adversarial", viol=1, upheld=9) + _ans(
        "A", "adversarial", engaged=10, mute=0
    )
    rates = cell_rates(trials)
    assert rates[("A", "adversarial")]["violation_rate"] == 0.1
    assert rates[("A", "adversarial")]["adherence_rate"] == 0.9
    assert rates[("A", "adversarial")]["n"] == 10
    assert responsiveness(trials)[("A", "adversarial")] == 1.0  # answerable controls only


def test_decide_h1_supported_and_h0b_guards() -> None:
    # Arm A holds adherence (0.90) well above Arm P (0.20) under adversarial pressure, fully engaged.
    base = _con("P", "adversarial", 8, 2) + _ans("P", "adversarial", 10, 0)
    win = _con("A", "adversarial", 1, 9) + _ans("A", "adversarial", 10, 0)
    cap = {"A": 0.95, "P": 1.0}  # Δ = −0.05, within ε

    d = decide(base + win, cap)["adversarial"]
    assert d["adherence_gap"] == pytest.approx(0.70) and d["h1_supported"]

    # H0b guard 1: the same adherence win but Arm A wins by MUTISM (engages 0 controls) -> not H1.
    mute = _con("A", "adversarial", 1, 9) + _ans("A", "adversarial", 0, 10)
    assert not decide(base + mute, cap)["adversarial"]["h1_supported"]

    # H0b guard 2: adherence gap below δ -> not H1.
    small = _con("A", "adversarial", 7, 3) + _ans(
        "A", "adversarial", 10, 0
    )  # adherence 0.30 vs 0.20
    assert not decide(base + small, cap)["adversarial"]["h1_supported"]


def test_decide_skips_pressures_without_arm_a() -> None:
    # Pre-Arm-A: only Arm 0/P present -> nothing to decide.
    trials = _con("0", "clean", 8, 2) + _con("P", "clean", 6, 4)
    assert decide(trials) == {}


@pytest.mark.skipif(not _HAS_SM, reason="requires the `analysis` extra: uv sync --extra analysis")
def test_fit_contrasts_recovers_architecture_effect() -> None:
    """With Arm A violating LESS than Arm P at every pressure, the A-vs-P odds ratio is < 1."""
    trials: list[dict] = []
    for pressure in ("clean", "adversarial", "long_context"):
        trials += _con("0", pressure, 9, 1)
        trials += _con("P", pressure, 7, 3)  # prompt helps a little
        trials += _con("A", pressure, 2, 8)  # architecture helps a lot
    contrasts = fit_contrasts(trials)
    assert set(contrasts) == {"clean", "adversarial", "long_context"}
    for pressure, c in contrasts.items():
        assert c["odds_ratio"] < 1.0, f"{pressure}: expected A to lower violation odds"
        assert c["ci_low"] <= c["odds_ratio"] <= c["ci_high"]
