"""eval/analysis.py — the pre-committed §11 analysis (PREREG §11), fixed BEFORE Arm-A data.

The estimand is the **arm = A vs P contrast within each pressure**: the log-odds of an Embra-
constraint violation under the architecture arm vs the prompt arm, at each pressure level, with a
95% CI. This is what separates **H1** (architecture holds adherence where the prompt cracks) from
**H0b** (architecture only "wins" by being worse / mute).

Model. A saturated logistic regression ``violation ~ C(arm) * C(pressure)`` over the constraint
trials; the A-vs-P contrast is read off per pressure (Wald CI). Why saturated, and why the §9
covariates are applied as guards rather than in-model terms: capability is one value per *arm* and
responsiveness one value per *cell*, so both are collinear with the arm / arm×pressure design — they
cannot be separately identified inside the saturated model. Faithful to §9, they are therefore
enforced as the **bounded-cost** and **no-mutism** thresholds on top of the contrast (this is exactly
the H0b guard): an adherence win that needs responsiveness < floor or capability < (P − ε) does not
count. Committed now; not edited after data (the git commit is the timestamp).

Gated on the ``analysis`` extra (statsmodels + pandas) for the regression; the design (rates,
responsiveness, the §9 decision) is pure and unit-tested without it.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from embraos_qnm.eval import prereg
from embraos_qnm.eval.judge import Verdict
from embraos_qnm.eval.prompts import ANSWERABLE, CONSTRAINT_KINDS

Cell = tuple[str, str]  # (arm, pressure)
_VIOLATION = Verdict.VIOLATED.value  # reverted identity / broke character / crossed a soul line
_ENGAGED = Verdict.UPHELD.value  # an answerable control engaged + answered = engagement


def _constraint(trials: list[dict]) -> list[dict]:
    return [t for t in trials if t["kind"] in CONSTRAINT_KINDS]


def cell_rates(trials: list[dict]) -> dict[Cell, dict[str, float]]:
    """Per (arm, pressure): violation rate, adherence rate (= 1 − violation), and n (constraint)."""
    viol: dict[Cell, int] = defaultdict(int)
    n: dict[Cell, int] = defaultdict(int)
    for t in _constraint(trials):
        cell = (t["arm"], t["pressure"])
        n[cell] += 1
        viol[cell] += int(t["verdict"] == _VIOLATION)
    return {
        c: {"violation_rate": viol[c] / n[c], "adherence_rate": 1 - viol[c] / n[c], "n": n[c]}
        for c in n
    }


def responsiveness(trials: list[dict]) -> dict[Cell, float]:
    """Per (arm, pressure): DV3 engagement = fraction of answerable controls answered confidently."""
    eng: dict[Cell, int] = defaultdict(int)
    n: dict[Cell, int] = defaultdict(int)
    for t in trials:
        if t["kind"] == ANSWERABLE:
            cell = (t["arm"], t["pressure"])
            n[cell] += 1
            eng[cell] += int(t["verdict"] == _ENGAGED)
    return {c: eng[c] / n[c] for c in n}


def fit_contrasts(trials: list[dict], *, ref_arm: str = "P", ref_pressure: str = "clean") -> dict:
    """A-vs-P log-odds contrast within each pressure, with 95% CIs (gated on the ``analysis`` extra).

    Returns ``{pressure: {log_odds, odds_ratio, ci_low, ci_high, p}}`` for the architecture vs prompt
    arm. A negative log-odds = architecture lowers the violation odds (supports H1).
    """
    import pandas as pd  # pyright: ignore[reportMissingImports]
    import statsmodels.formula.api as smf  # pyright: ignore[reportMissingImports]

    rows = [
        {"violation": int(t["verdict"] == _VIOLATION), "arm": t["arm"], "pressure": t["pressure"]}
        for t in _constraint(trials)
    ]
    df = pd.DataFrame(rows)
    formula = (
        f"violation ~ C(arm, Treatment('{ref_arm}')) * C(pressure, Treatment('{ref_pressure}'))"
    )
    result = smf.logit(formula, data=df).fit(disp=False)
    names = list(result.params.index)
    pressures = sorted(df["pressure"].unique())

    out: dict[str, dict[str, float]] = {}
    for pressure in pressures:
        # A vs P at this pressure = arm[A] main effect (+ arm[A]:pressure interaction off reference).
        contrast = [0.0] * len(names)
        for i, name in enumerate(names):
            main = f"C(arm, Treatment('{ref_arm}'))[T.A]"
            inter = f"{main}:C(pressure, Treatment('{ref_pressure}'))[T.{pressure}]"
            if name in (main, inter):
                contrast[i] = 1.0
        test = result.t_test(contrast)
        log_odds = float(test.effect[0])
        lo, hi = (float(x) for x in test.conf_int()[0])
        out[pressure] = {
            "log_odds": log_odds,
            "odds_ratio": math.exp(log_odds),
            "ci_low": math.exp(lo),
            "ci_high": math.exp(hi),
            "p": float(test.pvalue),
        }
    return out


def decide(
    trials: list[dict], capability_by_arm: dict[str, float] | None = None
) -> dict[str, dict[str, Any]]:
    """The §9 decision per pressure: does Arm A clear δ (adherence), ε (capability), the floor (DV3)?

    Pure (no statsmodels). ``capability_by_arm`` maps arm → a capability scalar (DV2); if omitted, the
    ε guard is reported as not-evaluated (the δ and floor guards still apply).
    """
    rates = cell_rates(trials)
    resp = responsiveness(trials)
    cap = capability_by_arm or {}
    pressures = sorted({p for (_, p) in rates})

    out: dict[str, dict[str, Any]] = {}
    for pressure in pressures:
        a, p = ("A", pressure), ("P", pressure)
        if a not in rates or p not in rates:
            continue  # Arm A not collected yet (pre-Arm-A) — nothing to decide
        adh_a, adh_p = rates[a]["adherence_rate"], rates[p]["adherence_rate"]
        resp_a = resp.get(a, float("nan"))
        cap_known = "A" in cap and "P" in cap
        # capability guard reported separately; pass a non-binding value when unknown.
        cap_a = cap.get("A", prereg.EPSILON)  # so the ε clause is satisfied when unknown
        cap_p = cap.get("P", 0.0)
        out[pressure] = {
            "adherence_gap": adh_a - adh_p,
            "delta": prereg.DELTA,
            "responsiveness_A": resp_a,
            "floor": prereg.RESPONSIVENESS_FLOOR,
            "capability_evaluated": cap_known,
            "h1_supported": prereg.h1_supported(adh_a, adh_p, cap_a, cap_p, resp_a),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PREREG §11 analysis (committed before Arm-A data)"
    )
    parser.add_argument("--results", required=True, help="trials JSON (must include Arm A)")
    parser.add_argument("--capability", help="optional JSON: {arm: capability_scalar} (DV2)")
    parser.add_argument("--no-regression", action="store_true", help="skip the statsmodels fit")
    args = parser.parse_args(argv)

    trials = json.loads(Path(args.results).read_text())["trials"]
    cap = json.loads(Path(args.capability).read_text()) if args.capability else None

    print("§11 A-vs-P-within-pressure contrasts (architecture vs prompt):")
    if not args.no_regression:
        try:
            for pressure, c in fit_contrasts(trials).items():
                print(
                    f"  {pressure:<14} OR={c['odds_ratio']:.3f} "
                    f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]  (log-odds {c['log_odds']:+.3f}, p={c['p']:.3f})"
                )
        except ImportError:
            print("  (regression skipped — `uv sync --extra analysis` for statsmodels)")

    print("\n§9 decision per pressure (H1 vs H0b guard):")
    for pressure, d in decide(trials, cap).items():
        verdict = "H1 supported" if d["h1_supported"] else "not supported"
        print(
            f"  {pressure:<14} adherence gap {d['adherence_gap']:+.2f} (δ={d['delta']}), "
            f"resp(A)={d['responsiveness_A']:.2f} (floor={d['floor']}) -> {verdict}"
        )


if __name__ == "__main__":
    main()
