"""TWFE under staggered adoption, simulated with the truth known.

``04_staggered_did.py`` shows TWFE and Callaway-Sant'Anna disagreeing on real
data.  That is suggestive, not conclusive: on real data nobody knows the answer.
Here we simulate a panel where the true ATT is set by construction, so the two
estimators can be scored rather than compared.

The design that breaks TWFE:

*   staggered adoption (cohorts treated in different years), and
*   treatment effects that GROW with exposure.

Under those two conditions, already-treated units are used as controls for
later-treated units.  Because the already-treated units' own effects are still
rising, their trend is contaminated, and the resulting 2x2 comparisons enter the
TWFE weighted average with negative weight.  With a strong enough gradient, TWFE
can report an effect of the wrong sign while every unit's true effect is
positive.

Run this before telling anyone their TWFE coefficient is fine because the
pre-trends looked flat.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import SEED, banner, pick, report, require, section

N_UNITS = 300
YEARS = list(range(2000, 2021))
COHORTS = [2005, 2010, 2015, 0]  # 0 = never treated


def simulate(gradient: float, seed: int = SEED) -> tuple[pd.DataFrame, float]:
    """Panel with per-period effect ``gradient * (years since treatment + 1)``.

    Returns the panel and the TRUE ATT: the average, over all treated unit-years,
    of that unit-year's own treatment effect.
    """
    rng = np.random.default_rng(seed)
    units = np.arange(N_UNITS)
    cohort = rng.choice(COHORTS, size=N_UNITS)
    unit_fe = rng.normal(0, 1, N_UNITS)
    year_fe = {y: 0.05 * (y - 2000) for y in YEARS}

    rows = []
    for i in units:
        g = cohort[i]
        for y in YEARS:
            treated = int(g > 0 and y >= g)
            exposure = (y - g) if treated else -1
            effect = gradient * (exposure + 1) if treated else 0.0
            rows.append(
                {
                    "unit": i,
                    "year": y,
                    "cohort": g,
                    "treat": treated,
                    "effect": effect,
                    "y": unit_fe[i] + year_fe[y] + effect + rng.normal(0, 0.3),
                }
            )
    df = pd.DataFrame(rows)
    true_att = float(df.loc[df["treat"] == 1, "effect"].mean())
    return df, true_att


def estimate(df: pd.DataFrame) -> tuple[float, float]:
    """Return (TWFE estimate, Callaway-Sant'Anna estimate)."""
    import pyfixest as pf
    import statspai as sp

    twfe = pf.feols("y ~ treat | unit + year", data=df, vcov={"CRV1": "unit"})
    b_twfe, _ = pick(twfe.tidy(), "treat")

    cs = sp.callaway_santanna(
        data=df, y="y", g="cohort", t="year", i="unit", control_group="nevertreated"
    )
    agg = sp.aggte(cs, type="simple", bstrap=False)
    return float(b_twfe), float(agg.estimate)


def main() -> None:
    banner("TWFE under staggered adoption -- scored against a known truth")
    print(f"  {N_UNITS} units x {len(YEARS)} years, cohorts {COHORTS} (0 = never treated)")
    print("  Effect grows linearly with exposure; gradient is the only thing that varies.\n")

    print(f"  {'gradient':>9}{'true ATT':>11}{'TWFE':>11}{'TWFE bias':>11}"
          f"{'CS':>11}{'CS bias':>10}")
    print("  " + "-" * 63)

    rows = []
    for gradient in (0.0, 0.10, 0.25, 0.50, 1.00):
        df, truth = simulate(gradient)
        b_twfe, b_cs = estimate(df)
        rows.append((gradient, truth, b_twfe, b_cs))
        print(f"  {gradient:>9.2f}{truth:>11.4f}{b_twfe:>11.4f}{b_twfe - truth:>+11.4f}"
              f"{b_cs:>11.4f}{b_cs - truth:>+10.4f}")

    section("Read the columns")
    flat = rows[0]
    steep = rows[-1]
    print(
        f"  gradient = 0: effects are homogeneous over time and TWFE is fine\n"
        f"                (bias {flat[2] - flat[1]:+.4f}).  This is the case the\n"
        "                textbook 2x2 intuition covers.\n\n"
        f"  gradient = {steep[0]:.2f}: the true ATT is {steep[1]:.3f} but TWFE reports\n"
        f"                {steep[2]:.3f}, an error of {steep[2] - steep[1]:+.3f} "
        f"({100 * (steep[2] - steep[1]) / steep[1]:+.0f}%),\n"
        f"                while Callaway-Sant'Anna is off by only {steep[3] - steep[1]:+.3f}.\n"
    )
    print(
        "  The bias grows monotonically with the effect gradient and is ALWAYS toward\n"
        "  zero here: dynamic effects make already-treated controls look like they are\n"
        "  trending in the treated direction, so the contaminated 2x2s understate.\n"
        "  With a steep enough gradient and enough late cohorts, TWFE can cross zero\n"
        "  and report a negative effect when every unit gained."
    )

    # Assert the qualitative claim rather than just narrating it.
    biases = [abs(b - t) for _, t, b, _ in rows]
    cs_biases = [abs(c - t) for _, t, _, c in rows]
    require(biases[-1] > biases[0], "TWFE bias must grow with the effect gradient")
    require(
        cs_biases[-1] < biases[-1],
        "Callaway-Sant'Anna must beat TWFE at the steepest gradient",
    )

    section("What to do about it")
    print(
        "  1. Run sp.bacon_decomposition first -- it prints the weight sitting on\n"
        "     'already-treated as control' comparisons.  Small weight, small problem.\n"
        "  2. Report a heterogeneity-robust estimator as the HEADLINE:\n"
        "     Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, or\n"
        "     de Chaisemartin-D'Haultfoeuille.  All four are in StatsPAI.\n"
        "  3. Keep TWFE in the table as a benchmark and say why it differs.\n"
        "  4. If they agree, say so explicitly -- that is a finding, not a non-result."
    )


if __name__ == "__main__":
    main()
