"""Honest DiD: what to do instead of eyeballing a flat pre-trend.

Data: ``castle.dta``, full staggered panel.

The standard parallel-trends defence is a plot of pre-treatment coefficients
plus a joint F-test.  Roth (2022) shows why that is weaker than it looks:

*   The pre-test has **low power** against exactly the violations that matter.
    A linear differential trend large enough to overturn your result is often
    one you would fail to detect most of the time.
*   Conditioning on having passed the pre-test **distorts** the sampling
    distribution of the estimate you then report -- a pre-test bias.
*   "We cannot reject parallel trends" is not "parallel trends holds".  Absence
    of evidence, at low power, is very weak evidence of absence.

Rambachan & Roth (2023) replace the binary test with a sensitivity analysis:
assume the post-treatment violation of parallel trends is no larger than M
times the largest violation observed pre-treatment, then report the breakdown
value of M -- the point at which your conclusion flips.  That number is
reportable, comparable across papers, and honest about what the data can bear.

This script reports:
  1. the conventional pre-trend test (and its power)
  2. the Honest DiD robust confidence sets across a grid of M
  3. the breakdown M
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import banner, read_mixtape, report, require, section


def load() -> pd.DataFrame:
    df = read_mixtape("castle.dta")[["sid", "year", "l_homicide", "effyear"]].copy()
    df["first_treat"] = df["effyear"].fillna(0).astype(int)
    return df.dropna(subset=["l_homicide"])


def main() -> None:
    banner("Honest DiD -- sensitivity instead of a pre-trend eyeball test")
    df = load()
    import statspai as sp

    # ---- 1. the event study ----------------------------------------------
    section("1. Callaway-Sant'Anna event study")
    cs = sp.callaway_santanna(
        data=df, y="l_homicide", g="first_treat", t="year", i="sid",
        control_group="nevertreated",
    )
    dyn = sp.aggte(cs, type="dynamic", bstrap=False)
    tidy = dyn.tidy()
    ev = tidy[tidy["type"] == "event_study"] if "type" in tidy.columns else tidy
    if len(ev):
        print("  " + ev.to_string(index=False).replace("\n", "\n  "))
    simple = sp.aggte(cs, type="simple", bstrap=False)
    report("overall ATT", float(simple.estimate), float(simple.se))

    # ---- 2. the conventional pre-test, and its power ---------------------
    section("2. The conventional pre-trend test -- and what it can actually detect")
    pre = getattr(cs, "pretrend_test", None)
    require(callable(pre), "Callaway-Sant'Anna result must expose a pre-trend test")
    pre_result = pre()
    require(isinstance(pre_result, dict), "pre-trend test must return a result mapping")
    pre_stat = float(pre_result.get("statistic", np.nan))
    pre_pvalue = float(pre_result.get("pvalue", np.nan))
    require(
        np.isfinite(pre_stat) and np.isfinite(pre_pvalue) and 0 <= pre_pvalue <= 1,
        "pre-trend test must return a finite statistic and p-value in [0, 1]",
    )
    print(f"  pre-trend test: {pre_result}")
    power = sp.pretrends_power(dyn, alpha=0.05)
    print("\n  Power analysis (Roth 2022):")
    for k in ("power", "power_joint", "alpha", "df", "noncentrality", "warning"):
        if k in power:
            print(f"    {k:<18} {power[k]}")
    p_ind, p_joint = power.get("power"), power.get("power_joint")
    require(
        p_ind is not None and p_joint is not None,
        "pre-trend power must return both power measures",
    )
    require(
        np.isfinite(float(p_ind)) and np.isfinite(float(p_joint)),
        "pre-trend power measures must be finite",
    )
    print(
        f"\n  Read these numbers literally.  Against the trend this design is powered\n"
        f"  to detect, an individual pre-period test catches a violation only\n"
        f"  {100 * float(p_ind):.0f}% of the time, and the JOINT test only "
        f"{100 * float(p_joint):.0f}%.\n"
        "  Conventional practice wants 80%.  A pre-trend plot that 'looks flat'\n"
        "  under this much noise is close to uninformative -- which is precisely\n"
        "  why the sensitivity analysis below replaces it rather than supplements it."
    )

    # ---- 3. Honest DiD ----------------------------------------------------
    section("3. Rambachan-Roth robust confidence sets over M")
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    method = "relative_magnitude"
    hd = sp.honest_did(dyn, e=0, m_grid=grid, method=method)
    print(f"  method = {method}")
    print("  " + str(hd).replace("\n", "\n  "))

    # ---- 4. the breakdown value -----------------------------------------
    section("4. The number to put in the paper: the breakdown M")
    table = hd if isinstance(hd, pd.DataFrame) else getattr(hd, "table", None)
    require(
        isinstance(table, pd.DataFrame)
        and {"M", "ci_lower", "ci_upper", "rejects_zero"}.issubset(table.columns),
        "Honest DiD must return M, confidence bounds, and rejection decisions",
    )
    require(table["M"].is_unique, "Honest DiD must return exactly one row per M value")
    require(
        len(table) == len(grid)
        and np.allclose(np.sort(table["M"].to_numpy(dtype=float)), np.asarray(grid)),
        "Honest DiD must return the complete requested M grid",
    )
    require(
        np.isfinite(table[["M", "ci_lower", "ci_upper"]].to_numpy(dtype=float)).all(),
        "Honest DiD sensitivity bounds must be finite",
    )
    if isinstance(table, pd.DataFrame) and "rejects_zero" in table.columns:
        rejecting = table.loc[table["rejects_zero"], "M"]
        failing = table.loc[~table["rejects_zero"], "M"]
        if len(rejecting) and len(failing):
            lo, hi = float(rejecting.max()), float(failing.min())
            print(f"  Result survives up to M = {lo:.2f}; the CI first covers zero at M = {hi:.2f}.")
            print(f"  BREAKDOWN M is between {lo:.2f} and {hi:.2f}.\n")
            if hi <= 1.0:
                print(
                    "  This is a FRAGILE result.  It cannot withstand a post-treatment\n"
                    "  violation of parallel trends as large as the worst one already\n"
                    "  visible in the pre-period.  Say that in the paper.\n"
                )
        elif len(rejecting) == len(table):
            print(f"  Significant across the whole grid up to M = {grid[-1]}: robust.\n")
        else:
            print("  The CI covers zero even at M = 0: the baseline result is not significant.\n")
    print(
        "  M = 0     assumes exactly parallel trends (the conventional DiD assumption)\n"
        "  M = 1     allows post-treatment violation as large as the WORST pre-period one\n"
        "  M = 2     allows twice that\n\n"
        "  The breakdown M is where the robust confidence set first contains zero.\n"
        "  Report it as a sentence: 'the estimate remains significant for violations of\n"
        "  parallel trends up to M times the largest pre-treatment violation.'\n"
        "  A breakdown below 1 means your result cannot survive a violation no bigger\n"
        "  than what you already observe before treatment -- that is a fragile result,\n"
        "  no matter how flat the pre-trend plot looked."
    )

    section("What replaces 'the pre-trends are flat'")
    print(
        "  1. Event-study plot (still show it -- it is a description, not a test)\n"
        "  2. Joint pre-trend test WITH its power against a meaningful trend\n"
        "  3. Honest DiD robust CI across a grid of M\n"
        "  4. The breakdown M, stated in words\n"
        "  Items 2-4 are what a 2025 referee expects; item 1 alone is not a defence."
    )
    require(np.isfinite(float(simple.estimate)), "the underlying ATT must be finite")


if __name__ == "__main__":
    main()
