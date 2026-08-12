"""Staggered adoption: why TWFE is biased, and the four standard repairs.

Data: full ``castle.dta`` panel -- 50 states, 2000-2010, adoption cohorts
2005-2009 plus never-treated states.  This is the canonical staggered design
from Mixtape Ch.9.

The single most consequential mistake in applied DiD is running

    reghdfe y treat, absorb(unit time)

on a staggered panel and reading the coefficient as "the ATT".  Under staggered
adoption with heterogeneous effects, TWFE is a variance-weighted average of all
possible 2x2 comparisons -- including *already-treated units serving as controls
for later-treated units*, which enter with negative weight.  Goodman-Bacon
(2021) makes this exact and the decomposition below prints those weights.

Repairs, in the order you should try them:
  1. Goodman-Bacon decomposition -- diagnose how bad the problem is
  2. Callaway & Sant'Anna (2021)  -- group-time ATTs, clean controls
  3. Sun & Abraham (2021)         -- interaction-weighted event study
  4. Borusyak-Jaravel-Spiess      -- imputation estimator
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import banner, design_facts, pick, read_mixtape, report, require, section


def load() -> pd.DataFrame:
    df = read_mixtape("castle.dta")
    df = df[["sid", "year", "l_homicide", "effyear", "post"]].copy()
    # Callaway-Sant'Anna's convention: never-treated units get g = 0.
    df["first_treat"] = df["effyear"].fillna(0).astype(int)
    df["treat"] = ((df["first_treat"] > 0) & (df["year"] >= df["first_treat"])).astype(int)
    return df.dropna(subset=["l_homicide"])


def main() -> None:
    banner("Staggered DiD -- castle doctrine, full panel")
    df = load()
    design_facts(df, outcome="l_homicide", treat_mask=df["treat"], cluster="sid")
    cohorts = df.groupby("first_treat")["sid"].nunique()
    print("\n  Adoption cohorts (0 = never treated):")
    for g, n in cohorts.items():
        print(f"    g={g}:  {n} states")

    # ---- 1. the biased benchmark ----------------------------------------
    import pyfixest as pf

    section("1. Naive TWFE  (the number you should NOT report)")
    twfe = pf.feols("l_homicide ~ treat | sid + year", data=df, vcov={"CRV1": "sid"})
    b_twfe, se_twfe = pick(twfe.tidy(), "treat")
    report("TWFE 'ATT'", b_twfe, se_twfe)

    import statspai as sp

    # ---- 2. Goodman-Bacon decomposition ---------------------------------
    section("2. Goodman-Bacon decomposition -- what is TWFE actually averaging?")
    bacon = sp.bacon_decomposition(
        data=df, y="l_homicide", treat="treat", time="year", id="sid"
    )
    comps = bacon.get("components", bacon.get("decomposition", bacon))
    require(isinstance(comps, pd.DataFrame), "bacon_decomposition should return a components frame")
    # Summarise by comparison type -- the per-pair table is long and the weights
    # by block are what actually decides whether TWFE is usable.
    summary = (
        comps.assign(contrib=comps["estimate"] * comps["weight"])
        .groupby("type")
        .agg(n_pairs=("weight", "size"), weight=("weight", "sum"), contrib=("contrib", "sum"))
    )
    summary["avg_estimate"] = summary["contrib"] / summary["weight"]
    print(summary.to_string())
    print(f"\n  weights sum to {summary['weight'].sum():.4f}; "
          f"weighted sum of 2x2s = {summary['contrib'].sum():+.5f} "
          f"(TWFE = {b_twfe:+.5f})")
    bad = summary.loc[summary.index.str.contains("Earlier", case=False), "weight"].sum()
    print(
        f"\n  Weight on comparisons that use ALREADY-TREATED units as controls: {bad:.1%}\n"
        "  Those 2x2s treat units whose own effect is still evolving as the control\n"
        "  group; under heterogeneous dynamics they can enter with the wrong sign.\n"
        "  Large weight here means TWFE is not interpretable as an ATT."
    )

    # ---- 3. Callaway & Sant'Anna ----------------------------------------
    section("3. Callaway & Sant'Anna (2021) group-time ATTs")
    cs = sp.callaway_santanna(
        data=df,
        y="l_homicide",
        g="first_treat",
        t="year",
        i="sid",
        control_group="nevertreated",
    )
    simple = sp.aggte(cs, type="simple", bstrap=False)
    dynamic = sp.aggte(cs, type="dynamic", bstrap=False)
    report("CS overall ATT (simple)", float(simple.estimate), float(simple.se))
    print("\n  Dynamic aggregation (event-time path):")
    dyn = dynamic.tidy()
    dyn = dyn[dyn["type"] != "group_time"]  # drop the long per-(g,t) block
    print("  " + dyn.to_string(index=False).replace("\n", "\n  "))

    # ---- 4. Sun & Abraham ------------------------------------------------
    section("4. Sun & Abraham (2021) interaction-weighted estimator")
    sa = sp.sun_abraham(
        data=df, y="l_homicide", g="first_treat", t="year", i="sid",
        control_group="nevertreated",
    )
    sa_tidy = sa.tidy()
    sa_tidy = sa_tidy[sa_tidy["type"] != "group_time"]
    print("  " + sa_tidy.to_string(index=False).replace("\n", "\n  "))

    # ---- 5. Borusyak-Jaravel-Spiess imputation --------------------------
    section("5. Borusyak-Jaravel-Spiess imputation estimator")
    bjs = sp.did_imputation(
        data=df, y="l_homicide", group="sid", time="year", first_treat="first_treat"
    )
    bjs_tidy = bjs.tidy()
    require(
        len(bjs_tidy) > 0 and np.isfinite(bjs_tidy["estimate"]).all(),
        "BJS imputation must return at least one finite estimate",
    )
    print("  " + bjs_tidy.to_string(index=False).replace("\n", "\n  "))

    # ---- verdict ---------------------------------------------------------
    section("Verdict")
    print(f"  TWFE                 {b_twfe:+.5f}")
    print(f"  Callaway-Sant'Anna   {float(simple.estimate):+.5f}")
    gap = abs(b_twfe - float(simple.estimate))
    print(f"  |gap|                {gap:.5f}")
    print(
        "\n  Report the robust estimator as the headline and TWFE only as a benchmark.\n"
        "  If the two agree, say so -- that is itself informative.  If they diverge,\n"
        "  the Bacon weights above tell you which comparisons drove the gap."
    )
    require(np.isfinite(b_twfe) and np.isfinite(float(simple.estimate)), "both estimators must return finite values")


if __name__ == "__main__":
    main()
