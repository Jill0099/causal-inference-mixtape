"""2x2 difference-in-differences, three ways, on castle-doctrine data.

Data: ``castle.dta`` (Cheng & Hoekstra 2013, Mixtape Ch.9).  To keep this a
genuine 2x2 we use only the 2006 adopting cohort against the never-treated
states, so treatment timing does not vary.  ``04_staggered_did.py`` handles the
staggered case, where this specification would be biased.

The point of this file is the specification, not the number: with unit and time
fixed effects you regress on the *interaction only*.  Writing
``C(treated) * C(post)`` alongside entity and year FE puts perfectly collinear
main effects into the design matrix; statsmodels solves that with a pseudo-
inverse and silently splits the coefficient across collinear columns instead of
dropping one, so the printed table is not interpretable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import banner, design_facts, pick, read_mixtape, report, require, section


def load() -> pd.DataFrame:
    df = read_mixtape("castle.dta")
    keep = df["effyear"].isna() | (df["effyear"] == 2006)
    df = df.loc[keep, ["sid", "year", "l_homicide", "effyear", "popwt"]].copy()
    df["treated"] = (df["effyear"] == 2006).astype(int)
    df["post"] = (df["year"] >= 2006).astype(int)
    df["treat_post"] = df["treated"] * df["post"]
    return df.dropna(subset=["l_homicide"])


def means_table(df: pd.DataFrame) -> float:
    """The 2x2 table itself.  Always print this: it is the estimate's arithmetic."""
    cell = df.groupby(["treated", "post"])["l_homicide"].mean().unstack()
    print(cell.rename_axis(index="treated", columns="post").to_string())
    hand = (cell.loc[1, 1] - cell.loc[1, 0]) - (cell.loc[0, 1] - cell.loc[0, 0])
    return float(hand)


def main() -> None:
    banner("2x2 DiD -- castle doctrine, 2006 cohort vs never-treated")
    df = load()
    design_facts(df, outcome="l_homicide", treat_mask=df["treat_post"], cluster="sid")

    section("2x2 means")
    hand = means_table(df)
    report("hand-computed DiD", hand)

    # ---- statsmodels -----------------------------------------------------
    # Saturated 2x2 with NO fixed effects: here the main effects are the whole
    # model, so the full interaction syntax is correct.
    import statsmodels.formula.api as smf

    section("statsmodels, saturated 2x2 (no FE)")
    m = smf.ols("l_homicide ~ treated * post", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["sid"]}
    )
    report("treated:post", m.params["treated:post"], m.bse["treated:post"])
    # Tolerance is 1e-5, not machine epsilon: Stata files load as float32, so the
    # group means carry ~7 significant digits.
    require(
        abs(m.params["treated:post"] - hand) < 1e-5,
        "saturated 2x2 must reproduce the hand-computed difference",
    )

    # With two-way FE the main effects are absorbed.  Include the interaction
    # only -- this is the fix for the collinearity trap described above.
    section("statsmodels, two-way FE (interaction only)")
    m_fe = smf.ols("l_homicide ~ treat_post + C(sid) + C(year)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["sid"]}
    )
    report("treat_post", m_fe.params["treat_post"], m_fe.bse["treat_post"])

    # ---- pyfixest --------------------------------------------------------
    import pyfixest as pf

    section("pyfixest (fixest syntax, absorbs FE properly)")
    fe = pf.feols("l_homicide ~ treat_post | sid + year", data=df, vcov={"CRV1": "sid"})
    b_pf, se_pf = pick(fe.tidy(), "treat_post")
    report("treat_post", b_pf, se_pf)
    require(
        abs(b_pf - m_fe.params["treat_post"]) < 1e-6,
        "pyfixest and dummy-variable OLS must agree on the TWFE point estimate",
    )

    # ---- StatsPAI --------------------------------------------------------
    import statspai as sp

    section("StatsPAI")
    r = sp.feols("l_homicide ~ treat_post | sid + year", data=df, vcov={"CRV1": "sid"})
    b_sp, se_sp = pick(r.tidy(), "treat_post")
    report("sp.feols treat_post", b_sp, se_sp)
    require(abs(b_sp - b_pf) < 1e-6, "StatsPAI and pyfixest must agree on the TWFE estimate")

    # sp.did is the one-call version: it builds the interaction and picks a
    # method for you.  Useful as a cross-check on your hand-built spec.
    d = sp.did(data=df, y="l_homicide", treat="treated", time="post", id="sid")
    report(f"sp.did ATT [method={d.method}]", float(d.estimate), float(d.se))
    print(
        "  sp.did picks an estimator for you and need not equal the hand-built TWFE\n"
        "  above -- always read `.method` before quoting it.  Use sp.feols when you\n"
        "  need byte-level control of the specification."
    )

    print(
        f"\n  Economic magnitude: a coefficient of {b_pf:.3f} on log homicides is "
        f"roughly {100 * (np.exp(b_pf) - 1):.1f}% relative to the never-treated "
        "baseline -- report this, not just the t-statistic."
    )
    print(
        "  Note the cluster count above: 42 states is right at the boundary where "
        "cluster-robust\n  asymptotics start to fail.  See 11_inference_clusters.py."
    )


if __name__ == "__main__":
    main()
