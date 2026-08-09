"""Sharp RDD on Lee-Moretti-Butler (2004) US House data.

Data: ``lmb-data.dta``.  Running variable is the Democratic vote share in the
previous election, cutoff 0.5, treatment is "a Democrat holds the seat", outcome
is the representative's ADA score.

Two things this file is meant to settle:

*   **A hand-rolled ``y ~ D * x`` OLS inside a chosen bandwidth is not an RDD
    estimate.**  It uses a uniform kernel, no bias correction, and conventional
    standard errors.  Calonico-Cattaneo-Titiunik showed the resulting confidence
    interval undercovers, because the optimal bandwidth is chosen to trade off
    bias against variance and therefore leaves first-order bias in the estimate.
    Report the robust bias-corrected interval.
*   **Manipulation testing is not optional and no longer needs R.**  The old
    advice to call ``rdd::DCdensity`` is obsolete: ``rddensity`` (Cattaneo-
    Jansson-Ma) has a native Python implementation, and StatsPAI exposes both it
    and the original McCrary (2008) test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import banner, design_facts, read_mixtape, report, require, section

CUTOFF = 0.5


#: Predetermined district characteristics, used for the balance table in step 7.
COVARIATES = ["totpop", "medianincome", "votingpop"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (estimation sample, sample retaining predetermined covariates)."""
    raw = read_mixtape("lmb-data.dta")
    cols = ["score", "lagdemvoteshare", "democrat", "lagdemocrat", "state", "year"]
    df = raw[cols].dropna(subset=["score", "lagdemvoteshare"]).copy()
    df["x_centered"] = df["lagdemvoteshare"] - CUTOFF
    df["above"] = (df["lagdemvoteshare"] >= CUTOFF).astype(int)
    cov = raw[["lagdemvoteshare", *COVARIATES]].dropna(subset=["lagdemvoteshare"]).copy()
    return df, cov


def main() -> None:
    banner("Sharp RDD -- Lee-Moretti-Butler, ADA score at the 50% vote-share cutoff")
    df, df_cov = load()
    design_facts(df, outcome="score", treat_mask=df["above"], cluster="state")

    # Sanity: is the design actually sharp?  Check that crossing the cutoff maps
    # (nearly) one-for-one into holding the seat, before assuming sharp RDD.
    compliance = df.groupby("above")["lagdemocrat"].mean()
    print(f"\n  P(Democrat holds seat | below cutoff) = {compliance.loc[0]:.4f}")
    print(f"  P(Democrat holds seat | above cutoff) = {compliance.loc[1]:.4f}")
    require(
        compliance.loc[1] - compliance.loc[0] > 0.9,
        "sharp RDD requires near-perfect compliance at the cutoff",
    )

    import statspai as sp

    # ---- 1. manipulation test -------------------------------------------
    section("1. Manipulation / density test at the cutoff (run this FIRST)")
    dens = sp.rddensity(data=df, x="lagdemvoteshare", c=CUTOFF)
    print(dens.summary())
    print(
        "  A rejection here means units sort across the cutoff and the design is\n"
        "  dead -- no bandwidth choice repairs it.  Run this before any estimate."
    )

    # ---- 2. the naive estimate people actually write ---------------------
    section("2. Naive local-linear OLS (a benchmark, NOT the headline)")
    import statsmodels.formula.api as smf

    h0 = 0.05
    for bw in (h0 / 2, h0, 2 * h0):
        sub = df[df["x_centered"].abs() <= bw]
        m = smf.ols("score ~ above * x_centered", data=sub).fit(cov_type="HC1")
        report(f"uniform kernel, h={bw:.3f}, n={len(sub)}", m.params["above"], m.bse["above"])
    print(
        "  These move around with the bandwidth and their intervals are too narrow:\n"
        "  they ignore the bias that the bandwidth choice deliberately leaves in."
    )

    # ---- 3. the estimate to report --------------------------------------
    section("3. rdrobust -- MSE-optimal bandwidth, triangular kernel, bias-corrected CI")
    rd = sp.rdrobust(data=df, y="score", x="lagdemvoteshare", c=CUTOFF)
    print(rd.summary())

    # ---- 4. robustness sweeps -------------------------------------------
    section("4. Bandwidth robustness")
    rows = []
    for mult, label in [(0.5, "h/2"), (1.0, "h (MSE-optimal)"), (2.0, "2h")]:
        h = float(rd.diagnostics.get("bandwidth_h", rd.diagnostics.get("h", np.nan))) * mult
        if not np.isfinite(h):
            continue
        r = sp.rdrobust(data=df, y="score", x="lagdemvoteshare", c=CUTOFF, h=h)
        rows.append((label, h, float(r.estimate), float(r.se)))
    for label, h, b, s in rows:
        report(f"{label} (h={h:.4f})", b, s)
    print(
        "  Passing h explicitly also changes how the bias-correction bandwidth b is\n"
        "  chosen, so the middle row need not reproduce step 3 exactly.  Report the\n"
        "  MSE-optimal result from step 3 as the headline and this sweep as evidence\n"
        "  that the sign and rough magnitude do not hinge on the bandwidth."
    )

    section("5. Polynomial-order robustness")
    for p in (1, 2):
        r = sp.rdrobust(data=df, y="score", x="lagdemvoteshare", c=CUTOFF, p=p)
        report(f"local polynomial order p={p}", float(r.estimate), float(r.se))
    print(
        "  Stop at p=2.  Gelman & Imbens (2019): high-order global polynomials give\n"
        "  noisy weights and poor coverage.  Never report a quartic as the headline."
    )

    # ---- 6. placebo cutoffs ---------------------------------------------
    section("6. Placebo cutoffs (there should be no jump away from 0.5)")
    for c in (0.35, 0.40, 0.60, 0.65):
        try:
            r = sp.rdrobust(data=df, y="score", x="lagdemvoteshare", c=c)
            flag = "  <-- SUSPICIOUS" if abs(float(r.estimate) / float(r.se)) > 1.96 else ""
            report(f"placebo cutoff c={c}", float(r.estimate), float(r.se), flag)
        except Exception as exc:
            print(f"  placebo c={c}: skipped ({exc})")

    # ---- 7. covariate balance -------------------------------------------
    section("7. Covariate continuity at the cutoff")
    # District characteristics are fixed before the election is decided, so they
    # must be continuous at the cutoff.  A jump here says the running variable is
    # picking up something other than the electoral coin-flip.
    #
    # NOTE the contrast with the sharpness check at the top: `democrat` SHOULD
    # jump discontinuously -- that is the design working, not a balance failure.
    # Only *predetermined* variables belong in a balance table.
    for cov in ["totpop", "medianincome", "votingpop"]:
        sub = df_cov.dropna(subset=[cov])
        r = sp.rdrobust(data=sub, y=cov, x="lagdemvoteshare", c=CUTOFF)
        t = abs(float(r.estimate) / float(r.se))
        flag = "  <-- IMBALANCE" if t > 1.96 else "  ok"
        report(f"jump in {cov}", float(r.estimate), float(r.se), flag)

    print(
        "\n  Report as a package: density test, RD plot, MSE-optimal point estimate with\n"
        "  the robust bias-corrected CI, bandwidth sweep, polynomial sweep, placebo\n"
        "  cutoffs, covariate continuity.  Any one of them alone is not an RDD paper."
    )


if __name__ == "__main__":
    main()
