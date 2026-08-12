"""Potential outcomes: the decomposition every estimator is trying to kill.

Mixtape Ch.4.  This is the layer the rest of the repository sits on, and the one
most code-first treatments skip straight past.  Nothing here needs a dataset:
we build a population where both potential outcomes are known, so every quantity
can be computed exactly and the central identity can be *verified*, not asserted.

Definitions, with Y(1) and Y(0) the potential outcomes and D the treatment:

    ATE  = E[Y(1) - Y(0)]                  everyone
    ATT  = E[Y(1) - Y(0) | D = 1]          the treated
    ATU  = E[Y(1) - Y(0) | D = 0]          the untreated

The fundamental problem of causal inference is that you never observe both
Y(1) and Y(0) for the same unit.  What you can compute from data is the simple
difference in observed outcomes:

    SDO = E[Y | D=1] - E[Y | D=0]

and the Mixtape's central decomposition says

    SDO = ATE
        + (E[Y(0)|D=1] - E[Y(0)|D=0])      <- selection bias
        + (1 - pi) * (ATT - ATU)           <- heterogeneous treatment effect bias

where pi = P(D=1).  Every design in this repository is a strategy for zeroing
the second and third terms: randomisation kills both, DiD kills selection bias
in *changes*, IV recovers the ATT for compliers only, and so on.

Two assumptions are doing silent work throughout:

*   **SUTVA** -- no interference between units (my treatment does not change
    your outcome) and no hidden variation in treatment (one version of D).
    Spillovers and dosage heterogeneity both break it.
*   **Independence** -- (Y(0), Y(1)) independent of D.  Randomisation delivers
    this by construction; nothing else does without an argument.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import SEED, banner, read_mixtape, report, require, section

N = 200_000


def build_population(seed: int = SEED) -> pd.DataFrame:
    """A population with known Y(0), Y(1) and deliberate selection into treatment.

    Units with higher ability have better outcomes either way (selection bias)
    AND gain more from treatment (heterogeneous effects), and they are more
    likely to take it.  Both bias terms are therefore non-zero.
    """
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, N)
    y0 = 10 + 2.0 * ability + rng.normal(0, 1, N)
    gain = 3.0 + 1.5 * ability  # heterogeneous treatment effect
    y1 = y0 + gain
    # Selection: higher ability selects in.
    p = 1 / (1 + np.exp(-(0.8 * ability - 0.2)))
    d = rng.binomial(1, p)
    return pd.DataFrame(
        {"ability": ability, "y0": y0, "y1": y1, "d": d, "y": np.where(d == 1, y1, y0)}
    )


def main() -> None:
    banner("Potential outcomes -- ATE / ATT / ATU and the selection-bias identity")
    pop = build_population()

    # ---- 1. the quantities you could compute if you were omniscient -----
    section("1. Estimands (computable only because we simulated Y(0) and Y(1))")
    te = pop["y1"] - pop["y0"]
    ate = te.mean()
    att = te[pop["d"] == 1].mean()
    atu = te[pop["d"] == 0].mean()
    pi = pop["d"].mean()
    report("ATE  E[Y1-Y0]", ate)
    report("ATT  E[Y1-Y0 | D=1]", att)
    report("ATU  E[Y1-Y0 | D=0]", atu)
    print(f"  pi = P(D=1) = {pi:.4f}")
    print(
        "\n  These three differ whenever effects are heterogeneous AND selection is\n"
        "  informative.  'The' treatment effect is not a well-posed request; ask which."
    )

    # ---- 2. what the data actually gives you ----------------------------
    section("2. What a naive comparison delivers")
    sdo = pop.loc[pop["d"] == 1, "y"].mean() - pop.loc[pop["d"] == 0, "y"].mean()
    report("SDO  E[Y|D=1] - E[Y|D=0]", sdo, extra=f"[error vs ATE: {sdo - ate:+.5f}]")

    # ---- 3. the decomposition, verified ---------------------------------
    section("3. The decomposition -- verified numerically, not asserted")
    selection_bias = pop.loc[pop["d"] == 1, "y0"].mean() - pop.loc[pop["d"] == 0, "y0"].mean()
    het_bias = (1 - pi) * (att - atu)
    print(f"  ATE                                          {ate: .5f}")
    print(f"  + selection bias   E[Y0|D=1] - E[Y0|D=0]     {selection_bias: .5f}")
    print(f"  + het-effect bias  (1-pi)(ATT - ATU)         {het_bias: .5f}")
    print(f"  {'-' * 60}")
    print(f"  = reconstructed SDO                          {ate + selection_bias + het_bias: .5f}")
    print(f"    actual SDO                                 {sdo: .5f}")
    residual = abs(sdo - (ate + selection_bias + het_bias))
    print(f"    identity residual                          {residual:.2e}")
    require(residual < 1e-9, "the Mixtape decomposition must hold to machine precision")
    print(
        "\n  Read the middle line as the answer to 'what is endogeneity, numerically'.\n"
        "  Every design in this repository is an argument that one or both bias terms\n"
        "  is zero -- not a computation that makes them zero."
    )

    # ---- 4. randomisation kills both terms ------------------------------
    section("4. Randomise the same population and watch both bias terms vanish")
    rng = np.random.default_rng(SEED + 1)
    pop_r = pop.copy()
    pop_r["d"] = rng.binomial(1, 0.5, len(pop_r))
    pop_r["y"] = np.where(pop_r["d"] == 1, pop_r["y1"], pop_r["y0"])
    sdo_r = pop_r.loc[pop_r["d"] == 1, "y"].mean() - pop_r.loc[pop_r["d"] == 0, "y"].mean()
    sel_r = pop_r.loc[pop_r["d"] == 1, "y0"].mean() - pop_r.loc[pop_r["d"] == 0, "y0"].mean()
    te_r = pop_r["y1"] - pop_r["y0"]
    het_r = (1 - pop_r["d"].mean()) * (
        te_r[pop_r["d"] == 1].mean() - te_r[pop_r["d"] == 0].mean()
    )
    report("SDO under randomisation", sdo_r, extra=f"[error vs ATE: {sdo_r - ate:+.5f}]")
    print(f"  selection bias now  {sel_r:+.5f}   (was {selection_bias:+.5f})")
    print(f"  het-effect bias now {het_r:+.5f}   (was {het_bias:+.5f})")
    print(
        "  Independence is what randomisation buys.  Note it also collapses ATT, ATU\n"
        "  and ATE onto the same number -- which is why RCT papers can say 'the effect'."
    )

    # ---- 5. conditioning on observables ---------------------------------
    section("5. Selection on OBSERVABLES: condition on ability and it comes back")
    import statsmodels.formula.api as smf

    naive = smf.ols("y ~ d", data=pop).fit(cov_type="HC1")
    adjusted = smf.ols("y ~ d + ability", data=pop).fit(cov_type="HC1")
    report("OLS y ~ d", naive.params["d"], naive.bse["d"], extra=f"[error {naive.params['d'] - ate:+.4f}]")
    report("OLS y ~ d + ability", adjusted.params["d"], adjusted.bse["d"],
           extra=f"[error {adjusted.params['d'] - ate:+.4f}]")
    print(f"\n  for reference: ATE {ate:.4f}   ATT {att:.4f}   ATU {atu:.4f}")
    print(
        "\n  Controlling for the TRUE confounder removes almost all of the selection-bias\n"
        "  term -- the error drops from ~2.0 to ~0.1.  But it does not land exactly on\n"
        "  any estimand: with effects that vary in ability and no interaction term, OLS\n"
        "  returns a variance-weighted average of individual effects, not the ATE, the\n"
        "  ATT, or the ATU.  If you want a named estimand, target it explicitly\n"
        "  (interact the treatment, or weight) rather than hoping OLS delivers it."
    )

    # ---- 6. the book's subclassification example ------------------------
    section("6. Subclassification on the Mixtape's training example")
    tr = read_mixtape("training_example.dta")
    # Some columns arrive as strings with blanks for the shorter arms; coerce.
    num = tr.apply(pd.to_numeric, errors="coerce")
    raw = num["earnings_treat"].mean() - num["earnings_control"].mean()
    matched = num["earnings_treat"].mean() - num["earnings_matched"].mean()
    tr = num
    print(f"  mean age, treated  {tr['age_treat'].mean():.1f}")
    print(f"  mean age, control  {tr['age_control'].mean():.1f}   <- older, and age drives earnings")
    print(f"  mean age, matched  {tr['age_matched'].mean():.1f}   <- matched on age")
    report("raw difference (treated vs all controls)", raw)
    report("difference vs age-matched controls", matched)
    print(
        "  Matching on the confounder flips the sign of the naive comparison.  This is\n"
        "  the same identity as section 3, done by hand on 10 observations."
    )


if __name__ == "__main__":
    main()
