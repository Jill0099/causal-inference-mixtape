"""IV / 2SLS on Card (1995) college proximity, plus the two classic IV bugs.

Data: ``card.dta``.  Outcome ``lwage``, endogenous regressor ``educ``,
instrument ``nearc4`` (grew up near a four-year college).

Beyond mechanics, this file demonstrates two errors that survive code review
because they produce plausible numbers:

1.  **Manual two-step 2SLS gives the wrong standard errors.**  Regressing the
    endogenous variable on the instruments, then plugging the fitted values into
    a second OLS, reproduces the 2SLS *point estimate* exactly -- and then
    computes residuals against the fitted regressor instead of the actual one.
    The variance formula is simply the wrong one; the sign of the error depends
    on the data, so you cannot even sign your mistake.  Use a 2SLS routine.
2.  **The forbidden regression.**  When the endogenous regressor is binary it is
    tempting to run a logit/probit first stage and plug in the predicted
    probability.  This is inconsistent unless the first stage happens to be
    exactly right, because the projection is no longer linear and the residual
    is no longer orthogonal to the instruments.  The correct move is either
    plain linear 2SLS, or using the nonlinear fitted value as an *instrument*
    (not as a regressor) so consistency does not depend on the first-stage
    functional form.

Also covered: the weak-instrument threshold.  "F > 10" is an old relative-bias
rule of thumb; in the single-instrument model studied by Lee, McCrary, Moreira
& Porter (2022), conventional 5% t-ratio inference needs a first-stage F above
roughly 104.7, or an adjusted critical value.  This is not a universal cutoff
for multiple-instrument, heteroskedastic, or clustered designs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import banner, design_facts, read_mixtape, report, require, section

CONTROLS = ["exper", "expersq", "black", "south", "smsa"]


def load() -> pd.DataFrame:
    df = read_mixtape("card.dta")
    cols = ["lwage", "educ", "nearc4", "nearc2", *CONTROLS]
    return df[cols].dropna().copy()


def main() -> None:
    banner("IV / 2SLS -- Card (1995), returns to schooling")
    df = load()
    design_facts(df, outcome="lwage", treat_mask=pd.Series(df["nearc4"] > 0))

    import statsmodels.formula.api as smf
    import statspai as sp
    from linearmodels.iv import IV2SLS

    ctrl = " + ".join(CONTROLS)

    # ---- 0. OLS benchmark ------------------------------------------------
    section("0. OLS benchmark (biased if schooling is endogenous)")
    ols = smf.ols(f"lwage ~ educ + {ctrl}", data=df).fit(cov_type="HC1")
    report("OLS educ", ols.params["educ"], ols.bse["educ"])

    # ---- 1. first stage --------------------------------------------------
    section("1. First stage -- report this table, always")
    fs = smf.ols(f"educ ~ nearc4 + {ctrl}", data=df).fit(cov_type="HC1")
    report("nearc4 -> educ", fs.params["nearc4"], fs.bse["nearc4"])
    ftest = fs.f_test("nearc4 = 0")
    fstat = float(np.squeeze(ftest.fvalue))
    print(f"  First-stage F on the excluded instrument: {fstat:.2f}")
    verdict = (
        "passes the F>10 rule but FAILS the single-IV Lee et al. (2022) F>104.7 bar"
        if 10 < fstat < 104.7
        else ("passes both stated thresholds" if fstat >= 104.7 else "fails both stated thresholds")
    )
    print(f"  -> {verdict}")
    print(
        "  In this single-instrument setting, 10 < F < 104.7 does not justify an\n"
        "  unadjusted 5% t-test.  Report tF-adjusted or Anderson-Rubin inference;\n"
        "  use design-appropriate weak-IV diagnostics in other IV settings."
    )

    # ---- 2. reduced form -------------------------------------------------
    section("2. Reduced form -- the instrument's total effect on the outcome")
    rf = smf.ols(f"lwage ~ nearc4 + {ctrl}", data=df).fit(cov_type="HC1")
    report("nearc4 -> lwage", rf.params["nearc4"], rf.bse["nearc4"])
    print(
        f"  Indirect check: reduced form / first stage = "
        f"{rf.params['nearc4'] / fs.params['nearc4']:.5f}"
    )
    print("  With one instrument this ratio IS the 2SLS estimate (Wald / indirect LS).")

    # ---- 3. proper 2SLS --------------------------------------------------
    section("3. Proper 2SLS")
    iv = IV2SLS.from_formula(f"lwage ~ 1 + {ctrl} + [educ ~ nearc4]", data=df).fit(cov_type="robust")
    b_iv, se_iv = float(iv.params["educ"]), float(iv.std_errors["educ"])
    report("linearmodels IV2SLS educ", b_iv, se_iv)

    sp_iv = sp.ivreg(f"lwage ~ (educ ~ nearc4) + {ctrl}", data=df, robust="hc1")
    tid = sp_iv.tidy().set_index("term")
    report("StatsPAI sp.ivreg educ", float(tid.loc["educ", "estimate"]), float(tid.loc["educ", "std_error"]))
    require(
        abs(float(tid.loc["educ", "estimate"]) - b_iv) < 1e-6,
        "StatsPAI and linearmodels must agree on the 2SLS point estimate",
    )
    print(sp_iv.summary())

    # ---- 4. BUG DEMO: manual two-step -----------------------------------
    section("4. BUG DEMO -- manual two-step 2SLS: right beta, wrong SE")
    tmp = df.copy()
    tmp["educ_hat"] = smf.ols(f"educ ~ nearc4 + {ctrl}", data=df).fit().fittedvalues
    # Compare under CLASSICAL standard errors, where the two variance formulas
    # differ analytically.  (Under HC1 the gap here happens to be tiny, which is
    # exactly why this bug survives code review.)
    iv_c = IV2SLS.from_formula(f"lwage ~ 1 + {ctrl} + [educ ~ nearc4]", data=df).fit(
        cov_type="unadjusted"
    )
    manual = smf.ols(f"lwage ~ educ_hat + {ctrl}", data=tmp).fit()
    b_man, se_man = manual.params["educ_hat"], manual.bse["educ_hat"]
    b_c, se_c = float(iv_c.params["educ"]), float(iv_c.std_errors["educ"])
    report("manual two-step (classical SE)", b_man, se_man)
    report("correct 2SLS    (classical SE)", b_c, se_c)
    require(abs(b_man - b_c) < 1e-6, "manual two-step must reproduce the 2SLS point estimate")
    print(
        f"  Point estimates agree to {abs(b_man - b_c):.2e}; the manual SE is "
        f"{100 * (se_man / se_c - 1):+.1f}% off.\n"
        "  Correct 2SLS forms residuals as y - X*beta using the ACTUAL regressor;\n"
        "  the manual route uses y - Xhat*beta, so the residual absorbs the first-\n"
        "  stage projection error.  Whether that inflates or deflates the SE depends\n"
        "  on the data -- you cannot even sign your own mistake.  Never hand-roll it."
    )

    # ---- 5. BUG DEMO: forbidden regression ------------------------------
    section("5. BUG DEMO -- the forbidden regression")
    # Make the endogenous variable binary so a nonlinear first stage is tempting:
    # "attended any college".
    tmp["college"] = (tmp["educ"] >= 13).astype(int)
    fs_bin = smf.ols(f"college ~ nearc4 + {ctrl}", data=tmp).fit()
    print(
        f"  binary treatment share {tmp['college'].mean():.3f}, "
        f"linear first-stage F {float(np.squeeze(fs_bin.f_test('nearc4 = 0').fvalue)):.1f}"
    )
    probit = smf.logit(f"college ~ nearc4 + {ctrl}", data=tmp).fit(disp=0)
    tmp["college_hat"] = probit.predict()

    forbidden = smf.ols(f"lwage ~ college_hat + {ctrl}", data=tmp).fit(cov_type="HC1")
    report("forbidden (plug logit fit in as regressor)", forbidden.params["college_hat"], forbidden.bse["college_hat"])

    linear_2sls = IV2SLS.from_formula(f"lwage ~ 1 + {ctrl} + [college ~ nearc4]", data=tmp).fit(cov_type="robust")
    report("correct: linear 2SLS", float(linear_2sls.params["college"]), float(linear_2sls.std_errors["college"]))

    # The legitimate way to exploit a nonlinear first stage (Wooldridge 2010,
    # procedure 21.1): use the fitted probability as an INSTRUMENT, not a regressor.
    tmp["z_hat"] = tmp["college_hat"]
    fs_zhat = smf.ols(f"college ~ z_hat + {ctrl}", data=tmp).fit()
    f_zhat = float(np.squeeze(fs_zhat.f_test("z_hat = 0").fvalue))
    correct_nl = IV2SLS.from_formula(f"lwage ~ 1 + {ctrl} + [college ~ z_hat]", data=tmp).fit(cov_type="robust")
    report(
        "sanctioned: logit fit used as INSTRUMENT",
        float(correct_nl.params["college"]),
        float(correct_nl.std_errors["college"]),
        f"[first-stage F on z_hat = {f_zhat:.1f}]",
    )
    print(
        f"  Forbidden ({forbidden.params['college_hat']:+.3f}) vs consistent linear 2SLS "
        f"({float(linear_2sls.params['college']):+.3f}):\n"
        "  different magnitude AND different sign.  The forbidden version is\n"
        "  inconsistent unless the logit happens to be exactly the right model, and\n"
        "  its standard error does not know that, so nothing warns you.\n\n"
        "  Lines 2 and 3 are both legitimate and both weak here (first-stage F around\n"
        f"  10-15), so they sit about two standard errors apart -- sampling noise, plus\n"
        "  the fact that a nonlinear instrument reweights compliers differently.  With\n"
        "  instruments this weak neither is worth reporting as a point estimate; the\n"
        "  lesson is the contrast with line 1, not the gap between lines 2 and 3."
    )

    # ---- 6. over-identification + LATE framing ---------------------------
    section("6. Two instruments: over-identification test")
    iv2 = IV2SLS.from_formula(f"lwage ~ 1 + {ctrl} + [educ ~ nearc4 + nearc2]", data=df).fit(cov_type="robust")
    report("2SLS with nearc4 + nearc2", float(iv2.params["educ"]), float(iv2.std_errors["educ"]))
    print(f"  Sargan/Wooldridge overid: {iv2.wooldridge_overid}")
    print(
        "\n  A passing overid test is NOT a validity test: it only says the instruments\n"
        "  agree with each other.  If both violate exclusion in the same direction it\n"
        "  passes happily.  The exclusion restriction is defended in prose, not code."
    )

    section("7. What did we actually estimate?")
    print(
        "  Under heterogeneous effects, 2SLS identifies a LATE: the return to schooling\n"
        "  for COMPLIERS -- people who got more education because they grew up near a\n"
        "  college and would not have otherwise.  This is neither the ATE nor the ATT.\n"
        f"  Compliers are a {fs.params['nearc4']:.3f}-year-of-schooling slice of the sample.\n"
        "  Requires monotonicity: no 'defiers' who get LESS schooling from proximity.\n"
        f"  Note IV ({b_iv:.4f}) > OLS ({ols.params['educ']:.4f}) -- if simple ability bias\n"
        "  were the whole story you would expect the opposite, which is itself a clue\n"
        "  that the complier population differs from the average worker."
    )


if __name__ == "__main__":
    main()
