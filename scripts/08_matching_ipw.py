"""Matching, propensity scores and IPW on LaLonde/NSW (Mixtape Ch.5).

Data: ``nsw_mixtape.dta`` (the randomised NSW job-training experiment) and
``cps_mixtape.dta`` (a non-experimental CPS comparison pool).

This is the best teaching dataset in applied econometrics because it has a known
right answer: the experimental contrast is unbiased by construction.  Swapping in
the CPS controls creates a large, entirely artificial "effect", and the job of
every selection-on-observables method is to claw its way back to the experimental
benchmark.  Every estimator below is scored against that benchmark.

Points the original template missed:

*   **Common support is a precondition, not a robustness check.**  If treated and
    control propensity-score distributions barely overlap, no weighting scheme
    rescues you; it just moves the extrapolation somewhere less visible.
*   **Untrimmed IPW explodes.**  A control unit with pscore 0.999 gets weight
    1000.  Trim, or use normalised (Hajek) weights, and say which you did.
*   **Matching standard errors are not OLS standard errors.**  Abadie & Imbens
    (2006, 2011) show the naive post-match regression SE is wrong, and that
    nearest-neighbour matching on more than one continuous covariate carries an
    asymptotic bias term that does not vanish -- hence the bias correction.
*   **Propensity-score MATCHING specifically is fragile.**  King & Nielsen (2019)
    show it can increase imbalance as you prune.  Prefer weighting, or match on
    covariates directly with bias correction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import banner, read_mixtape, report, require, section

COVS = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    nsw = read_mixtape("nsw_mixtape.dta")
    cps = read_mixtape("cps_mixtape.dta")
    return nsw, cps


def abadie_imbens_att(
    df: pd.DataFrame, *, y: str, treat: str, covs: list[str], M: int = 1
) -> tuple[float, float]:
    """Nearest-neighbour ATT with and without the Abadie-Imbens bias correction.

    Implemented by hand because the correction is the lesson.  For each treated
    unit i we find its M nearest controls on covariates standardised by their
    own standard deviation (the diagonal-Mahalanobis metric of Abadie & Imbens
    2006), then form

        naive:      mean_i [ Y_i - (1/M) sum_j Y_j ]
        corrected:  mean_i [ Y_i - (1/M) sum_j ( Y_j + mu0(X_i) - mu0(X_j) ) ]

    where ``mu0`` is a regression of the outcome on covariates fit on controls
    only.  The extra term removes the first-order bias caused by the fact that
    with more than one continuous covariate the matches are never exact, and
    the discrepancy shrinks too slowly for the bias to vanish asymptotically.

    Returns ``(naive_att, bias_corrected_att)``.
    """
    from scipy.spatial import cKDTree

    treated = df[df[treat] == 1]
    control = df[df[treat] == 0]

    scale = df[covs].std(ddof=0).replace(0, 1.0)
    Xt = (treated[covs] / scale).to_numpy(float)
    Xc = (control[covs] / scale).to_numpy(float)

    _, idx = cKDTree(Xc).query(Xt, k=M)
    idx = np.atleast_2d(idx.T).T if M > 1 else idx.reshape(-1, 1)

    yt = treated[y].to_numpy(float)
    yc = control[y].to_numpy(float)
    matched_y = yc[idx].mean(axis=1)
    naive = float(np.mean(yt - matched_y))

    # mu0: outcome model estimated on CONTROLS only.
    import statsmodels.api as sm

    Xc_design = sm.add_constant(control[covs].to_numpy(float))
    mu0 = sm.OLS(yc, Xc_design).fit()
    pred_t = mu0.predict(sm.add_constant(treated[covs].to_numpy(float), has_constant="add"))
    pred_c = mu0.predict(Xc_design)
    matched_pred = pred_c[idx].mean(axis=1)

    corrected = float(np.mean(yt - matched_y - (pred_t - matched_pred)))
    return naive, corrected


def main() -> None:
    banner("Matching / PSM / IPW -- LaLonde NSW vs CPS")
    nsw, cps = load()

    # ---- 0. the benchmark ------------------------------------------------
    section("0. Experimental benchmark (randomisation => this is the truth)")
    t = nsw.loc[nsw["treat"] == 1, "re78"].mean()
    c = nsw.loc[nsw["treat"] == 0, "re78"].mean()
    benchmark = t - c
    print(f"  E[re78 | treated]            {t:>10,.2f}   (n={int((nsw.treat == 1).sum())})")
    print(f"  E[re78 | experimental ctrl]  {c:>10,.2f}   (n={int((nsw.treat == 0).sum())})")
    report("EXPERIMENTAL ATT", benchmark)
    require(1500 < benchmark < 2100, "NSW experimental effect should be near $1,800")

    # ---- 1. break it -----------------------------------------------------
    section("1. Replace the control group with CPS -- selection bias, visible")
    obs = pd.concat([nsw[nsw["treat"] == 1], cps.assign(treat=0)], ignore_index=True)
    naive = obs.loc[obs["treat"] == 1, "re78"].mean() - obs.loc[obs["treat"] == 0, "re78"].mean()
    report("naive difference in means", naive, extra=f"[bias {naive - benchmark:+,.0f}]")
    print(
        "  The CPS pool is older, better educated and far richer at baseline.  Nothing\n"
        "  about the programme changed; only the comparison group did."
    )

    import statsmodels.formula.api as smf
    import statspai as sp

    # ---- 2. regression adjustment ---------------------------------------
    section("2. Regression adjustment (linear control for covariates)")
    reg = smf.ols("re78 ~ treat + " + " + ".join(COVS), data=obs).fit(cov_type="HC1")
    report("OLS with covariates", reg.params["treat"], reg.bse["treat"],
           extra=f"[bias {reg.params['treat'] - benchmark:+,.0f}]")

    # ---- 3. propensity score & overlap ----------------------------------
    section("3. Propensity score: estimate, then CHECK OVERLAP before using it")
    ps_formula = "treat ~ " + " + ".join(
        COVS + ["I(age**2)", "I(educ**2)", "I(re74**2)", "I(re75**2)"]
    )
    ps_model = smf.logit(ps_formula, data=obs).fit(disp=0)
    obs = obs.assign(pscore=ps_model.predict())

    q = obs.groupby("treat")["pscore"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    print(q[["count", "min", "1%", "5%", "50%", "95%", "99%", "max"]].to_string())
    lo = obs.loc[obs["treat"] == 1, "pscore"].min()
    hi = obs.loc[obs["treat"] == 1, "pscore"].max()
    in_support = obs["pscore"].between(lo, hi)
    print(f"\n  Treated pscore support: [{lo:.4f}, {hi:.4f}]")
    print(f"  Controls inside that support: {int((in_support & (obs.treat == 0)).sum()):,} "
          f"of {int((obs.treat == 0).sum()):,} "
          f"({100 * (in_support & (obs.treat == 0)).mean() / (obs.treat == 0).mean():.1f}%)")
    print(
        "  Most CPS controls are nowhere near the treated units.  That is the real\n"
        "  finding here -- any estimator that ignores it is extrapolating."
    )

    # ---- 4. IPW, with and without trimming -------------------------------
    section("4. IPW -- untrimmed vs trimmed vs normalised")
    d = obs.copy()
    d["w_ate"] = np.where(d["treat"] == 1, 1 / d["pscore"], 1 / (1 - d["pscore"]))
    print(f"  max weight, untrimmed: {d['w_ate'].max():,.1f}  "
          f"(a single observation carrying {100 * d['w_ate'].max() / d['w_ate'].sum():.1f}% of total weight)")

    untrimmed = smf.wls("re78 ~ treat", data=d, weights=d["w_ate"]).fit(cov_type="HC1")
    report("IPW, untrimmed", untrimmed.params["treat"], untrimmed.bse["treat"],
           extra=f"[bias {untrimmed.params['treat'] - benchmark:+,.0f}]")

    trim = d[d["pscore"].between(0.1, 0.9)]
    tr = smf.wls("re78 ~ treat", data=trim, weights=trim["w_ate"]).fit(cov_type="HC1")
    report("IPW, trimmed to [0.1, 0.9]", tr.params["treat"], tr.bse["treat"],
           extra=f"[bias {tr.params['treat'] - benchmark:+,.0f}]  n={len(trim):,}")

    # Note what the block above actually did: `trim` DISCARDS rows outside
    # [0.1, 0.9].  Contrast that with a library that CLIPS instead.
    # StatsPAI's ipw: Hajek weights, bootstrap SEs, and its own ``trim``.
    sp_ipw0 = sp.ipw(data=obs, y="re78", treat="treat", covariates=COVS,
                     estimand="ATT", trim=0.0, n_bootstrap=200, seed=20060815)
    sp_ipw1 = sp.ipw(data=obs, y="re78", treat="treat", covariates=COVS,
                     estimand="ATT", trim=0.1, n_bootstrap=200, seed=20060815)
    report("sp.ipw ATT, trim=0.0", float(sp_ipw0.estimate), float(sp_ipw0.se),
           extra=f"[bias {float(sp_ipw0.estimate) - benchmark:+,.0f}]")
    report("sp.ipw ATT, trim=0.1", float(sp_ipw1.estimate), float(sp_ipw1.se),
           extra=f"[bias {float(sp_ipw1.estimate) - benchmark:+,.0f}]")

    # KNOW WHICH TRIMMING YOUR LIBRARY DOES.  Verified against StatsPAI 1.21:
    # ``trim`` CLIPS the propensity score into [trim, 1-trim] and keeps every row.
    info = sp_ipw1.model_info or {}
    print(
        f"\n  sp.ipw(trim=0.1) kept n_obs={sp_ipw1.n_obs:,} and moved pscore_min to "
        f"{info.get('pscore_min'):.4f}.\n"
        f"  The manual DROP above kept only n={len(trim):,} rows and gives "
        f"{tr.params['treat']:+,.0f}.\n"
        "  These are two different estimands, not two implementations of one:\n"
        "    * CLIPPING censors the weight of off-support controls but still averages\n"
        "      over them -- here it drags the ATT far NEGATIVE, because 15,000 CPS\n"
        "      controls whose true pscore is ~0 get their control weight raised to\n"
        "      p/(1-p) = 0.111 apiece.\n"
        "    * DROPPING (Crump et al. 2009) redefines the estimand to the overlap\n"
        "      population and lands within a few hundred dollars of the experiment.\n"
        "  Read your library's docstring, then verify it on data where you know the\n"
        "  answer.  'trim' is not a standardised word."
    )

    # ---- 5. balance ------------------------------------------------------
    section("5. Covariate balance -- the only evidence weighting worked")
    bal = sp.balance_diagnostics(data=trim, treatment="treat", covariates=COVS, weights=trim["w_ate"].to_numpy())
    print(bal.summary() if hasattr(bal, "summary") else bal)
    print(
        "  Rule of thumb: |standardised difference| < 0.1 after weighting.  A balance\n"
        "  table is mandatory; a p-value on a t-test of balance is not a substitute\n"
        "  (it conflates imbalance with sample size)."
    )

    # ---- 6. matching, with and without bias correction ------------------
    section("6. Nearest-neighbour matching, bias correction ON vs OFF")
    print(
        "  Abadie-Imbens: with k>1 continuous matching variables, exact matches do not\n"
        "  exist, the match discrepancy shrinks too slowly, and the resulting bias term\n"
        "  does NOT vanish asymptotically.  Bias-corrected matching regression-adjusts\n"
        "  each matched pair for the residual covariate gap."
    )
    naive_att, bc_att = abadie_imbens_att(obs, y="re78", treat="treat", covs=COVS, M=1)
    report("NN matching, bias correction OFF", naive_att,
           extra=f"[bias {naive_att - benchmark:+,.0f}]")
    report("NN matching, bias correction ON ", bc_att,
           extra=f"[bias {bc_att - benchmark:+,.0f}]")
    print(f"  Correction moved the estimate by {bc_att - naive_att:+,.0f}.")

    # StatsPAI's Stata-faithful psmatch2 for cross-checking the plain PSM route.
    try:
        pm = sp.psmatch2(data=obs, treat="treat", covariates=COVS, outcome="re78",
                         neighbor=1, common_support="minmax")
        report("sp.psmatch2 ATT (pscore NN, minmax support)", float(pm.att), float(pm.se),
               extra=f"[bias {float(pm.att) - benchmark:+,.0f}]")
    except Exception as exc:
        print(f"  sp.psmatch2: {type(exc).__name__}: {str(exc)[:110]}")
    print(
        "  King & Nielsen (2019): matching on the propensity score approximates a\n"
        "  completely randomised experiment, not a blocked one, so pruning can raise\n"
        "  imbalance.  Match on covariates directly (with bias correction) or weight."
    )

    # ---- 7. scoreboard ---------------------------------------------------
    section("7. Scoreboard against the experimental benchmark")
    print(f"  {'estimator':<34}{'estimate':>12}{'bias':>12}")
    print(f"  {'-' * 58}")
    for label, val in [
        ("EXPERIMENTAL (truth)", benchmark),
        ("naive diff-in-means", naive),
        ("OLS with covariates", reg.params["treat"]),
        ("IPW untrimmed", untrimmed.params["treat"]),
        ("IPW drop off-support [0.1,0.9]", tr.params["treat"]),
        ("sp.ipw ATT (clip) trim=0.0", float(sp_ipw0.estimate)),
        ("sp.ipw ATT (clip) trim=0.1", float(sp_ipw1.estimate)),
        ("NN match, bias correction OFF", naive_att),
        ("NN match, bias correction ON", bc_att),
    ]:
        print(f"  {label:<34}{val:>12,.0f}{val - benchmark:>+12,.0f}")
    print(
        "\n  Selection on observables is a strong assumption, not a technique.  When it\n"
        "  holds, these methods work; the NSW/CPS pairing is famous precisely because\n"
        "  it usually does not, and the scoreboard shows how far off you can land."
    )


if __name__ == "__main__":
    main()
