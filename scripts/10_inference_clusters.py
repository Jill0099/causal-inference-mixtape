"""Clustered inference: the failure mode nobody writes a robustness table for.

Data: ``castle.dta``, the same 2006-cohort DiD sample as ``02_did_2x2.py``.

The applied literature has internalised "cluster your standard errors" and
stopped there.  Three things go wrong after that:

1.  **Too few clusters.**  Cluster-robust variance is consistent as the NUMBER
    OF CLUSTERS goes to infinity, not the number of observations.  With 42
    states -- or 20, or 6 -- the CRVE is downward biased and the t-statistic is
    not t-distributed.  Cameron-Gelbach-Miller (2008) and Cameron-Miller (2015)
    give the fix: the wild cluster bootstrap.
2.  **Serial correlation.**  Bertrand, Duflo & Mullainathan (2004) showed that
    DiD on long panels with serially correlated outcomes produces rejection
    rates of 45% at a nominal 5% level when SEs ignore it.  Clustering on the
    unit fixes this; collapsing to pre/post is the other classic remedy.
3.  **Clustering at the wrong level.**  Abadie, Athey, Imbens & Wooldridge
    (2023): cluster at the level of TREATMENT ASSIGNMENT.  Clustering finer than
    assignment understates uncertainty; clustering much coarser throws away
    precision for no design reason.

Everything below is Python-native.  The old advice to shell out to Stata's
``boottest`` for a wild cluster bootstrap is obsolete.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import SEED, banner, pick, read_mixtape, report, require, section


def load() -> pd.DataFrame:
    df = read_mixtape("castle.dta")
    keep = df["effyear"].isna() | (df["effyear"] == 2006)
    df = df.loc[keep, ["sid", "year", "l_homicide", "effyear"]].copy()
    df["treated"] = (df["effyear"] == 2006).astype(int)
    df["post"] = (df["year"] >= 2006).astype(int)
    df["treat_post"] = df["treated"] * df["post"]
    return df.dropna(subset=["l_homicide"])


def main() -> None:
    banner("Clustered inference -- how many clusters is enough?")
    df = load()
    import pyfixest as pf
    import statsmodels.formula.api as smf
    import statspai as sp

    n_clusters = df["sid"].nunique()
    print(f"  N = {len(df):,},  clusters = {n_clusters},  treated clusters = "
          f"{df.loc[df['treated'] == 1, 'sid'].nunique()}")

    # ---- 1. the ladder of standard errors --------------------------------
    section("1. The same coefficient under four variance estimators")
    base = "l_homicide ~ treat_post + C(sid) + C(year)"
    iid = smf.ols(base, data=df).fit()
    hc1 = smf.ols(base, data=df).fit(cov_type="HC1")
    cl = smf.ols(base, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["sid"]})
    report("iid (wrong: ignores within-state correlation)", iid.params["treat_post"], iid.bse["treat_post"])
    report("HC1 (still wrong: heteroskedasticity only)", hc1.params["treat_post"], hc1.bse["treat_post"])
    report("cluster by state (right level)", cl.params["treat_post"], cl.bse["treat_post"])
    print(
        f"  Clustering inflates the SE by {100 * (cl.bse['treat_post'] / iid.bse['treat_post'] - 1):.0f}%\n"
        "  relative to iid.  That gap IS the serial correlation BDM warned about."
    )

    # ---- 2. wrong-level clustering ---------------------------------------
    section("2. Clustering finer than assignment understates uncertainty")
    # Treatment is assigned at the STATE level.  Clustering by state-year (finer)
    # pretends each state-year is an independent draw.
    df2 = df.assign(state_year=df["sid"].astype(str) + "_" + df["year"].astype(str))
    fine = smf.ols(base, data=df2).fit(cov_type="cluster", cov_kwds={"groups": df2["state_year"]})
    report("cluster by state-year (TOO FINE)", fine.params["treat_post"], fine.bse["treat_post"])
    report("cluster by state (assignment level)", cl.params["treat_post"], cl.bse["treat_post"])
    print(
        f"  Clustering at the wrong, finer level shrinks the SE by "
        f"{100 * (1 - fine.bse['treat_post'] / cl.bse['treat_post']):.0f}%.\n"
        "  Abadie-Athey-Imbens-Wooldridge: match the cluster to the level at which\n"
        "  treatment was ASSIGNED, not to the finest identifier in your data."
    )

    # ---- 3. wild cluster bootstrap ---------------------------------------
    section("3. Wild cluster bootstrap (Cameron-Gelbach-Miller)")
    print(
        "  The CRVE t-statistic is not t-distributed with few clusters.  The wild\n"
        "  cluster bootstrap imposes the null, resamples Rademacher signs at the\n"
        "  CLUSTER level, and reads the p-value off the bootstrap distribution."
    )
    b_cl, se_cl = cl.params["treat_post"], cl.bse["treat_post"]
    print(f"\n  Analytic CRVE:  b={b_cl:+.5f}  se={se_cl:.5f}  t={b_cl / se_cl:.3f}  "
          f"p={cl.pvalues['treat_post']:.4f}")

    # sp.wild_cluster_bootstrap takes a plain list of regressor COLUMNS -- it has
    # no fixed-effect syntax.  Passing x=["treat_post"] alone silently estimates a
    # DIFFERENT model (no FE) and returns a beta that is not your DiD estimate.
    # Materialise the fixed effects as dummies so the bootstrap matches the spec.
    fe = pd.get_dummies(df[["sid", "year"]].astype(str), drop_first=True).astype(float)
    boot_df = pd.concat([df[["l_homicide", "treat_post", "sid"]].reset_index(drop=True),
                         fe.reset_index(drop=True)], axis=1)
    regressors = ["treat_post", *fe.columns.tolist()]
    try:
        wcb = sp.wild_cluster_bootstrap(
            data=boot_df, y="l_homicide", x=regressors, cluster="sid",
            test_var="treat_post", h0=0.0, n_boot=1999, weight_type="rademacher", seed=SEED,
        )
        payload = wcb if isinstance(wcb, dict) else wcb.to_dict()
        for k, v in payload.items():
            if isinstance(v, (int, float, str, bool)):
                print(f"  {k:<28} {v}")
        got = float(payload.get("beta_hat", np.nan))
        require(
            abs(got - b_cl) < 1e-6,
            f"bootstrap must be run on the SAME specification (got {got:.5f}, want {b_cl:.5f})",
        )
        if payload.get("p_boot") == 0.0:
            print(
                "\n  NOTE: p_boot reported as exactly 0.0 -- the bootstrap p-value omits\n"
                "  the +1 correction, same issue as in 09_randomization_inference.py.\n"
                f"  With B=1999 the smallest attainable value is {1 / 2000:.5f}."
            )
    except Exception as exc:
        print(f"  sp.wild_cluster_bootstrap: {type(exc).__name__}: {str(exc)[:160]}")

    # pyfixest exposes the same thing on a fitted model
    try:
        fit = pf.feols("l_homicide ~ treat_post | sid + year", data=df, vcov={"CRV1": "sid"})
        boot = fit.wildboottest(param="treat_post", reps=1999, seed=SEED)
        print(f"\n  pyfixest wildboottest:\n{boot}")
    except Exception as exc:
        print(f"  pyfixest wildboottest unavailable: {type(exc).__name__}: {str(exc)[:120]}")

    # ---- 4. how bad does it get? -----------------------------------------
    section("4. Shrinking the cluster count on purpose")
    print("  Same design, fewer states retained.  Watch the analytic p-value.")
    print(f"  {'clusters':>9}{'beta':>12}{'CRVE se':>12}{'CRVE p':>10}")
    rng = np.random.default_rng(SEED)
    treated_states = df.loc[df["treated"] == 1, "sid"].unique()
    control_states = df.loc[df["treated"] == 0, "sid"].unique()
    print(f"  ({len(treated_states)} treated and {len(control_states)} control states available; "
          "each row keeps the treated:control ratio)")
    for k in (42, 30, 20, 12, 8):
        # Subsample BOTH arms so the cluster count really falls; keeping all
        # treated states would floor the count at 13 and the sweep would be fake.
        n_t = max(2, round(k * len(treated_states) / df["sid"].nunique()))
        n_c = max(2, k - n_t)
        if n_t > len(treated_states) or n_c > len(control_states):
            continue
        sub_states = np.concatenate([
            rng.choice(treated_states, size=n_t, replace=False),
            rng.choice(control_states, size=n_c, replace=False),
        ])
        sub = df[df["sid"].isin(sub_states)]
        m = smf.ols(base, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["sid"]})
        print(f"  {sub['sid'].nunique():>9}{m.params['treat_post']:>12.5f}"
              f"{m.bse['treat_post']:>12.5f}{m.pvalues['treat_post']:>10.4f}")
    print(
        "  The point estimate wanders because the sample changes; what matters is that\n"
        "  the analytic p-value is a lottery once the cluster count gets small."
    )
    print(
        "\n  Rules of thumb: below ~42 clusters treat the analytic p-value as\n"
        "  suggestive only; below ~20 report the wild cluster bootstrap as the\n"
        "  headline; below ~10 consider randomisation inference instead, which needs\n"
        "  no asymptotics at all (see 09_randomization_inference.py)."
    )

    # ---- 5. two-way clustering -------------------------------------------
    section("5. Two-way clustering, when shocks are correlated in both dimensions")
    tw = pf.feols("l_homicide ~ treat_post | sid + year", data=df, vcov={"CRV1": "sid"})
    b1, s1 = pick(tw.tidy(), "treat_post")
    report("one-way, cluster by state", b1, s1)
    try:
        tw2 = pf.feols("l_homicide ~ treat_post | sid + year", data=df, vcov={"CRV1": "year"})
        b2, s2 = pick(tw2.tidy(), "treat_post")
        report("one-way, cluster by year", b2, s2)
    except Exception as exc:
        print(f"  year clustering unavailable: {exc}")
    print(
        "  With 11 years, clustering on year is itself a few-clusters problem.  Two-way\n"
        "  clustering needs BOTH dimensions to have many groups; it is not a free\n"
        "  upgrade.  If one dimension is small, cluster on the large one and defend\n"
        "  the other with a bootstrap or randomisation inference."
    )
    require(np.isfinite(b1) and np.isfinite(s1), "clustered estimation must return finite values")


if __name__ == "__main__":
    main()
