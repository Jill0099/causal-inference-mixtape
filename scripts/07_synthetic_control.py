"""Synthetic control on Texas prison capacity (Mixtape Ch.10).

Data: ``texas.dta`` -- a balanced 51-state x 16-year panel.  Treated unit is
Texas (statefip 48), treatment year 1993 (court-ordered prison expansion),
outcome ``bmprison`` (Black male prison population).

Two claims this file retires:

*   **"Python has no mature synthetic control, use rpy2."**  That was true for
    years and is no longer true.  ``StatsPAI`` ships ``sp.synth`` with the
    inference apparatus attached (placebo distribution, RMSPE ratios,
    leave-one-out) plus ``sp.sdid`` for synthetic DiD, all natively.  There is
    also ``scpi`` (Cattaneo and coauthors' own Python package, with prediction
    intervals) and ``pysyncon``.  The rpy2 bridge is a fallback, not the plan.
*   **"The gap plot is the result."**  It is not.  A synthetic control has no
    conventional standard error; inference comes from the *permutation
    distribution* of placebo gaps, summarised by the post/pre RMSPE ratio.  What
    you report is the treated unit's RANK in that distribution.

Performance note, measured on this machine with StatsPAI 1.21: fitting with
``covariates=`` on a 51-unit panel runs the nested V-weight optimisation and
takes minutes.  Dropping it (predicting from pre-treatment outcome lags, the
classic Abadie-Diamond-Hainmueller setup) takes about five seconds.  Start
without covariates, add them only if the pre-treatment fit demands it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import banner, read_mixtape, report, require, section

TREATED_STATE = 48  # Texas
TREAT_YEAR = 1993


def load() -> pd.DataFrame:
    df = read_mixtape("texas.dta")[["statefip", "year", "bmprison"]].copy()
    # Keep a balanced panel: donors observed in every year.  An unbalanced donor
    # pool is silently truncated by the optimiser and is then not the pool you
    # described in the paper.
    counts = df.dropna(subset=["bmprison"]).groupby("statefip")["year"].nunique()
    full = counts[counts == counts.max()].index
    out = df[df["statefip"].isin(full)].copy()
    require(
        not out.duplicated(["statefip", "year"]).any(),
        "synthetic control needs exactly one row per unit-period",
    )
    return out


def main() -> None:
    banner("Synthetic control -- Texas prison capacity, treated 1993")
    df = load()
    print(f"  Panel: {df['statefip'].nunique()} units x {df['year'].nunique()} years "
          f"({df['year'].min()}-{df['year'].max()})")
    print(f"  Treated unit: statefip {TREATED_STATE}, treatment year {TREAT_YEAR}")
    print(f"  Donor pool: {df['statefip'].nunique() - 1} states")
    print(f"  Pre-treatment periods: {int((df['year'] < TREAT_YEAR).sum() / df['statefip'].nunique())}")
    require(TREATED_STATE in set(df["statefip"]), "Texas must survive the balanced-panel filter")

    import statspai as sp

    # ---- 1. fit ----------------------------------------------------------
    section("1. Fit (placebo distribution computed alongside by default)")
    sc = sp.synth(
        data=df,
        outcome="bmprison",
        unit="statefip",
        time="year",
        treated_unit=TREATED_STATE,
        treatment_time=TREAT_YEAR,
        placebo=True,
    )
    print(sc.summary())

    # ---- 2. pre-treatment fit is the credibility test --------------------
    section("2. Pre-treatment fit -- the credibility test")
    # StatsPAI puts the fit statistics on .model_info, not .diagnostics.
    info = dict(sc.model_info or {})
    diag = dict(sc.diagnostics or {})
    pre = info.get("pre_treatment_rmse")
    scale = df.loc[(df["statefip"] == TREATED_STATE) & (df["year"] < TREAT_YEAR), "bmprison"].mean()
    print(f"  pre-treatment periods   {info.get('n_pre_periods')}")
    print(f"  post-treatment periods  {info.get('n_post_periods')}")
    print(f"  donors                  {info.get('n_donors')}")
    if pre is not None:
        print(f"  pre-treatment RMSE      {float(pre):,.2f}")
        print(f"  treated pre-mean        {scale:,.2f}")
        print(f"  RMSE as % of pre-mean   {100 * float(pre) / scale:.2f}%")
    print(f"  effective donors (1/HHI){diag.get('effective_n_donors', 'n/a'):>8}")
    print(
        "  A synthetic control that cannot track the treated unit BEFORE treatment\n"
        "  has no claim to track its counterfactual after.  If the pre-RMSPE is large\n"
        "  relative to the outcome's scale, stop -- do not report a gap plot."
    )

    # ---- 3. donor weights ------------------------------------------------
    section("3. Donor weights -- report them; sparsity is the selling point")
    # NB: .params on a synth result is the ATT, not the donor weights.  The
    # weights live on .model_info['weights'].
    raw_w = info.get("weights")
    w = None
    if isinstance(raw_w, pd.Series):
        w = raw_w
    elif isinstance(raw_w, dict):
        w = pd.Series(raw_w)
    elif raw_w is not None:
        arr = np.asarray(raw_w)
        if arr.ndim == 2 and arr.shape[1] == 2:  # (donor_id, weight) pairs
            w = pd.Series(arr[:, 1], index=arr[:, 0].astype(int))
        elif arr.ndim == 1:
            w = pd.Series(arr)
    if w is not None:
        w = w[w.abs() > 1e-4].sort_values(ascending=False)
        print("  statefip   weight")
        for unit, val in w.items():
            print(f"  {str(unit):<10} {val:.4f}")
        print(f"\n  {len(w)} donors carry non-trivial weight; top donor holds {w.iloc[0]:.1%}")
        print(f"  weights sum to {w.sum():.4f} (should be 1, and all non-negative)")
    print(
        "  Check for interpolation bias: if the optimiser leans on donors far from\n"
        "  the treated unit on the predictors, the 'counterfactual' is an\n"
        "  extrapolation dressed up as a weighted average."
    )

    # ---- 4. inference by permutation ------------------------------------
    section("4. Inference -- the placebo distribution IS the p-value")
    report("ATT (post-treatment average gap)", float(sc.estimate),
           float(sc.se) if np.isfinite(sc.se or np.nan) else None)
    print(f"  permutation p-value: {float(sc.pvalue):.4f}"
          if sc.pvalue is not None else "  permutation p-value unavailable")
    print(
        "\n  Interpretation: this is the fraction of donor states that, when falsely\n"
        "  assigned treatment in 1993, produced a post/pre RMSPE ratio at least as\n"
        "  extreme as Texas's.  With ~50 donors the finest attainable p-value is\n"
        f"  about {1 / df['statefip'].nunique():.3f}; do not quote more precision than that."
    )
    if sc.pvalue is not None and sc.se:
        print(
            f"\n  READ THIS CONTRAST.  The ratio estimate/se is {float(sc.estimate) / float(sc.se):.2f},\n"
            f"  which looks overwhelming, while the permutation p-value is {float(sc.pvalue):.2f}.\n"
            "  The first number is not a valid test: with ONE treated unit there is no\n"
            "  sampling distribution to appeal to.  The permutation p-value is the\n"
            "  inference.  Quoting the t-ratio here would be a serious over-claim."
        )

    # ---- 5. time placebo --------------------------------------------------
    section("5. In-time placebo: pretend treatment happened earlier")
    fake_year = TREAT_YEAR - 3
    try:
        pre_only = df[df["year"] < TREAT_YEAR]
        sc_fake = sp.synth(
            data=pre_only, outcome="bmprison", unit="statefip", time="year",
            treated_unit=TREATED_STATE, treatment_time=fake_year, placebo=False,
        )
        report(f"placebo ATT at fake treatment year {fake_year}", float(sc_fake.estimate))
        print(
            "  This should be near zero.  A large 'effect' before anything happened\n"
            "  means the pre-treatment fit is riding a trend, not a match."
        )
    except Exception as exc:
        print(f"  (in-time placebo unavailable: {type(exc).__name__}: {str(exc)[:110]})")

    # ---- 6. synthetic DiD -------------------------------------------------
    section("6. Synthetic DiD (Arkhangelsky et al. 2021) as a robustness check")
    try:
        sd = sp.sdid(
            data=df, outcome="bmprison", unit="statefip", time="year",
            treated_unit=TREATED_STATE, treatment_time=TREAT_YEAR,
        )
        report("sdid ATT", float(sd.estimate), float(sd.se))
        print(
            "  SDiD adds a level shift and time weights, so it does not require exact\n"
            "  pre-treatment fit.  Agreement between synth and sdid is real reassurance;\n"
            "  disagreement usually means the pre-fit was doing too much work."
        )
    except Exception as exc:
        print(f"  (sdid unavailable: {type(exc).__name__}: {str(exc)[:110]})")

    section("What to report")
    print(
        "  1. Path plot (treated vs synthetic over the full period)\n"
        "  2. Gap plot with the treatment date marked\n"
        "  3. Donor weight table\n"
        "  4. Pre-treatment RMSPE, in units the reader can scale\n"
        "  5. Placebo distribution and the treated unit's RANK -- this is the p-value\n"
        "  6. Leave-one-out donor sensitivity and an in-time placebo\n"
        "  A gap plot on its own is a picture, not evidence."
    )


if __name__ == "__main__":
    main()
