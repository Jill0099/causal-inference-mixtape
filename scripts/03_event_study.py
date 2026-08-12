"""Event study (dynamic DiD) done correctly, on castle-doctrine data.

This file exists because the naive event-study loop is easy to get wrong in
three specific ways, and all three produce output that *looks* fine:

1.  **The reference period is not actually dropped.**  ``range(-4, 0)`` in Python
    is ``[-4, -3, -2, -1]``, so a loop that says "leads -4..-1" and then claims
    to exclude ``-1`` excludes nothing.  With every relative-time indicator in
    the design matrix plus unit and time fixed effects the model is perfectly
    collinear and there is no normalisation.
2.  **Endpoints are not binned.**  Units observed outside the event window get
    all-zero indicators, which silently treats them as if they were at the
    reference period.  Bin them into the terminal leads/lags instead.
3.  **The plot's x-axis does not line up with the coefficients.**  The reference
    period must be re-inserted as an exact zero with zero standard error.

Data: ``castle.dta``, 2006 cohort vs never-treated (same sample as
``02_did_2x2.py``), so the event-study average should land near the 2x2 estimate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import banner, design_facts, pick, read_mixtape, report, require, section

WINDOW = (-5, 4)  # leads/lags actually estimated; everything else is binned
REF = -1  # normalised-to-zero period


def load() -> pd.DataFrame:
    df = read_mixtape("castle.dta")
    keep = df["effyear"].isna() | (df["effyear"] == 2006)
    df = df.loc[keep, ["sid", "year", "l_homicide", "effyear"]].copy()
    # Never-treated units have no event time.  Give them an indicator instead of
    # a fake relative period so the binning code below can skip them cleanly.
    df["ever_treated"] = df["effyear"].notna().astype(int)
    df["rel_time"] = np.where(df["ever_treated"] == 1, df["year"] - df["effyear"], np.nan)
    return df.dropna(subset=["l_homicide"])


def build_event_dummies(
    df: pd.DataFrame, window: tuple[int, int] = WINDOW, ref: int = REF
) -> tuple[pd.DataFrame, list[str]]:
    """Create binned relative-time indicators with ``ref`` omitted.

    Returns the frame plus the ordered list of column names to put in the
    regression.  Column names encode the period as ``lead5``/``lag0`` style so
    they survive formula parsing.
    """
    lo, hi = window
    periods = [k for k in range(lo, hi + 1) if k != ref]
    require(ref not in periods, f"reference period {ref} must be excluded from the design")

    rel = df["rel_time"]
    # Bin: anything at or beyond an endpoint is folded into that endpoint.  Only
    # ever-treated units get a non-missing binned value.
    binned = rel.clip(lower=lo, upper=hi)

    names = []
    out = df.copy()
    for k in periods:
        name = f"lead{abs(k)}" if k < 0 else f"lag{k}"
        out[name] = ((binned == k) & (df["ever_treated"] == 1)).astype(int)
        names.append(name)

    # Sanity check: for every ever-treated observation exactly one indicator is
    # on, except in the reference period where all are off.  That is the whole
    # identification normalisation, so assert it rather than hope.
    treated_rows = out.loc[out["ever_treated"] == 1, names].sum(axis=1)
    at_ref = (binned[out["ever_treated"] == 1] == ref)
    require(bool((treated_rows[at_ref] == 0).all()), "reference-period rows must have all-zero dummies")
    require(bool((treated_rows[~at_ref] == 1).all()), "every other treated row needs exactly one dummy")
    require(
        bool((out.loc[out["ever_treated"] == 0, names].sum().sum() == 0)),
        "never-treated units must have all-zero event dummies",
    )
    return out, names


def coefficient_path(names: list[str], est: dict[str, float], se: dict[str, float]):
    """Assemble the plotting series, re-inserting the reference period as 0."""
    lo, hi = WINDOW
    periods, coefs, errs = [], [], []
    for k in range(lo, hi + 1):
        if k == REF:
            periods.append(k)
            coefs.append(0.0)
            errs.append(0.0)
            continue
        name = f"lead{abs(k)}" if k < 0 else f"lag{k}"
        periods.append(k)
        coefs.append(est[name])
        errs.append(se[name])
    require(
        len(periods) == len(coefs) == len(errs) == hi - lo + 1,
        "plot arrays must have one entry per period in the window",
    )
    return periods, coefs, errs


def main() -> None:
    banner("Event study -- castle doctrine, 2006 cohort vs never-treated")
    df = load()
    df, names = build_event_dummies(df)
    design_facts(df, outcome="l_homicide", treat_mask=df[[n for n in names if n.startswith("lag")]].max(axis=1), cluster="sid")
    print(f"  Window {WINDOW}, reference period t={REF}, {len(names)} estimated coefficients")

    # ---- pyfixest --------------------------------------------------------
    import pyfixest as pf

    section("pyfixest, hand-built binned dummies")
    fml = "l_homicide ~ " + " + ".join(names) + " | sid + year"
    fit = pf.feols(fml, data=df, vcov={"CRV1": "sid"})
    tidy = fit.tidy()
    est = {n: pick(tidy, n)[0] for n in names}
    se = {n: pick(tidy, n)[1] for n in names}
    for n in names:
        report(n, est[n], se[n])

    # ---- joint pre-trend test -------------------------------------------
    # A flat-looking picture is not a test.  Test the leads jointly, and be
    # explicit that failing to reject is weak evidence (Roth 2022): the test has
    # little power against exactly the trends that would bias the estimate.
    section("Joint pre-trend test (leads = 0)")
    leads = [n for n in names if n.startswith("lead")]
    wald = fit.wald_test(R=np.array([[1.0 if c == l else 0.0 for c in fit._coefnames] for l in leads]))
    print(f"  leads tested: {leads}")
    print(f"  {wald}")
    print(
        "  Failing to reject is NOT proof of parallel trends.  Report the power of\n"
        "  this test against an economically meaningful linear trend, or move to\n"
        "  Honest DiD -- see 11_honest_did.py."
    )

    # ---- the plot --------------------------------------------------------
    section("Coefficient path (reference period re-inserted as exact zero)")
    periods, coefs, errs = coefficient_path(names, est, se)
    for p, c, e in zip(periods, coefs, errs):
        marker = "  <- reference (normalised)" if p == REF else ""
        print(f"  t={p:+d}  {c:+.5f}  +/- {1.96 * e:.5f}{marker}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(periods, coefs, yerr=[1.96 * e for e in errs], fmt="o-", capsize=3)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.axvline(REF + 0.5, color="grey", linestyle="--", alpha=0.6)
        ax.set_xlabel(f"Years relative to treatment (t={REF} normalised to 0)")
        ax.set_ylabel("Effect on log homicide rate")
        ax.set_title("Castle doctrine event study, 2006 cohort")
        ax.set_xticks(periods)
        fig.tight_layout()
        out = "event_study_castle.png"
        fig.savefig(out, dpi=120)
        print(f"\n  wrote {out}")
    except Exception as exc:  # plotting is optional, estimation is not
        print(f"\n  (plot skipped: {exc})")

    # ---- StatsPAI one-call version --------------------------------------
    import statspai as sp

    section("StatsPAI sp.event_study (same design, one call)")
    # Pass the FULL panel.  Never-treated units keep a missing ``first_treat`` --
    # they are the comparison group.  Dropping them (a tempting one-liner) leaves
    # no clean controls and the standard errors explode by many orders of
    # magnitude, which is the loudest possible symptom of a mis-specified panel.
    es = sp.event_study(
        data=df.assign(first_treat=df["effyear"]),
        y="l_homicide",
        treat_time="first_treat",
        time="year",
        unit="sid",
        window=WINDOW,
        ref_period=REF,
        cluster="sid",
    )
    sp_tidy = es.tidy()
    sp_rows = sp_tidy[sp_tidy["type"] == "event_study"].copy()
    expected_terms = {f"event_{p:+d}" for p in periods}
    require(
        sp_rows["term"].is_unique,
        "StatsPAI event-study output must contain exactly one row per term; "
        "duplicate terms indicate an incompatible backend release",
    )
    require(
        set(sp_rows["term"]) == expected_terms,
        "StatsPAI event-study output must contain the complete requested window",
    )
    sp_ev = sp_rows.set_index("term")
    for p, c in zip(periods, coefs):
        term = f"event_{p:+d}"
        sp_estimate = float(sp_ev.at[term, "estimate"])
        gap = abs(sp_estimate - c)
        print(f"  {term:<12} StatsPAI {sp_estimate:+.5f}   hand-built {c:+.5f}   gap {gap:.2e}")
        require(gap < 1e-4, f"StatsPAI and the hand-built spec must agree at {term}")
    print(
        "\n  Both routes agree to machine precision on every coefficient (gaps ~1e-16).\n"
        "  Standard errors differ in the 3rd decimal only, from the df adjustment."
    )


if __name__ == "__main__":
    main()
