"""Randomisation inference on Thornton (2008) HIV-testing incentives.

Data: ``thornton_hiv.dta``.  Outcome ``got`` (returned to learn the HIV result),
treatment ``any`` (received any cash incentive).  Randomised, so the difference
in means is unbiased -- the question here is how to do *inference*.

Randomisation inference tests a different null than a t-test:

    Fisher sharp null   H0: Y_i(1) = Y_i(0) for EVERY unit i
    Neyman null         H0: E[Y(1) - Y(0)] = 0 on average

Under the sharp null every potential outcome is known, so the randomisation
distribution of any test statistic can be enumerated (or sampled) exactly.  That
makes RI valid in small samples, with no asymptotics, no distributional
assumption, and no cluster-count requirement -- which is exactly where
cluster-robust standard errors fall apart.

The p-value must include the observed assignment:

    p = (1 + #{|t*| >= |t_obs|}) / (1 + B)

Dropping the "+1" makes p = 0 possible, which is not a probability any
permutation test can produce.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from _common import SEED, banner, design_facts, read_mixtape, report, require, section


def load() -> pd.DataFrame:
    df = read_mixtape("thornton_hiv.dta")
    return df[["got", "any", "villnum", "tinc", "age", "male"]].dropna(subset=["got", "any"]).copy()


def ri_pvalue(
    y: np.ndarray,
    d: np.ndarray,
    *,
    n_perm: int = 10_000,
    clusters: np.ndarray | None = None,
    seed: int = SEED,
) -> tuple[float, float, np.ndarray]:
    """Two-sided RI p-value for the difference in means under the sharp null.

    ``clusters`` permutes assignment at the cluster level, which is what you must
    do when treatment was assigned to groups rather than individuals.
    """
    rng = np.random.default_rng(seed)
    observed = y[d == 1].mean() - y[d == 0].mean()

    if clusters is None:
        pool = d.copy()
        draws = np.empty(n_perm)
        for b in range(n_perm):
            perm = rng.permutation(pool)
            draws[b] = y[perm == 1].mean() - y[perm == 0].mean()
    else:
        uniq = np.unique(clusters)
        # Cluster-level treatment status; requires treatment constant within cluster.
        cl_treat = np.array([d[clusters == g][0] for g in uniq])
        draws = np.empty(n_perm)
        for b in range(n_perm):
            perm_cl = rng.permutation(cl_treat)
            mapping = dict(zip(uniq, perm_cl))
            perm = np.array([mapping[g] for g in clusters])
            draws[b] = y[perm == 1].mean() - y[perm == 0].mean()

    # The +1 in numerator and denominator counts the observed assignment, which
    # IS one of the possible randomisations.  Without it p can hit exactly 0.
    p = (1 + np.sum(np.abs(draws) >= np.abs(observed))) / (1 + n_perm)
    return float(observed), float(p), draws


def main() -> None:
    banner("Randomisation inference -- Thornton (2008) HIV testing")
    df = load()
    design_facts(df, outcome="got", treat_mask=df["any"], cluster="villnum")

    y = df["got"].to_numpy(float)
    d = df["any"].to_numpy(int)

    # ---- 1. the estimate --------------------------------------------------
    section("1. Difference in means (unbiased: this was randomised)")
    import statsmodels.formula.api as smf

    ols = smf.ols("got ~ any", data=df).fit(cov_type="HC1")
    report("ATE", ols.params["any"], ols.bse["any"], extra=f"asymptotic p={ols.pvalues['any']:.4f}")

    # ---- 2. RI, individual-level permutation -----------------------------
    section("2. Randomisation inference, individual-level permutation")
    obs, p_ri, draws = ri_pvalue(y, d, n_perm=10_000)
    print(f"  observed difference     {obs:+.5f}")
    print(f"  RI p-value (B=10,000)   {p_ri:.5f}")
    print(f"  null distribution       mean {draws.mean():+.5f}, sd {draws.std():.5f}")
    print(f"  95% of the null mass in [{np.percentile(draws, 2.5):+.4f}, {np.percentile(draws, 97.5):+.4f}]")
    require(abs(obs - ols.params["any"]) < 1e-9, "RI statistic must equal the OLS contrast")

    # ---- 3. the +1 bug ----------------------------------------------------
    section("3. Why the +1 matters")
    naive_p = np.mean(np.abs(draws) >= abs(obs))
    print(f"  p WITHOUT the +1 correction: {naive_p:.5f}")
    print(f"  p WITH    the +1 correction: {p_ri:.5f}")
    print(
        "  Here the effect is enormous so both round to ~0, but the uncorrected\n"
        "  formula can return exactly 0.00000 -- a p-value no permutation test can\n"
        "  produce, since the observed assignment is always in the reference set.\n"
        f"  The smallest attainable p with B={10_000:,} draws is {1 / (1 + 10_000):.5f}."
    )

    # ---- 4. cluster-level permutation ------------------------------------
    section("4. When assignment is clustered, permute clusters")
    # Thornton randomised at the individual level within villages, so this is a
    # demonstration of the mechanics rather than the correct analysis here.
    const = df.groupby("villnum")["any"].nunique().max() == 1
    print(f"  treatment constant within village? {bool(const)}")
    if const:
        obs_c, p_c, _ = ri_pvalue(y, d, n_perm=2_000, clusters=df["villnum"].to_numpy())
        print(f"  cluster-permuted RI p-value: {p_c:.5f}")
    else:
        print(
            "  It is not, so individual-level permutation is the right reference\n"
            "  distribution here.  Permute at whatever level was actually randomised;\n"
            "  permuting individuals when villages were assigned understates the\n"
            "  null variance and manufactures significance."
        )

    # ---- 5. StatsPAI ------------------------------------------------------
    import statspai as sp

    section("5. StatsPAI sp.ri_test")
    r = sp.ri_test(data=df, y="got", treat="any", stat="diff_means", n_perms=10_000, seed=SEED)
    # sp.ri_test returns a plain dict, not a result object -- print it as such.
    payload = r if isinstance(r, dict) else r.to_dict()
    for k, v in payload.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"  {k:<28} {v}")
    sp_p = payload.get("p_value", payload.get("pvalue"))
    if sp_p is not None:
        print(f"\n  hand-rolled RI p = {p_ri:.5f}   StatsPAI RI p = {float(sp_p):.5f}")
        require(
            abs(float(sp_p) - p_ri) < 0.01,
            "StatsPAI and the hand-rolled permutation test should broadly agree",
        )
        if float(sp_p) == 0.0:
            print(
                "\n  NOTE (verified against StatsPAI 1.21): sp.ri_test reports exactly\n"
                "  0.0 here, i.e. it uses #{|t*| >= |t|} / B without the +1 correction.\n"
                "  The substantive conclusion is identical -- this effect is huge -- but\n"
                "  an exact zero is not an attainable permutation p-value.  Report it as\n"
                f"  'p < {1 / (1 + 10_000):.5f}' rather than 'p = 0.000'."
            )

    # ---- 6. plot ----------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(draws, bins=60, alpha=0.8)
        ax.axvline(obs, color="red", linewidth=2, label=f"observed = {obs:+.3f}")
        ax.set_xlabel("Difference in means under the sharp null")
        ax.set_ylabel("Frequency")
        ax.set_title("Randomisation distribution, Thornton (2008)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("ri_thornton.png", dpi=120)
        print("\n  wrote ri_thornton.png")
    except Exception as exc:
        print(f"\n  (plot skipped: {exc})")


if __name__ == "__main__":
    main()
