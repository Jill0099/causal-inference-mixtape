"""Collider bias, simulated: why 'add more controls' is not a strategy.

Mixtape Ch.3.  The book's point is that a DAG tells you which variables you must
NOT condition on, and that this is not something a regression table can reveal.
Here we make it numeric.

Two classic cases:

1.  **The movie-star DAG.**  Talent and Beauty are independent in the population.
    Both cause Stardom (you get cast if you have enough of either).  Conditioning
    on Stardom -- e.g. by studying only working actors -- manufactures a negative
    correlation between talent and beauty that exists nowhere in the population.
    Stardom is a *collider* on the path Talent -> Stardom <- Beauty.

2.  **Sample selection as conditioning.**  The Mixtape's discrimination example:
    if occupation is a collider between discrimination and unobserved ability,
    "controlling for occupation" to isolate wage discrimination introduces bias
    rather than removing it.

The moral is asymmetric with the confounding case: omitting a confounder biases
you, and *including* a collider also biases you.  There is no direction of "more
controls is safer".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _common import SEED, banner, report, require, section

N = 100_000


def main() -> None:
    banner("Collider bias -- conditioning on a common EFFECT creates correlation")
    rng = np.random.default_rng(SEED)
    import statsmodels.formula.api as smf

    # ---- 1. the movie-star DAG -------------------------------------------
    section("1. Talent -> Stardom <- Beauty, with talent independent of beauty")
    talent = rng.normal(0, 1, N)
    beauty = rng.normal(0, 1, N)
    star = ((talent + beauty) > 1.6).astype(int)  # cast only if the sum is high

    pop_corr = np.corrcoef(talent, beauty)[0, 1]
    star_corr = np.corrcoef(talent[star == 1], beauty[star == 1])[0, 1]
    nonstar_corr = np.corrcoef(talent[star == 0], beauty[star == 0])[0, 1]

    print(f"  corr(talent, beauty) in the POPULATION      {pop_corr:+.4f}   <- true value is 0")
    print(f"  corr(talent, beauty) among STARS            {star_corr:+.4f}   <- manufactured")
    print(f"  corr(talent, beauty) among NON-STARS        {nonstar_corr:+.4f}   <- also manufactured")
    print(f"  share who are stars                         {star.mean():.3f}")
    require(abs(pop_corr) < 0.02, "talent and beauty are independent by construction")
    require(star_corr < -0.1, "conditioning on the collider must induce negative correlation")
    print(
        "\n  Nobody 'controlled' for anything here.  Restricting attention to working\n"
        "  actors IS conditioning on the collider.  Any dataset built from a selected\n"
        "  population carries this bias before you write a single line of regression."
    )

    # ---- 2. the same thing via a regression control ---------------------
    section("2. Now do it with a regression control, which is how it reaches papers")
    df = pd.DataFrame({"talent": talent, "beauty": beauty, "star": star})
    good = smf.ols("beauty ~ talent", data=df).fit(cov_type="HC1")
    bad = smf.ols("beauty ~ talent + star", data=df).fit(cov_type="HC1")
    report("beauty ~ talent               (correct)", good.params["talent"], good.bse["talent"])
    report("beauty ~ talent + star  (CONTAMINATED)", bad.params["talent"], bad.bse["talent"])
    print(
        "  The second specification has a lower residual variance, a higher R-squared,\n"
        "  and a beautifully significant t-statistic on a coefficient that should be\n"
        "  zero.  No diagnostic in the regression output distinguishes the two."
    )

    # ---- 3. contrast with a genuine confounder --------------------------
    section("3. Contrast: a genuine CONFOUNDER, where controlling is the right move")
    u = rng.normal(0, 1, N)  # common cause
    x = 0.7 * u + rng.normal(0, 1, N)
    y = 1.5 * x + 2.0 * u + rng.normal(0, 1, N)  # true effect of x on y is 1.5
    df2 = pd.DataFrame({"x": x, "y": y, "u": u})
    omit = smf.ols("y ~ x", data=df2).fit(cov_type="HC1")
    ctrl = smf.ols("y ~ x + u", data=df2).fit(cov_type="HC1")
    report("y ~ x       (confounder omitted)", omit.params["x"], omit.bse["x"],
           extra=f"[bias {omit.params['x'] - 1.5:+.4f}]")
    report("y ~ x + u   (confounder included)", ctrl.params["x"], ctrl.bse["x"],
           extra=f"[bias {ctrl.params['x'] - 1.5:+.4f}]")

    # ---- 4. the mediator case -------------------------------------------
    section("4. Third case: a MEDIATOR, where controlling answers a different question")
    d = rng.binomial(1, 0.5, N)
    m = 1.0 * d + rng.normal(0, 1, N)  # treatment works through m
    y3 = 0.8 * d + 1.2 * m + rng.normal(0, 1, N)  # plus a direct channel
    df3 = pd.DataFrame({"d": d, "m": m, "y": y3})
    total = smf.ols("y ~ d", data=df3).fit(cov_type="HC1")
    direct = smf.ols("y ~ d + m", data=df3).fit(cov_type="HC1")
    report("y ~ d       -> TOTAL effect", total.params["d"], total.bse["d"],
           extra="[truth 0.8 + 1.0*1.2 = 2.0]")
    report("y ~ d + m   -> DIRECT effect", direct.params["d"], direct.bse["d"],
           extra="[truth 0.8]")
    print(
        "  Neither is wrong -- they answer different questions.  The error is reporting\n"
        "  the second while describing the first, which happens whenever a 'control'\n"
        "  is actually a post-treatment variable."
    )

    # ---- 5. the rule -----------------------------------------------------
    section("The rule you cannot get from a regression table")
    print(
        "  CONFOUNDER (common cause of D and Y)   -> you MUST condition\n"
        "  COLLIDER   (common effect of D and Y)  -> you must NOT condition\n"
        "  MEDIATOR   (on the causal path D->Y)   -> condition only for the direct effect\n"
        "  and never for the total effect\n\n"
        "  Which is which is a claim about the world, encoded in a DAG, defended in\n"
        "  prose.  Write the DAG down before choosing controls.  In Python:\n"
        "    dowhy, causal-learn, networkx, or StatsPAI's sp.dag / sp.check_identification\n"
        "  all compute adjustment sets from a stated graph -- the old 'DAGs: R only'\n"
        "  entry in the comparison tables is out of date."
    )


if __name__ == "__main__":
    main()
