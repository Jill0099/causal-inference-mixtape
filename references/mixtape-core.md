# Mixtape Core — the conceptual layer under the code

The chapters of *Causal Inference: The Mixtape* that a code-first cheatsheet
usually skips, and that determine whether any of the code means anything.

Runnable companions:
[`scripts/01_potential_outcomes.py`](../scripts/01_potential_outcomes.py),
[`scripts/simulations/sim_collider_bias.py`](../scripts/simulations/sim_collider_bias.py),
[`scripts/simulations/sim_twfe_staggered_bias.py`](../scripts/simulations/sim_twfe_staggered_bias.py).

---

## §1. Potential outcomes (Ch. 4)

With `Y(1)`, `Y(0)` the potential outcomes and `D` the treatment:

| Estimand | Definition | Who it is about |
|---|---|---|
| ATE | `E[Y(1) − Y(0)]` | everyone |
| ATT | `E[Y(1) − Y(0) \| D = 1]` | the treated |
| ATU | `E[Y(1) − Y(0) \| D = 0]` | the untreated |
| LATE | `E[Y(1) − Y(0) \| complier]` | those moved by the instrument |
| ATT(g,t) | group-time ATT | cohort `g` at period `t` |

**"The treatment effect" is not a well-posed request.** These coincide only when
effects are homogeneous or assignment is independent of them.

### The decomposition

You cannot observe both potential outcomes for one unit. What data gives you is
the simple difference in observed outcomes:

```
SDO = E[Y|D=1] − E[Y|D=0]
    = ATE
    + ( E[Y(0)|D=1] − E[Y(0)|D=0] )     <- selection bias
    + (1 − π) · ( ATT − ATU )           <- heterogeneous treatment effect bias
```

with `π = P(D = 1)`. This identity holds exactly — `01_potential_outcomes.py`
verifies it to a residual of `2.7e-15` on a simulated population where all three
terms are non-zero (SDO 4.989 = ATE 2.999 + selection 1.413 + heterogeneity 0.578).

**Every design in this skill is an argument that one or both bias terms is zero.**
Not a computation that makes them zero — an argument, defended in prose:

| Design | What it assumes away |
|---|---|
| Randomisation | Both terms, by construction |
| Selection on observables | Selection bias, *conditional on X* |
| DiD | Selection bias in *changes* (parallel trends), not in levels |
| RDD | Both, *locally* at the cutoff (continuity) |
| IV | Both, *for compliers only* (exclusion + monotonicity) |
| Synthetic control | Selection bias, via a weighted donor counterfactual |

### The two silent assumptions

**SUTVA.** (a) No interference — one unit's treatment does not affect another's
outcome. Spillovers, general-equilibrium effects, and network exposure all break
it. (b) No hidden variation in treatment — one version of `D`. A "job training
programme" that differs across sites violates this.

**Independence.** `(Y(0), Y(1)) ⟂ D`. Randomisation delivers it; nothing else
does without an argument. Note this is about the *potential outcomes*, not the
observed ones, which is why it cannot be tested directly.

---

## §2. DAGs: which controls, and which never (Ch. 3)

| Role | Relation to D and Y | What to do |
|---|---|---|
| **Confounder** | common **cause** of D and Y | **must** condition |
| **Collider** | common **effect** of D and Y | must **not** condition |
| **Mediator** | on the causal path D → Y | condition only for the *direct* effect |

The asymmetry is the point: omitting a confounder biases you, and *including* a
collider also biases you. **There is no direction in which "more controls" is
safer.**

`sim_collider_bias.py` makes this numeric. Talent and beauty are independent by
construction; both cause stardom. Among stars their correlation is **−0.246**.
Running `beauty ~ talent + star` returns a coefficient of −0.246 with t = −78.6 —
higher R², lower residual variance, and completely spurious. **No diagnostic in
the regression output distinguishes it from the correct specification.**

Sample selection is conditioning. If your data covers only survivors, only
listed firms, only working actors, the collider bias is already in the dataset
before you write a line of code.

```python
import statspai as sp
sp.dag(edges=[('U','D'), ('U','Y'), ('D','Y')])
sp.check_identification(...)     # adjustment sets from a stated graph
```

Python has DAG tooling now — `sp.dag`, `sp.check_identification`, `dowhy`,
`causal-learn`. The "DAGs: R only" line in old comparison tables is out of date.

---

## §3. Matching and selection on observables (Ch. 5)

### Overlap is a precondition, not a robustness check

If treated and control propensity distributions barely overlap, no weighting
scheme rescues the design — it relocates the extrapolation somewhere less
visible. Report the overlap **before** the estimate.

On NSW/CPS, only 30% of CPS controls fall inside the treated units' propensity
support. That is the finding; anything estimated on the other 70% is
extrapolation.

### The estimator scoreboard

`08_matching_ipw.py` scores every method against the NSW experimental benchmark
of **+$1,794**:

| Estimator | Estimate | Bias |
|---|---|---|
| Experimental (truth) | +1,794 | — |
| Naive difference in means (CPS controls) | −8,498 | −10,292 |
| OLS with covariates | +699 | −1,095 |
| IPW, untrimmed | −7,805 | −9,599 |
| IPW, dropping off-support units | +1,454 | −340 |
| NN matching, bias correction off | +2,134 | +339 |
| NN matching, bias correction on | +2,202 | +408 |

### Three things the standard template gets wrong

1. **Untrimmed IPW explodes.** A control with pscore 0.999 gets weight 1000. On
   NSW/CPS a single observation carries 7% of the total weight. Trim or use
   normalised (Hájek) weights, and **say which** — and verify whether your
   library *drops* off-support units or merely *clips* the score. On this data
   that choice is worth $9,400.

2. **Matching standard errors are not OLS standard errors.** Abadie & Imbens
   (2006, 2011): the post-match regression SE ignores that matching is not a
   sampling scheme OLS knows about. Worse, Abadie & Imbens (2008) show the
   ordinary nonparametric bootstrap is *also* invalid for nearest-neighbour
   matching. Use the Abadie-Imbens variance.

3. **Bias correction is not optional with continuous covariates.** With more
   than one continuous matching variable, exact matches do not exist, the match
   discrepancy shrinks too slowly, and the bias term **does not vanish
   asymptotically**. The correction regression-adjusts each pair for the residual
   covariate gap.

### Prefer weighting or covariate matching over PSM

King & Nielsen (2019): propensity-score matching approximates a *completely
randomised* experiment rather than a *blocked* one, so pruning on the score can
**increase** imbalance as you prune. Match on covariates directly with bias
correction, or use entropy balancing / CBPS / overlap weights.

Balance is the only evidence that any of this worked. Target
`|standardised difference| < 0.1` after weighting. A t-test p-value on balance
is not a substitute — it conflates imbalance with sample size.

---

## §4. Instrumental variables: what you actually identify (Ch. 7)

### LATE, not ATE

Under heterogeneous effects, 2SLS identifies the effect **for compliers** — units
whose treatment status the instrument actually moves. Imbens & Angrist (1994):

| Type | `D(z=0)` | `D(z=1)` | Contributes to LATE? |
|---|---|---|---|
| Complier | 0 | 1 | **yes** |
| Always-taker | 1 | 1 | no |
| Never-taker | 0 | 0 | no |
| Defier | 1 | 0 | must not exist |

**Monotonicity** is the assumption that there are no defiers. It is not testable.
It is also not innocuous: in a judge-leniency design it says no defendant faces a
*harsher* outcome from being assigned a *more lenient* judge.

Two things follow that papers routinely elide:

- The complier population is **defined by the instrument**. A different
  instrument for the same treatment identifies a different LATE, and the two can
  legitimately disagree. This is why an over-identification test failing is not
  automatically evidence of invalidity.
- **Characterise your compliers.** Their share is the first stage; their
  covariate profile is recoverable (Abadie's kappa weights). "The return to
  schooling is 13%" means "for people whose schooling responded to living near a
  college."

On Card (1995), IV (0.132) exceeds OLS (0.074). If plain ability bias were the
story you would expect the opposite — which is itself evidence that compliers
differ from the average worker.

### Weak instruments: two thresholds

`F > 10` (Staiger-Stock) bounds *relative bias*. Lee, McCrary, Moreira & Porter
(2022) show valid 5% t-test inference needs `F > 104.7`, or a tF-adjusted
critical value. State which one you are invoking. Card's `nearc4` gives F ≈ 17.5:
usable point estimate, over-rejecting t-test.

When the instrument is weak, report an **Anderson-Rubin** confidence interval —
it is valid regardless of instrument strength (`sp.anderson_rubin_ci`).

### Over-identification tests are not validity tests

Sargan/Hansen only asks whether your instruments **agree with each other**. Two
instruments that violate exclusion in the same direction pass happily. The
exclusion restriction is defended in prose. Always.

### The judge / examiner leniency design

The most common modern IV template: cases are quasi-randomly assigned to
judges/examiners/caseworkers who differ in strictness, and leniency instruments
for the decision.

- Instrument: leave-one-out mean decision rate of the assigned judge, computed
  *excluding the focal case* (otherwise it is mechanically correlated with the
  outcome).
- Requires: quasi-random assignment within court × time cells (test it —
  balance of case characteristics on judge leniency), exclusion (the judge
  affects the outcome only through this decision — violated if judges also set
  bail amount, sentence length, or programme referral), and monotonicity.
- Many-instrument bias if you use judge dummies directly: use JIVE
  (`sp.jive`) or the leave-one-out leniency measure.

### Two bugs that survive code review

Both are demonstrated live in
[`scripts/06_iv_2sls.py`](../scripts/06_iv_2sls.py):

1. **Manual two-step 2SLS gives the wrong standard error.** It reproduces the
   point estimate exactly — which is what makes it convincing — then computes
   residuals against `D̂` instead of `D`. Never hand-roll 2SLS for inference.
2. **The forbidden regression.** A logit/probit first stage whose fitted value
   is plugged in as a *regressor* is inconsistent unless the nonlinear model is
   exactly right. On Card with `D = 1{educ ≥ 13}` it returns −0.070 where linear
   2SLS returns +0.785. If you want to use a nonlinear first stage, use its
   fitted value as an **instrument**, not a regressor (Wooldridge 2010, 21.1).

---

## §5. Panel data and fixed effects (Ch. 8)

Unit fixed effects absorb **time-invariant** unobserved heterogeneity. That is
all they do.

What they do **not** solve:

- **Time-varying confounders.** A firm FE does nothing about a shock that hits
  treated firms in the treatment year.
- **Reverse causality.** FE does not orient an arrow.
- **Measurement error.** Within transformations *amplify* attenuation bias by
  removing the signal variance while keeping the noise.
- **Bad controls.** A post-treatment variable is still a bad control inside a FE
  model.

Two further traps:

**FE can absorb the treatment itself.** `county × year` fixed effects absorb any
state-level policy that varies only at state × year. If your treatment
coefficient is suspiciously precise or the variable is dropped, check what the FE
structure ate.

**Nickell bias.** With a lagged dependent variable and unit FE, the within
transformation correlates the lagged `y` with the demeaned error. The bias is
`O(1/T)` — serious on short panels. Use Arellano-Bond / system GMM
(`sp.xtabond`, `sp.xtdpdsys`) or accept the bounds: FE and OLS bracket the true
coefficient.

**Lagged dependent variable vs unit FE is a choice, not a menu.** Angrist &
Pischke: they bound the truth from opposite sides. Report both.

---

## §6. Staggered DiD: why TWFE breaks (Ch. 9)

Under staggered adoption with dynamic effects, TWFE is a variance-weighted
average of *all* possible 2×2 comparisons — including already-treated units
serving as controls for later-treated units. Because those units' own effects are
still evolving, their trend is contaminated and the comparison enters with
negative weight (Goodman-Bacon 2021).

`sim_twfe_staggered_bias.py` scores this against a known truth:

| Effect gradient | True ATT | TWFE | TWFE bias | Callaway-Sant'Anna |
|---|---|---|---|---|
| 0.00 | 0.000 | −0.005 | −0.005 | +0.046 |
| 0.25 | 1.708 | 0.789 | −0.919 | 1.754 |
| 1.00 | 6.832 | 3.173 | **−3.659 (−54%)** | 6.878 |

With homogeneous effects TWFE is fine. With effects that grow, it understates by
more than half — and with a steep enough gradient it can cross zero and report a
negative effect when every unit gained.

**The workflow:**

1. `sp.bacon_decomposition` — how much weight sits on "already-treated as
   control"? Small weight, small problem. On `castle.dta` it is 9.2%, and the
   weighted sum of 2×2s reproduces the TWFE coefficient exactly.
2. A heterogeneity-robust estimator as the **headline**: Callaway-Sant'Anna,
   Sun-Abraham, Borusyak-Jaravel-Spiess, or de Chaisemartin-D'Haultfœuille.
3. TWFE stays in the table as a benchmark, with the gap explained.
4. If they agree, **say so** — that is a finding, not a non-result.

---

## §7. Mixtape datasets

All load from `https://raw.github.com/scunning1975/mixtape/master/<name>`.
`scripts/_common.py` provides `read_mixtape()` with a disk cache.

| File | Chapter | Use |
|---|---|---|
| `yule.dta` | 2 | Yule (1899) pauperism regression |
| `training_example.dta` | 4/5 | Potential outcomes, subclassification by hand |
| `thornton_hiv.dta` | 4 | Thornton (2008) HIV-testing RCT; randomisation inference |
| `nsw_mixtape.dta` | 5 | LaLonde NSW experimental sample (benchmark +$1,794) |
| `cps_mixtape.dta` | 5 | CPS non-experimental control pool |
| `lmb-data.dta` | 6 | Lee-Moretti-Butler sharp RDD, US House |
| `card.dta` | 7 | Card (1995) college-proximity IV |
| `judge_fe.dta` | 7 | Judge leniency / examiner design |
| `castle.dta` | 9 | Castle-doctrine staggered DiD |
| `abortion.dta` | 9 | Donohue-Levitt DDD |
| `texas.dta` | 10 | Texas prison-capacity synthetic control |

`close_college.dta` **404s** on the public mirror — use `card.dta` instead.

Stata files load as float32 or as string columns with blanks. Cross-backend
equality checks need a `1e-5` tolerance, not machine epsilon.
