# Design Router — from assignment mechanism to estimator

Users do not arrive saying "I need Callaway-Sant'Anna." They say *"a policy
rolled out across provinces in different years and I want to know if it worked."*
This file routes from **how treatment was assigned** to a design, an estimator,
and the diagnostics that make it publishable.

Always ask about the assignment mechanism first. The data structure follows from
it, never the other way round.

---

## §1. The first four questions

1. **How was treatment assigned?** Randomised, by a threshold rule, by a policy
   date, by self-selection, by something plausibly-as-good-as-random?
2. **Who is untreated, and why?** Never-treated units, not-yet-treated units, or
   units just below a cutoff?
3. **Does timing vary across units?** One date for everyone, or staggered?
4. **What estimand does the question call for?** ATE, ATT, or a local effect for
   a specific subpopulation? (See [`mixtape-core.md`](mixtape-core.md) §1 — these
   differ, and the design determines which one you can have.)

---

## §2. The routing tree

```
How was treatment assigned?
│
├─ RANDOMISED (you or someone else ran the experiment)
│   └─ Difference in means + covariate adjustment
│      Inference: randomisation inference using the actual assignment mechanism;
│      cluster at the level randomised.  Check attrition and balance.
│      -> scripts/09_randomization_inference.py
│
├─ BY A THRESHOLD on a continuous running variable
│   ├─ compliance ~100%      -> SHARP RDD
│   ├─ compliance imperfect  -> FUZZY RDD  (crossing instruments for treatment)
│   ├─ the slope, not the level, jumps -> REGRESSION KINK
│   ├─ the cutoff pre-dated the policy  -> DIFF-IN-DISCONTINUITIES
│   └─ several cutoffs / running variables -> MULTI-CUTOFF / MULTI-SCORE RD
│      Required first: manipulation density test.  If it fails, the design is
│      dead and no bandwidth choice repairs it.
│      -> scripts/05_rdd.py
│
├─ BY A POLICY DATE, panel data available
│   ├─ one date for all treated units
│   │   └─ 2x2 DiD, then EVENT STUDY for dynamics
│   │      -> scripts/02_did_2x2.py, scripts/03_event_study.py
│   ├─ dates VARY across units (staggered)
│   │   └─ do NOT report TWFE as the ATT
│   │      1. sp.bacon_decomposition to size the problem
│   │      2. Callaway-Sant'Anna / Sun-Abraham / BJS as the headline
│   │      -> scripts/04_staggered_did.py
│   ├─ treatment is CONTINUOUS in dose
│   │   └─ continuous DiD (Callaway-Goodman-Bacon-Sant'Anna); parallel trends
│   │      must hold at every dose
│   └─ ONE treated unit, many untreated
│       └─ SYNTHETIC CONTROL (+ synthetic DiD as robustness)
│          Inference is the permutation rank, never a t-ratio
│          -> scripts/07_synthetic_control.py
│
├─ SELF-SELECTED, but you have an instrument
│   └─ IV / 2SLS -- identifies a LATE for compliers, not the ATE
│      Report: first stage, reduced form, F against a stated threshold,
│      exclusion argument in prose, complier characterisation
│      -> scripts/06_iv_2sls.py
│      Common templates: judge/examiner leniency, weather shocks,
│      shift-share (Bartik), distance/proximity, lottery
│
├─ SELF-SELECTED, no instrument, but rich observables
│   └─ SELECTION ON OBSERVABLES: matching / IPW / doubly robust / DML
│      Check OVERLAP FIRST -- it is a precondition, not a robustness check
│      Then: balance table, Oster/sensemakr sensitivity to unobservables
│      -> scripts/08_matching_ipw.py
│      Be explicit that this is the weakest of the designs here.
│
└─ NONE OF THE ABOVE
    └─ Say so.  Use associational language (see identification-writing-patterns
       .md §3), report bounds (Manski, Lee, Horowitz-Manski) or a sensitivity
       analysis, and do not write "causes".
```

---

## §3. Ask StatsPAI for a second opinion

StatsPAI has a built-in router. Use it to check your reasoning, not to replace it
— it reads the data structure, which cannot tell it how treatment was assigned.

```python
import statspai as sp

sp.detect_design(df, unit="unit_id", time="year")
# -> {'design': 'panel', 'confidence': 1.0, 'candidates': [...]}

sp.recommend(df, y="outcome", treatment="treat", id="unit_id", time="year")
# -> Callaway-Sant'Anna (2021) — staggered DID
#    reason: "Multiple time periods with staggered treatment adoption.
#             Robust to heterogeneous treatment effects (unlike TWFE)."
#    warnings: [...]

sp.preflight(df, method="did", y="outcome", treat="treat", time="year", id="unit_id")
# -> {'verdict': 'PASS', 'summary': {'passed': 8, 'warning': 0, 'failed': 0}}
```

If your reasoning and the router disagree, one of you has misread the data.
Find out which before estimating.

---

## §4. Estimator selection within a design

### Staggered DiD

| Situation | Estimator |
|---|---|
| Never-treated group exists | Callaway-Sant'Anna, `control_group="nevertreated"` |
| No never-treated units | CS with `"notyettreated"`, or BJS imputation |
| Want an event-study path | Sun-Abraham, or `sp.aggte(cs, type="dynamic")` |
| Treatment can turn off and on | de Chaisemartin-D'Haultfœuille |
| Very many periods / units | BJS imputation (`sp.did_imputation`) — fastest |
| Covariates needed for parallel trends | CS with `x=[...]` (doubly robust) |

### RDD

| Situation | Estimator |
|---|---|
| Default | `sp.rdrobust`, MSE-optimal bandwidth, robust bias-corrected CI |
| Discrete running variable | `sp.rd_discrete`, or honest CIs |
| Want CIs robust to the bandwidth choice | `sp.rd_honest` (Armstrong-Kolesár) |
| Two running variables | `sp.rd2d` / `sp.multi_score_rd` |
| Many cutoffs | `sp.rdmc` / `sp.multi_cutoff_rd` |

### IV

| Situation | Estimator |
|---|---|
| One or few strong instruments | 2SLS |
| Weak instrument | Anderson-Rubin CI; tF-adjusted critical values |
| Many instruments (e.g. judge dummies) | JIVE, or leave-one-out leniency |
| Heteroskedastic, over-identified | LIML or GMM rather than 2SLS |

### Selection on observables

| Situation | Estimator |
|---|---|
| Good overlap, few covariates | Matching with Abadie-Imbens bias correction |
| Good overlap, many covariates | Entropy balancing, CBPS, or overlap weights |
| Many covariates, flexible nuisance | Double/debiased ML (`sp.dml`) |
| Want protection against one misspecification | Doubly robust / AIPW |
| Poor overlap | **Stop.** Redefine the estimand to the overlap population, or say the design does not support the claim. |

---

## §5. What you owe the reader, by design

| Design | Non-negotiable |
|---|---|
| RCT | Balance table, attrition, randomisation level = cluster level |
| DiD (2×2) | Event study, parallel-trends power, Honest DiD breakdown M |
| Staggered DiD | Bacon weights + a heterogeneity-robust headline estimator |
| RDD | Density test **before** estimating, bandwidth and polynomial sweeps, placebo cutoffs, covariate continuity |
| IV | First stage, reduced form, F against a stated threshold, exclusion argument, complier description |
| Synthetic control | Pre-treatment RMSPE, donor weights, permutation rank, in-time placebo |
| Matching / IPW | Overlap plot, balance table, trimming rule stated, sensitivity to unobservables |

See [`reporting-checklist.md`](reporting-checklist.md) for the full list, and
[`inference-and-standard-errors.md`](inference-and-standard-errors.md) for what
to do about clustering in each case.
