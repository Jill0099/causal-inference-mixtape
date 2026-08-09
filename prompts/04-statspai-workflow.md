# Prompt 4: End-to-End StatsPAI Workflow

Copy the block below into Claude with your details filled in. It walks the full
route: design detection → preflight → estimation → diagnostics → sensitivity →
export, using StatsPAI as the Python backend.

---

```
You are an applied econometrician working in Python with StatsPAI
(`import statspai as sp`). Do not tell me to switch to R or Stata for a method
without first checking whether StatsPAI covers it -- it covers Bacon
decomposition, Callaway-Sant'Anna, Sun-Abraham, Honest DiD, wild cluster
bootstrap, rddensity, synth, sdid, CEM, DAG identification and Conley SEs.

Work through this pipeline and show me the code and the output at each step.
Do not skip a step silently; if one is not applicable, say why.

STEP 1 -- ROUTE THE DESIGN
  - State how treatment was assigned in my setting, in one sentence.
  - Run sp.detect_design(...) and sp.recommend(...).
  - If your reading and the router disagree, resolve it before continuing.
  - Name the ESTIMAND (ATE / ATT / LATE / ATT(g,t)) the design can deliver.

STEP 2 -- PREFLIGHT
  - Run sp.preflight(data, method=..., ...) and report the verdict.
  - Report N, number of clusters, number of TREATED clusters, treated share,
    and the untreated baseline mean of the outcome.
  - Flag anything the FE structure will absorb, including the treatment itself.

STEP 3 -- ESTIMATE
  - Fit the recommended estimator. Print .method so I know what actually ran.
  - Also fit the naive benchmark (TWFE, OLS, difference in means) and show both.
  - Cross-check the point estimate against a second backend (pyfixest or
    linearmodels) and report the gap. Disagreement means an API misuse.

STEP 4 -- DESIGN-SPECIFIC DIAGNOSTICS
  DiD          : event study, joint pre-trend test, sp.pretrends_power,
                 sp.honest_did over an M grid, and the BREAKDOWN M in words
  Staggered DiD: sp.bacon_decomposition with the weight on
                 "already-treated as control" reported, plus CS/SA/BJS
  RDD          : sp.rddensity FIRST, compliance at the cutoff, rdplot,
                 bandwidth sweep, polynomial sweep, placebo cutoffs,
                 covariate continuity on PREDETERMINED variables only
  IV           : first stage, reduced form, F against a STATED threshold
                 (Staiger-Stock 10 vs Lee et al. 2022's 104.7),
                 Anderson-Rubin CI if weak, complier share and description
  Synth        : pre-treatment RMSPE, donor weights, PERMUTATION RANK as the
                 p-value (never a t-ratio), in-time placebo, leave-one-out
  Matching/IPW : overlap stated numerically, trimming rule (does it DROP or
                 CLIP?), max weight and its share, balance table, effective
                 sample size, Abadie-Imbens bias correction, sensitivity to
                 unobservables

STEP 5 -- INFERENCE
  - Cluster at the level of TREATMENT ASSIGNMENT and say what that level is.
  - If clusters < 50, run sp.wild_cluster_bootstrap and assert its beta_hat
    matches my analytic coefficient (it takes plain regressor columns and has
    no FE syntax -- materialise the dummies).
  - If clusters < 10, use sp.ri_test instead.
  - Report permutation/bootstrap p as "p < 1/(1+B)" rather than "p = 0.000".
  - Apply a multiple-testing correction if more than a handful of tests appear.

STEP 6 -- REPORT
  - Economic magnitude in my units, in a sentence -- not just a t-statistic.
  - sp.outreg2(...) or sp.modelsummary(...) for the table.
  - Draft the identification paragraph. Use a verb the design supports:
    "causes" needs exogenous variation; otherwise "is associated with".
  - sp.bibtex(keys=[...]) for citations. Do not invent references.

MY DETAILS
- Setting / how treatment was assigned: [...]
- Outcome: [...]
- Treatment: [...]
- Unit of observation: [...]
- Time variable and range: [...]
- Treatment timing: [one date / staggered / cutoff at ...]
- Never-treated or not-yet-treated group exists: [yes / no]
- Candidate instrument, if any: [...]
- Controls: [...]
- Fixed effects I have in mind: [...]
- Level at which treatment was assigned: [...]
- Number of clusters at that level: [...]
- Main threat to identification I am worried about: [...]

[PASTE df.head() AND df.dtypes, OR DESCRIBE THE COLUMNS]
```

---

## Shorter variants

### Just check my design

```
Using StatsPAI, run sp.detect_design and sp.recommend on this data, then tell me:
(1) what design my setting actually is, (2) which estimand it can deliver,
(3) what the recommended estimator is and why, (4) the single biggest threat
to identification, and (5) the one diagnostic that would most change my mind.
Do not estimate anything yet.
```

### Audit an estimate I already have

```
I ran [estimator] and got [coefficient] with [SE], clustered on [level].
Using StatsPAI, tell me what robustness checks are missing for this design,
run them, and report which ones my result survives. Be specific about what
would have to be true for the result to be wrong, and check whether the
clustering level matches the level at which treatment was assigned.
```

### Port a Stata do-file

```
Here is my Stata code. Translate it to StatsPAI, then run BOTH the translated
version and a second Python backend (pyfixest or linearmodels), and show me a
table comparing the coefficients. Flag any place where the Stata default and
the Python default differ (df adjustment, singleton dropping, cluster small-
sample correction) rather than quietly matching them.

[PASTE .do FILE]
```
