# Prompt 1: Implement a Causal Inference Method

Copy and paste the prompt below into Claude with your details filled in.

---

```
You are an expert econometrician implementing causal inference methods.

Implement a complete [METHOD] analysis pipeline in [LANGUAGE: Python / R / Stata].
In Python, use StatsPAI (`import statspai as sp`) as the primary backend and
cross-check the point estimate against pyfixest or linearmodels.

Requirements:
1. State the ESTIMAND this design identifies (ATE / ATT / LATE / ATT(g,t)).
2. Data preparation (variable creation, sample restrictions)
3. Main estimation with correct standard errors, clustered at the level at
   which TREATMENT WAS ASSIGNED
4. Design-specific diagnostics (below)
5. Publication-ready output, reporting N, number of clusters, number of TREATED
   clusters, treated share, untreated baseline mean, and the economic magnitude
   in words -- not a t-statistic alone

Method-specific requirements:

- DiD: Event-study plot with the reference period omitted from the design matrix
  and re-inserted in the plot as an exact zero; endpoints BINNED, not zeroed.
  Joint pre-trend test AND its power. Honest DiD (Rambachan-Roth) over an M grid,
  with the breakdown M stated in words. Placebo treatment date.
- Staggered DiD: do NOT report TWFE as the ATT. Bacon decomposition first, with
  the weight on "already-treated as control" comparisons; then Callaway-Sant'Anna
  / Sun-Abraham / BJS as the headline, TWFE as a benchmark.
- RDD: manipulation density test BEFORE estimating. Compliance at the cutoff.
  MSE-optimal bandwidth with the ROBUST BIAS-CORRECTED CI as the headline
  (a hand-rolled `y ~ D*x` OLS inside a chosen bandwidth is not an RDD estimate).
  Bandwidth sweep, polynomial sweep stopping at p=2, placebo cutoffs, covariate
  continuity on PREDETERMINED variables only.
- IV: first stage and reduced form. First-stage F against a STATED criterion
  (Staiger-Stock 10 vs Lee et al. 2022's 104.7). Anderson-Rubin CI if weak.
  Exclusion restriction argued in prose. Complier share and characterisation;
  state the estimand as a LATE. Use a real 2SLS routine -- never a manual
  two-step, and never a nonlinear first stage plugged in as a regressor.
- Synthetic Control: pre-treatment RMSPE, donor weights, PERMUTATION RANK as the
  p-value (never a t-ratio -- with one treated unit there is no sampling
  distribution), in-time placebo, leave-one-out.
- Matching/IPW: overlap stated numerically BEFORE estimating. State whether your
  trimming DROPS off-support units or CLIPS the propensity score. Max weight and
  its share of the total. Balance table with standardised differences. Abadie-
  Imbens bias correction for continuous covariates, and matching-appropriate
  standard errors -- not the post-match OLS default.
- Few clusters (<50): wild cluster bootstrap; assert its beta matches the
  analytic coefficient. Fewer than 10: randomisation inference.

My details:
- Method: [e.g., Difference-in-Differences]
- Language: [e.g., Python]
- Outcome variable: [e.g., firm_investment]
- Treatment variable: [e.g., reform_exposure]
- Treatment timing: [e.g., 2014 for all treated units / staggered]
- Key controls: [e.g., firm size, leverage, ROA]
- Fixed effects: [e.g., firm + year]
- Clustering level: [e.g., firm]
- Data format: [e.g., panel, entity_id + year columns]
- Sample size: [e.g., ~50,000 firm-years]
- Key concern: [e.g., contemporaneous policies]

[PASTE SAMPLE OF YOUR DATA STRUCTURE OR DESCRIBE COLUMNS]
```

---

## Method-Specific Variants

### For Staggered DiD

```
Additional requirement: Treatment timing varies across units.
- Use [Callaway & Sant'Anna / Sun & Abraham / Bacon decomposition] to address TWFE bias.
- Show that standard TWFE is potentially biased.
- Report group-time ATTs and aggregated dynamic effects.

My staggered details:
- Treatment cohorts: [e.g., 2010, 2012, 2014, 2016]
- Never-treated group exists: [yes/no]
- Preferred estimator: [e.g., Callaway & Sant'Anna]
```

### For Fuzzy RDD

```
Additional requirement: Treatment assignment is not sharp at the cutoff.
- Implement fuzzy RDD as IV where crossing the cutoff instruments for treatment.
- Report both reduced-form and 2SLS estimates.
- Show first-stage discontinuity in treatment probability.

My fuzzy RDD details:
- Running variable: [e.g., vote share]
- Cutoff: [e.g., 50%]
- Treatment: [e.g., policy implementation — not all units above cutoff comply]
- Compliance rate above cutoff: [e.g., ~75%]
```
