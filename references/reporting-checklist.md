# Reporting Checklist — what to print alongside every estimate

A coefficient on its own is not a result. This is the minimum any causal
estimate should carry when handed back to a user, a coauthor, or a referee.

`scripts/_common.py` provides `design_facts()`, which prints §1 in one call.

---

## §1. Always, regardless of design

- [ ] **N observations** actually used after listwise deletion and sample
      restrictions — not the size of the input file
- [ ] **Number of clusters**, and **number of TREATED clusters**
- [ ] **Treated share**
- [ ] **Untreated baseline mean of the outcome** — this is what turns a
      coefficient into an economic magnitude
- [ ] **Economic magnitude in the reader's units**, stated in words
      ("a 0.068 coefficient on log homicides is roughly 7.1% relative to the
      never-treated baseline")
- [ ] **Which estimator actually ran.** Convenience wrappers pick a method for
      you; read `.method` before quoting the number
- [ ] **Observations absorbed or dropped by the fixed-effect structure**
      (singletons, no-variation groups). If FE ate 30% of your sample, say so
- [ ] **Package versions**, if the result is going into a paper

A t-statistic is not on this list. It is the least informative number in the
output and the one most often reported alone.

---

## §2. Sanity checks to run before believing your own output

- [ ] Does the fixed-effect structure **absorb the treatment itself**? (A
      `county × year` FE eats a state-level policy.) Suspiciously precise
      coefficients and dropped variables are the symptom
- [ ] Is any "control" a **post-treatment variable**? That is a mediator, and
      conditioning on it answers a different question
- [ ] Does the sample come from a **selected population** (survivors, listed
      firms, completers)? That is conditioning on a collider before you began
- [ ] Do **two independent implementations agree** on the point estimate?
      Cross-backend disagreement means an API misuse, not a modelling choice
- [ ] Does the **hand-computed version** match, where one exists (a 2×2 table,
      a reduced-form/first-stage ratio)?

---

## §3. By design

### RCT / randomised

- [ ] Balance table on pre-treatment covariates
- [ ] Attrition rate, and whether it differs by arm
- [ ] Clustering matches the level of randomisation
- [ ] Randomisation inference if clusters are few

### DiD (single treatment date)

- [ ] Event-study plot, with the reference period marked as an exact zero
- [ ] Joint pre-trend test **and its power** (Roth 2022)
- [ ] Honest DiD robust CIs and the **breakdown M** (Rambachan-Roth 2023)
- [ ] Placebo treatment date
- [ ] Alternative control group
- [ ] Clustering at the treatment-assignment level

### Staggered DiD

Everything above, plus:

- [ ] Goodman-Bacon decomposition, with the weight on "already-treated as
      control" comparisons reported
- [ ] A heterogeneity-robust estimator as the **headline**, TWFE as a benchmark
- [ ] Cohort composition table (how many units in each adoption cohort)
- [ ] If robust and TWFE agree, say so explicitly

### RDD

- [ ] Manipulation / density test **before** any estimate
- [ ] Compliance at the cutoff (is the design sharp?)
- [ ] RD plot
- [ ] MSE-optimal bandwidth with the **robust bias-corrected** CI as the headline
- [ ] Bandwidth sweep (h/2, h, 2h)
- [ ] Polynomial order sweep (stop at 2 — Gelman & Imbens 2019)
- [ ] Placebo cutoffs away from the true one
- [ ] Covariate continuity for **predetermined** variables only
- [ ] Effective sample size on each side of the cutoff

### IV / 2SLS

- [ ] First-stage table
- [ ] Reduced form
- [ ] First-stage F, **against a stated criterion** (Staiger-Stock 10 vs
      Lee et al. 2022's 104.7)
- [ ] Anderson-Rubin CI when the instrument is weak
- [ ] Exclusion restriction argued in prose — not asserted
- [ ] Monotonicity discussed
- [ ] Complier share and characterisation; LATE stated as a LATE
- [ ] Over-identification test if multiple instruments, with the caveat that
      passing is not validity
- [ ] Estimated with a real 2SLS routine, never a manual two-step

### Synthetic control

- [ ] Path plot and gap plot
- [ ] Donor weight table
- [ ] Pre-treatment RMSPE, scaled to something the reader can interpret
- [ ] **Permutation rank as the p-value** — never a t-ratio
- [ ] Leave-one-out donor sensitivity
- [ ] In-time placebo
- [ ] Number of donors, so the reader knows the p-value's resolution

### Matching / IPW / selection on observables

- [ ] Propensity score distribution by arm, and **overlap** stated numerically
- [ ] Trimming rule, and whether it **drops** or **clips**
- [ ] Maximum weight, and the share of total weight it carries
- [ ] Balance table (standardised differences before and after)
- [ ] Effective sample size after weighting
- [ ] Bias correction for continuous matching variables (Abadie-Imbens)
- [ ] Matching-appropriate standard errors, not the post-match OLS default
- [ ] Sensitivity to unobservables (Oster δ, sensemakr, Rosenbaum bounds)

---

## §4. Language check before publishing

Match the verb to the evidence. From
[`identification-writing-patterns.md`](identification-writing-patterns.md) §3:

| Claim | Requires |
|---|---|
| *causes, drives, reduces* | exogenous variation + the robustness package above |
| *explains, accounts for, spans* | spanning regression + joint α test |
| *predicts, forecasts* | out-of-sample R², Clark-West |
| *is associated with* | within-unit FE + honest framing |
| *we document* | descriptive statistics |

If the design does not support the verb, change the verb — not the design.
