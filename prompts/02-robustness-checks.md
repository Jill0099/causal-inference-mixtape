# Prompt 2: Generate Robustness Check Code

Copy and paste the relevant section below based on your identification strategy.

---

## For DiD Papers

```
You are an expert econometrician. Generate robustness check code in [Python / R / Stata] for my DiD analysis.

Produce code for ALL of the following:

1. EVENT STUDY (parallel trends):
   - Dynamic specification with lead/lag dummies
   - Plot coefficients with 95% CI
   - Pre-period joint F-test

2. PLACEBO TREATMENT DATE:
   - Re-estimate using a fake treatment date [N] years before actual treatment
   - Expect null result

3. BACON DECOMPOSITION (if staggered):
   - Decompose TWFE into 2x2 comparisons; verify the weights sum to 1 and the
     weighted sum reproduces the TWFE coefficient
   - Report the WEIGHT on "already-treated as control" comparisons
   - Then re-estimate with Callaway-Sant'Anna, Sun-Abraham and BJS imputation,
     and report one of those as the headline with TWFE as a benchmark

3b. HONEST DiD (Rambachan-Roth 2023) -- do this instead of stopping at a flat
    pre-trend plot:
   - Report the POWER of the joint pre-trend test (Roth 2022). If it is below
     0.8, say so: "we cannot reject" is then very weak evidence.
   - Robust confidence sets over M = 0, 0.25, 0.5, 1, 2
   - State the BREAKDOWN M in words. A breakdown below 1 means the result
     cannot survive a violation no larger than one already visible pre-treatment.

3c. INFERENCE:
   - Cluster at the level treatment was ASSIGNED, and say what that level is
   - Report the number of clusters AND the number of TREATED clusters
   - If clusters < 50: wild cluster bootstrap, asserting its beta matches the
     analytic coefficient
   - If clusters < 10: randomisation inference instead
   - Permutation/bootstrap p-values: report "p < 1/(1+B)", never "p = 0.000"

4. ALTERNATIVE CONTROL GROUP:
   - Re-estimate dropping [specific units] from control group
   - Verify results hold

5. ENTROPY BALANCING / PSM-DID:
   - Re-weight sample to achieve covariate balance
   - Re-estimate on balanced sample

My details:
- Treatment: [e.g., 2014 SOE reform]
- Treated group: [e.g., listed SOEs]
- Control group: [e.g., non-SOE listed firms]
- Treatment timing: [e.g., 2014 for all / staggered]
- Outcome: [e.g., abnormal investment]
- Pre-period: [e.g., 2010-2013]
- Post-period: [e.g., 2015-2018]
- Covariates for balancing: [e.g., size, leverage, ROA, age]
```

---

## For RDD Papers

```
You are an expert econometrician. Generate robustness check code in [Python / R / Stata] for my RDD analysis.

Produce code for ALL of the following:

0. SHARPNESS CHECK:
   - P(treated | just below) vs P(treated | just above). If compliance is not
     near-perfect this is a FUZZY design and crossing the cutoff instruments
     for treatment -- say so before anything else.

1. MANIPULATION / DENSITY TEST -- run this FIRST, before any estimate:
   - Cattaneo-Jansson-Ma (2020) via rddensity (native in Python, R and Stata --
     the old `rdd::DCdensity` advice is obsolete)
   - Report test statistic and p-value; plot the density
   - If it rejects, the design is dead and no bandwidth choice repairs it

2. HEADLINE ESTIMATE:
   - MSE-optimal bandwidth, triangular kernel, ROBUST BIAS-CORRECTED CI
   - A hand-rolled `y ~ D*x` OLS inside a chosen bandwidth uses a uniform kernel,
     no bias correction and conventional SEs -- it undercovers. Show it only as
     a benchmark, never as the headline.
   - Report effective N on each side of the cutoff

3. COVARIATE BALANCE:
   - PREDETERMINED variables only. The treatment indicator is SUPPOSED to jump;
     putting it in a balance table is a category error.
   - Report coefficient, SE and a verdict per covariate

4. BANDWIDTH ROBUSTNESS:
   - Re-estimate at h/2, h, 2h (h = MSE-optimal)
   - Note that passing h explicitly also changes how the bias-correction
     bandwidth b is chosen, so the middle row need not match the headline

5. POLYNOMIAL ROBUSTNESS:
   - Local linear (p=1) and quadratic (p=2) only
   - STOP at 2: Gelman & Imbens (2019) show high-order global polynomials give
     noisy weights and poor coverage. Never report a cubic or quartic as headline.

6. PLACEBO CUTOFFS:
   - Re-estimate at false cutoffs away from the true one; expect nulls
   - Flag any placebo that is significant at 5%

My details:
- Running variable: [e.g., vote share percentage]
- Cutoff: [e.g., 50% majority threshold]
- Outcome: [e.g., firm ESG score]
- IK optimal bandwidth: [e.g., 8.3 percentage points]
- Covariates for balance test: [e.g., firm size, age, leverage]
```

---

## For IV Papers

```
You are an expert econometrician. Generate robustness check code in [Python / R / Stata] for my IV analysis.

Produce code for ALL of the following:

1. FIRST-STAGE DIAGNOSTICS:
   - First-stage regression with the F-statistic on the EXCLUDED instruments
   - Cragg-Donald / Kleibergen-Paap F
   - State WHICH criterion you are invoking: Staiger-Stock F>10 bounds relative
     bias; Lee, McCrary, Moreira & Porter (2022) show valid 5% t-test inference
     needs F>104.7 or a tF-adjusted critical value. Between 10 and 104.7 the
     point estimate is usable but a conventional t-test over-rejects.
   - If weak: report an ANDERSON-RUBIN confidence interval, which is valid
     regardless of instrument strength.

2. EXCLUSION RESTRICTION:
   - Argue it in PROSE. There is no test for it. Placebo regressions are
     suggestive at best -- do not present one as evidence of validity.
   - Name the specific alternative channel a sceptic would propose, and say why
     it does not operate here.

3. OVER-IDENTIFICATION TEST (if multiple instruments):
   - Hansen J / Sargan; report statistic and p-value
   - State the caveat: this only asks whether the instruments AGREE WITH EACH
     OTHER. Two instruments violating exclusion in the same direction pass.
     A failure may also just mean they identify different LATEs.

4. REDUCED FORM:
   - Regress the outcome directly on the instrument(s)
   - With one instrument, reduced form / first stage reproduces the 2SLS
     coefficient exactly -- verify this as an arithmetic check

5. OLS vs IV COMPARISON:
   - Report side by side with the Wu-Hausman test
   - If IV > OLS, ability-bias intuition alone does not explain it -- discuss
     what that says about who the compliers are

6. ESTIMAND:
   - State the result as a LATE, not "the effect"
   - Report the complier share (the first stage) and characterise compliers
   - Discuss MONOTONICITY explicitly: who would the defiers be, and why are
     there none?

7. IMPLEMENTATION HYGIENE:
   - Use a real 2SLS routine. A manual two-step reproduces the point estimate
     exactly and reports the WRONG standard error, in a data-dependent direction.
   - Never plug a logit/probit first-stage fitted value in as a REGRESSOR
     (the forbidden regression). Use it as an INSTRUMENT if you must.

My details:
- Endogenous variable: [e.g., corruption level]
- Instrument(s): [e.g., ethnic fractionalization, distance to coast]
- Outcome: [e.g., patent count]
- Controls: [e.g., GDP per capita, education, population]
- First-stage F: [e.g., 23.4]
```
