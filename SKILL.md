---
name: causal-inference-mixtape
description: This skill should be used when the user asks to "implement a DiD regression", "run diff-in-discontinuities", "set up an event study", "implement IV / 2SLS", "run a regression discontinuity design", "build synthetic control", "saturated interacted FE", "weather-shock IV", "Callaway-Sant'Anna", "Sun-Abraham", or needs causal-inference code templates in Python/R/Stata with finance applications.
version: 2.1.0
---

# Causal Inference: The Mixtape — Code Skill (v2.1)

Provides practitioner-oriented causal-inference templates covering 10 identification strategies in Python, R, and Stata, extended with 25 top-tier Journal of Finance (2021-2024) application case studies. Based on Scott Cunningham's *Causal Inference: The Mixtape* plus mined JF 2021-2024 top-cited papers for finance-specific patterns.

---

## Methods Covered

| Method | Python | R | Stata | Reference |
|--------|--------|---|-------|-----------|
| OLS / Regression | statsmodels | estimatr | reg/reghdfe | `references/method-patterns.md` §1 |
| Difference-in-Differences | statsmodels + C() | lfe/fixest | xtreg/reghdfe | `references/method-patterns.md` §2 |
| Event Study (Dynamic DiD) | manual lead/lag | estimatr | reghdfe | `references/method-patterns.md` §3 |
| Staggered DiD / TWFE | statsmodels | bacondecomp | bacondecomp | `references/method-patterns.md` §4 |
| Regression Discontinuity | statsmodels polynomial | rdrobust | rdplot/rdrobust | `references/method-patterns.md` §5 |
| Instrumental Variables | linearmodels IV2SLS | AER/ivreg | ivregress 2sls | `references/method-patterns.md` §6 |
| Synthetic Control | rpy2 → R Synth | Synth + SCtools | synth | `references/method-patterns.md` §7 |
| Matching / PSM / IPW | manual logit + weights | MatchIt + Zelig | teffects/cem | `references/method-patterns.md` §8 |
| DAGs / Collider Bias | dagitty (conceptual) | dagitty/ggdag | — | `references/method-patterns.md` §9 |
| Randomization Inference | permutation loop | ri2 | ritest | `references/method-patterns.md` §10 |

---

## Core Workflow

### Implement a Causal Method

1. Identify the method from the table above
2. Load the appropriate template from `references/method-patterns.md`
3. Adapt variable names, fixed effects, and clustering to the user's data
4. Add robustness checks (parallel trends for DiD, McCrary for RDD, first-stage F for IV)

### Choose the Right Language

| Scenario | Recommendation |
|----------|---------------|
| ML pipeline integration | Python (statsmodels + linearmodels) |
| Synthetic Control | R (Synth package) or Stata (synth) — Python lacks mature implementation |
| Bacon decomposition | R (bacondecomp) or Stata — no Python equivalent |
| Publication-ready tables | Stata (outreg2/esttab) or R (stargazer/modelsummary) |
| Coarsened Exact Matching | Stata (cem) or R (MatchIt) — no Python equivalent |
| Quick prototyping | Python with statsmodels |

For cross-language syntax equivalents (OLS, cluster SE, two-way FE, IV, DiD), see `references/r-stata-comparison.md` §Cross-Language Equivalents.

---

## Key Python Patterns

### DiD with Cluster-Robust SE

```python
import statsmodels.formula.api as smf

model = smf.ols('y ~ C(treated)*C(post) + controls', data=df)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['firm_id']})
```

### Event Study (Lead/Lag)

```python
# Create relative time dummies
for k in range(-4, 5):
    col = f'rel_{k}' if k >= 0 else f'rel_m{abs(k)}'
    df[col] = (df['relative_time'] == k).astype(int)

# Drop t=-1 as reference
formula = 'y ~ ' + ' + '.join([c for c in rel_cols if c != 'rel_m1']) + ' + C(id) + C(year)'
```

### IV / 2SLS

```python
from linearmodels.iv import IV2SLS

model = IV2SLS.from_formula('y ~ 1 + exog + [endog ~ instrument]', data=df)
results = model.fit(cov_type='clustered', clusters=df['cluster_var'])
```

---

## Robustness Check Patterns

| Method | Required Checks |
|--------|----------------|
| DiD | Parallel trends (event study plot), placebo treatment dates |
| RDD | McCrary density test, bandwidth robustness (half/double IK optimal), polynomial robustness |
| IV | First-stage F > 10, exclusion restriction argument, over-identification test |
| Synthetic Control | Pre-treatment RMSPE, placebo distribution, leave-one-out |
| Matching | Covariate balance table, caliper sensitivity |

---

## Common Pitfalls

1. **TWFE with staggered treatment** — standard two-way FE is biased when treatment timing varies. Use Bacon decomposition or Sun & Abraham / Callaway & Sant'Anna estimators.
2. **Synthetic Control with many treated units** — the Synth package handles one treated unit. For multiple, use augmented synthetic control or stacked approach.
3. **RDD without McCrary test** — always test for manipulation at the cutoff before estimating.
4. **IV weak instruments** — report first-stage F-statistic. Below 10 indicates weak instrument bias.
5. **Python Synth gap** — no mature Python Synth package exists. Use `rpy2` to call R's `Synth` from Python.
6. **Non-causal framing vs. causal identification** — asset-pricing "factor captures the cross-section" is a *spanning* claim, not a causal one. Do not use "cause / drive / affect" without exogenous variation. See `references/identification-writing-patterns.md` §3 for verb guide.
7. **Saturated FE absorbing treatment** — `county × year` FE absorbs any state-level policy. Always check that FE structure does not soak up the treatment itself.
8. **Over-clustering** — cluster at the treatment-assignment level (Abadie-Athey-Imbens-Wooldridge 2023), not the highest FE level.
9. **Cross-country panel ≠ causal** — rich FE reduce confounds but don't produce exogenous variation. Bolton 2023 is honest about this; follow that pattern.
10. **Multiple-testing / factor zoo** — apply Benjamini-Yekutieli FDR or Bayesian theme-shrinkage (Jensen-Kelly-Pedersen 2023) before trusting any single α.

---

## Additional Resources

### Reference Files

- **`references/method-patterns.md`** — Detailed code templates for all 10 methods with full examples
- **`references/r-stata-comparison.md`** — Cross-language package comparison and method coverage gaps
- **`references/finance-applications.md`** — 25-paper case-sketch catalog indexed by method (DiD / IV / RDD / event study / structural / Fama-MacBeth) with Y / D / unit / FE / clustering / data-source filled in from top-tier JF papers
- **`references/identification-writing-patterns.md`** — Exemplar identification paragraphs, finance-specific verb guide (causal vs. spanning vs. descriptive), and design-specific robustness checklists
- **`references/jf-case-studies.md`** — Five end-to-end worked examples (Bennedsen DiD + DDD + diff-in-disc, Kempf saturated FE, Brown weather-shock IV, Barber sharp RDD + outage, Meeuwis election × partisan DiD) with full Python / R / Stata code skeletons

### Prompt Files

- **`prompts/01-implement-method.md`** — Copy-paste prompt for implementing any causal method
- **`prompts/02-robustness-checks.md`** — Copy-paste prompt for generating robustness check code
- **`prompts/03-finance-application.md`** — Copy-paste prompt for implementing a causal method on a finance dataset, with robustness checklist and identification-paragraph drafting
