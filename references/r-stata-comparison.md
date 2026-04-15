# R & Stata Comparison — Methods Not Available in Python

Cross-language coverage gaps and package recommendations.

---

## Cross-Language Equivalents

| Task | Python | R | Stata |
|------|--------|---|-------|
| OLS with robust SE | `smf.ols().fit(cov_type='HC1')` | `lm_robust()` | `reg y x, robust` |
| Cluster SE | `fit(cov_type='cluster', cov_kwds={'groups': g})` | `felm(y ~ x | 0 | 0 | cluster)` | `reg y x, cluster(id)` |
| Two-way FE | `C(id) + C(time)` in formula | `felm(y ~ x | id + time)` or `feols(y ~ x | id + time)` | `reghdfe y x, absorb(id time)` |
| Saturated interacted FE | `pyhdfe.create([fe1, fe2, fe3])` | `feols(y ~ x | fe1^fe2 + fe3)` | `reghdfe y x, absorb(fe1#fe2 fe3)` |
| IV / 2SLS | `IV2SLS.from_formula('y ~ 1 + exog + [endog ~ inst]')` | `ivreg(y ~ exog | inst)` or `feols(y ~ x | fe | endog ~ inst)` | `ivregress 2sls y exog (endog = inst)` or `ivreghdfe` |
| DiD (2×2) | `C(treat)*C(post)` in formula | `treat:post` in formula | `did_multiplegt` or interaction |
| Event study | manual lead/lag dummies | `i(year, treat, ref = t0)` (fixest) | `reghdfe` with event-time dummies |
| Sharp RDD | `rdrobust` (py) or manual | `rdrobust` | `rdrobust` |
| McCrary density | `rddensity` (py) | `rddensity` or `rdd::DCdensity` | `rddensity` |
| Synthetic control | `rpy2` bridge | `Synth` + `SCtools` | `synth` |

---

## Method Coverage Matrix

| Method | Python | R | Stata |
|--------|--------|---|-------|
| OLS / Robust SE | statsmodels | estimatr | reg, robust |
| Cluster SE | statsmodels | estimatr/lfe | cluster() |
| Two-way FE | statsmodels (slow) | fixest (fast) | reghdfe (fast) |
| DiD (2x2) | statsmodels | did/fixest | reghdfe |
| Event Study | manual | fixest/did | reghdfe + coefplot |
| Bacon Decomposition | **None** | bacondecomp | bacondecomp |
| Callaway-Sant'Anna | **None** | did | csdid |
| Sun & Abraham | **None** | fixest (sunab) | eventstudyinteract |
| Staggered DiD (new) | differences (PyPI) | did / DIDmultiplegt | did_multiplegt_dyn |
| Diff-in-Discontinuities | rdrobust-py (manual) | rdrobust + fixest | rdrobust + reg |
| Sharp RDD | statsmodels (manual) | rdrobust | rdrobust |
| Fuzzy RDD | linearmodels | rdrobust | rdrobust |
| McCrary Test | **None** | rdd / rddensity | rddensity |
| Shift-Share / Bartik IV | ShiftShareSE (PyPI) | BartikInstruments | bartik (ssc) |
| Weak IV F (LMMP) | **None** (manual) | ivreg (with diag) | weakivtest |
| IV / 2SLS | linearmodels | AER/ivreg + fixest | ivregress / ivreghdfe |
| JIVE | **None** | **None** | jive |
| Synthetic Control | **rpy2 only** | Synth + SCtools | synth |
| Synthetic DiD | pysyntheticdid | synthdid | sdid |
| Augmented SC | **None** | augsynth | sdid |
| PSM / Matching | manual | MatchIt | teffects psmatch |
| CEM | **None** | MatchIt (method="cem") | cem |
| IPW | manual | ipw | teffects ipw |
| Randomization Inference | manual loop | ri2 | ritest |
| Event Study Package | eventstudies-py | eventstudies | eventstudy2 |
| Hierarchical Bayes MT | pymc / numpyro | rstanarm / brms | bayesmh |
| Benjamini-Yekutieli FDR | statsmodels.stats | stats::p.adjust | multproc |
| DAGs | **None** | dagitty + ggdag | **None** |

---

## Python Gaps — Recommended Workarounds

### Synthetic Control

No mature Python package. Two options:

1. **rpy2 bridge** (recommended for production):
```python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
ro.globalenv['df'] = df
ro.r('library(Synth); ...')
```

2. **SparseSC** (experimental): `pip install SparseSC` — limited functionality

### Bacon Decomposition

No Python implementation. Use R:
```r
library(bacondecomp)
bacon(y ~ treatment, data = df, id_var = "id", time_var = "year")
```

### Coarsened Exact Matching (CEM)

Stata's `cem` command is the gold standard. R alternative:
```r
library(MatchIt)
m.out <- matchit(treated ~ x1 + x2, data = df, method = "cem")
```

### McCrary Density Test

R implementation:
```r
library(rdd)
DCdensity(running_var, cutpoint = cutoff, plot = TRUE)
```

---

## Package Quick Reference

### R Packages

| Package | Purpose | Install |
|---------|---------|---------|
| estimatr | Robust/cluster SE OLS | `install.packages("estimatr")` |
| lfe | High-dimensional FE | `install.packages("lfe")` |
| fixest | Fast FE estimation | `install.packages("fixest")` |
| AER | IV / 2SLS | `install.packages("AER")` |
| rdrobust | RDD estimation | `install.packages("rdrobust")` |
| Synth | Synthetic control | `install.packages("Synth")` |
| SCtools | SC placebo tests | `install.packages("SCtools")` |
| MatchIt | Matching (PSM/CEM) | `install.packages("MatchIt")` |
| did | Callaway-Sant'Anna | `install.packages("did")` |
| bacondecomp | Bacon decomposition | `install.packages("bacondecomp")` |
| dagitty | DAG analysis | `install.packages("dagitty")` |
| ri2 | Randomization inference | `install.packages("ri2")` |
| ipw | Inverse probability weighting | `install.packages("ipw")` |

### Stata Packages

| Package | Purpose | Install |
|---------|---------|---------|
| reghdfe | High-dimensional FE | `ssc install reghdfe` |
| ivreghdfe | IV with high-dim FE | `ssc install ivreghdfe` |
| rdrobust | RDD estimation | `ssc install rdrobust` |
| rddensity | Manipulation test (McCrary) | `ssc install rddensity` |
| synth | Synthetic control | `ssc install synth` |
| sdid | Synthetic DiD / Augmented SC | `ssc install sdid` |
| cem | Coarsened exact matching | `ssc install cem` |
| bacondecomp | Bacon decomposition | `ssc install bacondecomp` |
| ritest | Randomization inference | `ssc install ritest` |
| did_multiplegt | Staggered DiD | `ssc install did_multiplegt` |
| did_multiplegt_dyn | Staggered DiD (dynamic) | `ssc install did_multiplegt_dyn` |
| eventstudyinteract | Sun & Abraham | `ssc install eventstudyinteract` |
| eventstudy2 | Financial event studies | `ssc install eventstudy2` |
| csdid | Callaway-Sant'Anna | `ssc install csdid` |
| weakivtest | Weak-IV F (LMMP 2022) | `ssc install weakivtest` |
| bartik | Shift-share IV | `ssc install bartik` |

### Python Packages

| Package | Purpose | Install |
|---------|---------|---------|
| statsmodels | OLS / WLS / GLM / logit | `pip install statsmodels` |
| linearmodels | IV2SLS / panel models | `pip install linearmodels` |
| pyhdfe | High-dimensional FE absorbing | `pip install pyhdfe` |
| rdrobust-py | RDD (Python port) | `pip install rdrobust` |
| differences | Staggered DiD / CS / Sun-Abraham | `pip install differences` |
| pysyntheticdid | Synthetic DiD | `pip install pysyntheticdid` |
| ShiftShareSE | Shift-share IV SE | `pip install ShiftShareSE` |
| plotnine | ggplot2-style plotting | `pip install plotnine` |
| rpy2 | Call R from Python | `pip install rpy2` |

---

## When to Switch Languages

| Situation | Recommendation |
|-----------|---------------|
| Already in a Python ML pipeline | Stay in Python, use rpy2 for gaps |
| Need Bacon decomposition | Switch to R or Stata |
| Synthetic control analysis | Use R (Synth) or Stata (synth) |
| Publication-ready regression tables | Stata (esttab) or R (modelsummary) |
| Exploratory / quick prototyping | Python statsmodels |
| Teaching / reproducibility | R (tidyverse ecosystem) |
| Referee asks for specific robustness | Match the language to the available package |
