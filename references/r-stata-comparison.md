# Cross-Language Coverage — Python / R / Stata

Package recommendations and coverage, refreshed against installed versions
(StatsPAI 1.21.0, pyfixest 0.50.1, statsmodels 0.14.6, linearmodels 6.1,
pandas 3.0.2, numpy 2.4.4).

> **The headline change from older versions of this table: Python no longer has
> holes.** Every method previously listed as "Python: **None**" — Bacon
> decomposition, Callaway-Sant'Anna, Sun-Abraham, McCrary, CEM, synthetic
> control, DAGs, wild cluster bootstrap — has a native Python implementation
> today. See [`statspai-guide.md`](statspai-guide.md). Do not tell a user to
> leave Python for any of them.

---

## §1. Cross-Language Equivalents

| Task | Python | R | Stata |
|---|---|---|---|
| OLS, robust SE | `sp.regress(..., robust="hc1")` | `lm_robust()` | `reg y x, robust` |
| Cluster SE | `sp.regress(..., cluster="id")` | `feols(y ~ x, cluster = ~id)` | `reg y x, cluster(id)` |
| Two-way FE | `sp.feols("y ~ x \| id + t")` | `feols(y ~ x \| id + t)` | `reghdfe y x, absorb(id t)` |
| Saturated interacted FE | `sp.feols("y ~ x \| a^b + c")` | `feols(y ~ x \| a^b + c)` | `reghdfe y x, absorb(a#b c)` |
| IV / 2SLS | `sp.ivreg("y ~ (d ~ z) + x")` | `ivreg()` / `feols(y ~ x \| fe \| d ~ z)` | `ivregress 2sls` / `ivreghdfe` |
| DiD (2×2) | `sp.feols("y ~ treat_post \| id + t")` | `feols(y ~ treat:post \| id + t)` | `reghdfe y treat_post, absorb(id t)` |
| Event study | `sp.event_study(...)` | `feols(y ~ i(t, treat, ref = -1) \| ...)` | `reghdfe` + event-time dummies |
| Sharp / fuzzy RDD | `sp.rdrobust(...)` | `rdrobust()` | `rdrobust` |
| Density / manipulation | `sp.rddensity(...)` | `rddensity()` | `rddensity` |
| Synthetic control | `sp.synth(...)` | `Synth` + `SCtools` | `synth` |
| Publication tables | `sp.outreg2()` / `sp.modelsummary()` | `modelsummary` / `stargazer` | `esttab` / `outreg2` |

---

## §2. Method Coverage Matrix

`✔` = native, maintained implementation.

| Method | Python | R | Stata |
|---|---|---|---|
| OLS / robust SE | ✔ statspai, statsmodels | ✔ estimatr, fixest | ✔ `reg, robust` |
| Cluster SE | ✔ statspai, pyfixest | ✔ estimatr, fixest | ✔ `cluster()` |
| High-dimensional FE | ✔ pyfixest, pyhdfe, statspai | ✔ fixest (fast) | ✔ reghdfe (fast) |
| DiD (2×2) | ✔ statspai, pyfixest | ✔ fixest, did | ✔ reghdfe |
| Event study | ✔ `sp.event_study` | ✔ fixest, did | ✔ reghdfe + coefplot |
| **Bacon decomposition** | ✔ `sp.bacon_decomposition` | ✔ bacondecomp | ✔ bacondecomp |
| **Callaway-Sant'Anna** | ✔ `sp.callaway_santanna`, differences, csdid | ✔ did | ✔ csdid |
| **Sun & Abraham** | ✔ `sp.sun_abraham` | ✔ fixest (`sunab`) | ✔ eventstudyinteract |
| **Borusyak-Jaravel-Spiess** | ✔ `sp.did_imputation` | ✔ didimputation | ✔ did_imputation |
| de Chaisemartin-D'Haultfœuille | ✔ `sp.did_multiplegt_dyn` | ✔ DIDmultiplegt | ✔ did_multiplegt_dyn |
| **Honest DiD (Rambachan-Roth)** | ✔ `sp.honest_did` | ✔ HonestDiD | ✔ honestdid |
| **Pre-trend power (Roth 2022)** | ✔ `sp.pretrends_power` | ✔ pretrends | ✔ pretrends |
| Diff-in-discontinuities | ✔ `sp.rdrobust` + `sp.feols` | ✔ rdrobust + fixest | ✔ rdrobust + reg |
| Sharp / fuzzy RDD | ✔ `sp.rdrobust`, rdrobust-py | ✔ rdrobust | ✔ rdrobust |
| **McCrary / CJM density** | ✔ `sp.rddensity`, `sp.mccrary_test` | ✔ rddensity | ✔ rddensity |
| RD honest (Armstrong-Kolesár) | ✔ `sp.rd_honest` | ✔ RDHonest | — |
| Shift-share / Bartik | ✔ `sp.bartik`, ShiftShareSE | ✔ BartikInstruments | ✔ bartik |
| Weak-IV F (Lee et al. 2022) | ✔ `sp.tF_adjustment`, `sp.effective_f_test` | ✔ ivDiag | ✔ weakivtest |
| Anderson-Rubin CI | ✔ `sp.anderson_rubin_ci` | ✔ ivmodel | ✔ weakiv |
| IV / 2SLS | ✔ statspai, linearmodels | ✔ AER, fixest | ✔ ivregress, ivreghdfe |
| JIVE | ✔ `sp.jive` | — | ✔ jive |
| **Synthetic control** | ✔ `sp.synth`, scpi, pysyncon | ✔ Synth, SCtools | ✔ synth |
| Synthetic DiD | ✔ `sp.sdid`, pysyntheticdid | ✔ synthdid | ✔ sdid |
| Augmented SC | ✔ `sp.augsynth` | ✔ augsynth | ✔ sdid |
| Matrix completion | ✔ `sp.mc_panel` | ✔ gsynth | — |
| PSM / matching | ✔ `sp.match`, `sp.psmatch2` | ✔ MatchIt | ✔ teffects psmatch |
| **Coarsened exact matching** | ✔ `sp.match(method="cem")` | ✔ MatchIt (`method="cem"`) | ✔ cem |
| Entropy balancing / CBPS | ✔ `sp.ebalance`, `sp.cbps` | ✔ ebal, CBPS | ✔ ebalance |
| IPW / doubly robust | ✔ `sp.ipw`, `sp.aipw`, `sp.drdid` | ✔ ipw, DRDID | ✔ teffects ipw |
| Randomisation inference | ✔ `sp.ri_test` | ✔ ri2 | ✔ ritest |
| **Wild cluster bootstrap** | ✔ `sp.wild_cluster_bootstrap`, wildboottest, pyfixest | ✔ fwildclusterboot | ✔ boottest |
| Conley spatial SE | ✔ `sp.conley` | ✔ conleyreg | ✔ ols_spatial_HAC |
| **DAGs / adjustment sets** | ✔ `sp.dag`, `sp.check_identification`, dowhy, causal-learn | ✔ dagitty, ggdag | — |
| Oster bounds / sensitivity | ✔ `sp.oster_bounds`, `sp.sensemakr` | ✔ robomit, sensemakr | ✔ psacalc |
| Double / debiased ML | ✔ `sp.dml`, DoubleML, EconML | ✔ DoubleML | — |
| Multiple-testing correction | ✔ `sp.romano_wolf`, `sp.benjamini_hochberg` | ✔ `p.adjust`, wildrwolf | ✔ rwolf, multproc |

Bold rows are the ones older cheatsheets list as Python gaps. None of them are.

---

## §3. Python package roles

| Package | Use it for | Install |
|---|---|---|
| **statspai** | Default backend: causal estimators + diagnostics + export | `pip install statspai` |
| pyfixest | Fastest HDFE; fixest-syntax ports from R; independent wild bootstrap | `pip install pyfixest` |
| linearmodels | Deepest IV / panel diagnostics (overid, Kleibergen-Paap, panel GMM) | `pip install linearmodels` |
| statsmodels | Plain OLS / GLM / logit; the formula layer | `pip install statsmodels` |
| pyhdfe | Residualising matrices against high-dimensional FE by hand | `pip install pyhdfe` |
| scpi | Reference synthetic control with prediction intervals (method's authors) | `pip install scpi-pkg` |
| wildboottest | Standalone wild cluster bootstrap; cross-check for `sp.` | `pip install wildboottest` |
| dowhy / causal-learn | DAG identification and causal discovery | `pip install dowhy causal-learn` |
| rpy2 | Fallback bridge when a referee asks for a specific CRAN package | `pip install rpy2` |

### Retired advice

| Old recommendation | Status | Use instead |
|---|---|---|
| `Zelig` (R) | Archived from CRAN | `MatchIt` + `marginaleffects` |
| `rdd::DCdensity` (R) | Superseded; package long unmaintained | `rddensity` (all three languages) |
| `lfe::felm` (R) | Effectively superseded by `fixest` | `fixest::feols` |
| "Use rpy2 for Synth" | Obsolete | `sp.synth`, `scpi`, `pysyncon` |
| "Bacon decomposition: R or Stata only" | Obsolete | `sp.bacon_decomposition` |
| "McCrary test: R only" | Obsolete | `sp.rddensity` |
| "DAGs: R only" | Obsolete | `sp.dag`, `dowhy`, `causal-learn` |
| "boottest: shell out to Stata" | Obsolete | `sp.wild_cluster_bootstrap`, `wildboottest` |

---

## §4. R packages

| Package | Purpose |
|---|---|
| fixest | Fast FE estimation, IV, `sunab`, `i()` event studies |
| estimatr | Robust / cluster SE OLS |
| AER | IV / 2SLS with diagnostics |
| rdrobust, rddensity, RDHonest | RDD estimation, manipulation test, honest CIs |
| did, didimputation, DIDmultiplegt | Staggered DiD family |
| HonestDiD, pretrends | Parallel-trends sensitivity and power |
| Synth, SCtools, augsynth, synthdid, gsynth | Synthetic control family |
| MatchIt, ebal, CBPS, DRDID | Matching, balancing, doubly robust |
| fwildclusterboot, conleyreg | Small-cluster and spatial inference |
| dagitty, ggdag | DAGs and adjustment sets |
| ri2 | Randomisation inference |
| modelsummary, stargazer | Publication tables |

## §5. Stata packages

| Package | Purpose | Install |
|---|---|---|
| reghdfe, ivreghdfe | High-dimensional FE, with IV | `ssc install reghdfe ivreghdfe` |
| rdrobust, rddensity | RDD estimation and manipulation test | `ssc install rdrobust rddensity` |
| csdid, did_multiplegt_dyn, eventstudyinteract, did_imputation | Staggered DiD family | `ssc install ...` |
| honestdid, pretrends | Parallel-trends sensitivity and power | `ssc install honestdid pretrends` |
| bacondecomp | Goodman-Bacon decomposition | `ssc install bacondecomp` |
| synth, sdid | Synthetic control and synthetic DiD | `ssc install synth sdid` |
| cem, psmatch2, ebalance | Matching and balancing | `ssc install cem psmatch2 ebalance` |
| boottest | Wild cluster bootstrap | `ssc install boottest` |
| weakivtest, weakiv | Weak-IV tests and robust CIs | `ssc install weakivtest weakiv` |
| ritest | Randomisation inference | `ssc install ritest` |
| rwolf, multproc | Multiple-testing corrections | `ssc install rwolf multproc` |
| eventstudy2 | Financial event studies | `ssc install eventstudy2` |

---

## §6. When to switch languages

| Situation | Recommendation |
|---|---|
| Already in a Python pipeline | Stay. The gaps that justified leaving are closed. |
| Referee asks for a specific CRAN/SSC package's output | Match the language to the package. |
| Brand-new estimator, weeks old | Usually R first; check StatsPAI before assuming. |
| Co-author's reproducibility depends on a `.do` file | Stata — or the `stata-code` MCP server to read and translate it. |
| Publication tables | Any: `sp.outreg2` / `modelsummary` / `esttab` all export Word and LaTeX. |
| You want to be confident a result is not an API misuse | **Two languages.** Cross-checking is cheap and catches silent misuse — several gotchas in `statspai-guide.md` §6 were found exactly this way. |
