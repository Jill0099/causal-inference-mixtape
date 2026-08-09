# StatsPAI — the Python-native causal inference workbench

**Use this as the default Python backend for every method in this skill.**

[StatsPAI](https://github.com/brycewang-stanford/statspai) (`pip install statspai`,
imported as `import statspai as sp`) is a single package covering the Stata/R
econometrics surface — `regress`, `ivregress`, `reghdfe`, `csdid`, `rdrobust`,
`synth`, `psmatch2`, `outreg2` and their R equivalents — with Python-native
result objects. It is why the "Python can't do X, switch to R" advice that
older causal-inference cheatsheets carry is now out of date for almost every X.

Everything in this file was executed against StatsPAI **1.21.0** with pandas
3.0.2 / numpy 2.4.4 on the Mixtape's own public datasets. The runnable proofs
live in [`scripts/`](../scripts/); `python scripts/validate_all.py` re-checks them.

---

## §1. Why it matters for this skill

The old coverage story for Python was a list of holes: no Bacon decomposition,
no Callaway-Sant'Anna, no Sun-Abraham, no McCrary test, no synthetic control, no
DAG tooling, no wild cluster bootstrap. Every one of those is now a single call:

| Gap the old tables listed as "Python: **None**" | StatsPAI entry point | Verified in |
|---|---|---|
| Goodman-Bacon decomposition | `sp.bacon_decomposition` | `scripts/04_staggered_did.py` |
| Callaway-Sant'Anna | `sp.callaway_santanna` + `sp.aggte` | `scripts/04_staggered_did.py` |
| Sun-Abraham | `sp.sun_abraham` | `scripts/04_staggered_did.py` |
| Borusyak-Jaravel-Spiess imputation | `sp.did_imputation` | `scripts/04_staggered_did.py` |
| McCrary / Cattaneo-Jansson-Ma density test | `sp.mccrary_test`, `sp.rddensity` | `scripts/05_rdd.py` |
| Synthetic control + placebo inference | `sp.synth` | `scripts/07_synthetic_control.py` |
| Synthetic DiD | `sp.sdid` | `scripts/07_synthetic_control.py` |
| Wild cluster bootstrap | `sp.wild_cluster_bootstrap` | `scripts/10_inference_clusters.py` |
| Honest DiD (Rambachan-Roth) | `sp.honest_did` (`backend="native"`, pure Python) | `scripts/11_honest_did.py` |
| Pre-trend power (Roth 2022) | `sp.pretrends_power` | `scripts/11_honest_did.py` |
| Conley spatial SE | `sp.conley` | — |
| DAG adjustment sets | `sp.dag`, `sp.check_identification` | `scripts/simulations/sim_collider_bias.py` |
| Coarsened exact matching | `sp.match(method="cem")` | — |
| Randomisation inference | `sp.ri_test` | `scripts/09_randomization_inference.py` |
| Oster bounds / sensitivity | `sp.oster_bounds`, `sp.sensemakr` | — |

There is also an **MCP server** (`statspai-mcp`) exposing the same functions as
agent tools, so an agent can estimate directly rather than emitting code. See §7.

---

## §2. Coming from Stata or R

| What you used before | StatsPAI |
|---|---|
| `reg y x, vce(robust)` / `lm()` + sandwich | `sp.regress("y ~ x", data=df, robust="hc1")` |
| `reg y x, cluster(id)` | `sp.regress("y ~ x", data=df, cluster="id")` |
| `reghdfe y x, absorb(i t) cluster(i)` / `feols(y ~ x \| i + t)` | `sp.feols("y ~ x \| i + t", data=df, vcov={"CRV1": "i"})` |
| `ivregress 2sls y (d = z) x` / `AER::ivreg` | `sp.ivreg("y ~ (d ~ z) + x", data=df, robust="hc1")` |
| `csdid` / `did::att_gt` | `sp.callaway_santanna(...)` then `sp.aggte(...)` |
| `eventstudyinteract` / `fixest::sunab` | `sp.sun_abraham(...)` |
| `bacondecomp` | `sp.bacon_decomposition(...)` |
| `rdrobust` (Stata or R) | `sp.rdrobust(data=df, y=..., x=..., c=...)` |
| `rddensity` | `sp.rddensity(data=df, x=..., c=...)` |
| `synth` / `Synth::synth` | `sp.synth(...)` |
| `sdid` | `sp.sdid(...)` |
| `psmatch2` | `sp.psmatch2(...)` |
| `teffects ipw` | `sp.ipw(...)` |
| `cem` / `MatchIt(method="cem")` | `sp.match(..., method="cem")` |
| `boottest` / `fwildclusterboot` | `sp.wild_cluster_bootstrap(...)` |
| `honestdid` / `HonestDiD` | `sp.honest_did(...)` |
| `ritest` / `ri2` | `sp.ri_test(...)` |
| `outreg2` / `esttab` | `sp.outreg2(r1, r2, filename="results.xlsx")` |
| `modelsummary` / `stargazer` | `sp.modelsummary(r1, r2, output="table.docx")` |

Note the **formula conventions differ by family**: `sp.regress` / `sp.ivreg` take
a formula string; `sp.feols` takes fixest-style `y ~ x | fe1 + fe2`; the causal
estimators (`callaway_santanna`, `sun_abraham`, `rdrobust`, `synth`, `ipw`,
`match`) take **keyword column names**, not formulas.

---

## §3. The result-object protocol

Estimators return a `CausalResult` or `EconometricResults` with a common surface:

```python
r = sp.callaway_santanna(data=df, y="y", g="first_treat", t="year", i="unit")
agg = sp.aggte(r, type="simple", bstrap=False)

agg.estimate        # point estimate (float)
agg.se              # standard error
agg.pvalue, agg.ci
agg.method          # WHICH estimator actually ran -- read this before quoting
agg.tidy()          # broom-style DataFrame
agg.summary()       # printable table
agg.model_info      # dict: weights, diagnostics, backend, convergence
agg.diagnostics     # dict: design-specific checks
agg.to_latex() / .to_markdown() / .to_excel() / .to_docx()
agg.to_agent_summary()   # compact structured payload for an LLM
```

**Tidy column names are broom-style, not fixest-style.** This bites when you mix
backends in one script:

| Backend | index | estimate column | SE column |
|---|---|---|---|
| `pyfixest` | coefficient names | `Estimate` | `Std. Error` |
| `StatsPAI` | `term` column | `estimate` | `std_error` |

`scripts/_common.py` ships a `pick(tidy, term)` helper that normalises both.

---

## §4. The recommended workflow

StatsPAI has a built-in design router. Use it as a second opinion on the design
you already chose, not as a substitute for choosing one.

```python
import statspai as sp

sp.detect_design(df, unit="sid", time="year")
# -> {'design': 'panel', 'confidence': 1.0, 'identified': {...}, 'candidates': [...]}

sp.recommend(df, y="l_homicide", treatment="treat", id="sid", time="year")
# -> recommendations[0] = Callaway-Sant'Anna (2021) — staggered DID
#    reason: "Multiple time periods with staggered treatment adoption.
#             Robust to heterogeneous treatment effects (unlike TWFE)."
#    warnings: ['60% observations have missing values...']

sp.preflight(df, method="did", y="l_homicide", treat="treat", time="year", id="sid")
# -> {'verdict': 'PASS', 'summary': {'passed': 8, 'warning': 0, 'failed': 0}, ...}
```

Then estimate, then audit:

```python
r  = sp.callaway_santanna(data=df, y="y", g="first_treat", t="year", i="unit")
hd = sp.honest_did(sp.aggte(r, type="dynamic", bstrap=False), e=0,
                   m_grid=[0, 0.5, 1.0, 2.0], method="relative_magnitude")
sp.bibtex(keys=["callaway2021", "rambachan2023"])   # verified citations, never invent them
```

---

## §5. Verified numbers

Estimates below were produced by the scripts in this repository on Mixtape data.
They are the regression tests: if a StatsPAI upgrade moves them, something changed.

| Check | Result |
|---|---|
| 2x2 DiD on castle (2006 cohort): hand-computed vs statsmodels vs pyfixest vs `sp.feols` | all `0.06824` |
| Event study on castle: 9 hand-built binned dummies vs `sp.event_study` | agree to ~`1e-16` on every coefficient |
| Bacon decomposition weighted sum vs TWFE coefficient | both `+0.08181`, weights sum to `1.0000` |
| Staggered castle: TWFE `+0.08181` vs Callaway-Sant'Anna `+0.11038` vs Sun-Abraham `+0.11028` | CS and SA agree; TWFE is the outlier |
| Card IV: `linearmodels.IV2SLS` vs `sp.ivreg`, coefficient on `educ` | both `0.13229` |
| LMB sharp RDD: `sp.rdrobust` robust bias-corrected effect | `18.449` (se `2.038`), MSE-optimal `h = 0.0863` |
| LMB density test (`sp.rddensity`, Cattaneo-Jansson-Ma) | p `0.4944` — no manipulation |
| NSW experimental benchmark | `$1,794.34` (matches the published LaLonde figure) |
| Mixtape `training_example.dta` subclassification | raw `−$26.25`, age-matched `+$1,695` (the book's numbers) |
| Wild cluster bootstrap on castle: `sp.wild_cluster_bootstrap` vs `pyfixest.wildboottest` | p `0.5733` vs `0.5693` |
| Texas synth ATT vs `sp.sdid` ATT | `21,482` vs `19,479`, permutation p `0.18` |
| Potential-outcomes decomposition identity residual | `2.7e-15` |

---

## §6. Gotchas found by running it

Each of these cost real debugging time and none is obvious from the docstring.
They are behaviours of StatsPAI 1.21.0.

1. **`sp.ipw(trim=α)` CLIPS the propensity score, it does not DROP off-support
   units.** The docstring cites Crump et al. (2009), which is a *discard* rule.
   On NSW/CPS the difference is enormous: dropping units outside `[0.1, 0.9]`
   gives `+$1,454` (near the `$1,794` experimental truth), while clipping gives
   `−$7,936`, because 15,000 CPS controls whose true pscore is ≈0 have their
   control weight raised to `p/(1−p) = 0.111` each. Check `model_info["n_obs"]`
   and `model_info["pscore_min"]` to see which one you got.

2. **`sp.ri_test` and `sp.wild_cluster_bootstrap` can report `p = 0.0` exactly.**
   Both omit the `+1` correction, i.e. they use `#{|t*| ≥ |t|} / B` rather than
   `(1 + #{|t*| ≥ |t|}) / (1 + B)`. An exact zero is not an attainable
   permutation p-value. Report `p < 1/(1+B)` instead.

3. **`sp.synth(covariates=[...])` runs the nested V-weight optimisation and is
   orders of magnitude slower.** On a 51-unit × 16-year panel: ~5 s without
   covariates, >2 minutes with seven of them. Start without (pre-treatment
   outcome lags are the classic Abadie predictor set) and add them only if the
   pre-treatment fit demands it.

4. **On a synth result, `.params` is the ATT — the donor weights are at
   `model_info["weights"]`,** as an `(n, 2)` array of `(unit_id, weight)` pairs.
   Fit statistics are at `model_info["pre_treatment_rmse"]`, not in `.diagnostics`.

5. **`sp.event_study` needs the FULL panel with never-treated units carrying a
   missing `treat_time`.** Dropping them first looks reasonable and produces
   standard errors ~1e14 — the loudest possible symptom, but only if you look.

6. **`sp.did` chooses its own estimator.** On the castle 2x2 it returned
   `0.05172` where the hand-built TWFE gives `0.06824`. Read `.method` before
   quoting it; use `sp.feols` when you need exact control of the specification.

7. **`sp.psmatch2` returns `.att`, not `.estimate`,** and `common_support` accepts
   only `{'none', 'minmax'}` — `'on'` raises.

8. **`sp.rdrobust(h=...)` also changes how the bias-correction bandwidth `b` is
   chosen,** so a manual bandwidth sweep will not reproduce the MSE-optimal run
   at the middle point. Report the MSE-optimal estimate as the headline and the
   sweep as robustness.

9. **`sp.wild_cluster_bootstrap(x=[...])` takes a plain list of regressor
   columns with no fixed-effect syntax.** Passing only your treatment variable
   silently estimates a model without fixed effects and returns a `beta_hat`
   that is not your DiD estimate. Materialise FE dummies and pass them, then
   assert `beta_hat` equals your analytic coefficient.

10. **`sp.detect_design` takes `unit` / `time` / `running_var` / `cutoff` only** —
    there is no `treat`/`treatment` keyword. `sp.recommend` is the one that takes
    `treatment=`.

11. **Some Mixtape `.dta` files load as float32 or as string columns with blanks.**
    Cross-backend equality checks need a `1e-5` tolerance, not machine epsilon,
    and `training_example.dta` needs `pd.to_numeric(..., errors="coerce")`.

---

## §7. The MCP server: estimate directly instead of emitting code

`statspai-mcp` exposes the library as agent tools. Prefer it over generating a
script when:

- the user wants **an answer**, not a deliverable script;
- you need a quick design check (`detect_design`, `preflight`, `recommend`) before
  committing to a specification;
- you want the robustness sweep enumerated for you (`audit_result` emits the
  missing checks with the function to call for each);
- you need **verified citations** — `bibtex(keys=[...])` reads from a curated
  `paper.bib`. Never hand-write a reference.

Prefer generating a script when the user needs something reproducible, wants it
in their repo, or is going to hand it to a coauthor or a referee.

Chaining pattern: pass `as_handle=True` to get a `result_id`, then feed that to
`audit_result`, `honest_did_from_result`, `sensitivity_from_result` and friends
instead of re-supplying data and columns.

---

## §8. When NOT to reach for StatsPAI

Being the default backend does not make it the only one.

- **`pyfixest`** is the fastest route for very large high-dimensional FE models
  and mirrors `fixest` syntax exactly — useful when porting R code line by line,
  and it carries an independent `wildboottest` implementation worth cross-checking
  against.
- **`linearmodels`** has the deepest IV/panel diagnostic surface (Wooldridge
  overid, Kleibergen-Paap, Anderson-Rubin, panel GMM).
- **`scpi`** (Cattaneo, Feng, Palomba, Titiunik) is the reference synthetic-control
  implementation with prediction intervals, by the authors of the method.
- **R** remains the right answer when a referee asks for a specific package's
  output, or for a brand-new estimator whose only implementation is on CRAN.
- **Stata** when the co-author's workflow is Stata and reproducibility means
  their `.do` file runs. The companion
  [`stata-code`](https://github.com/brycewang-stanford/stata-code/) MCP server
  lets an agent read, run and translate that workflow.

Cross-checking two independent implementations is cheap and it is the only real
defence against a silent API misuse. Several of the gotchas in §6 were found
exactly that way.
