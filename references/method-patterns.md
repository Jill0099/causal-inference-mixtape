# Method Patterns — Full Code Templates

Detailed code templates extracted from 58 Python scripts, ~56 R scripts, and ~60
Stata `.do` files in the Mixtape repository.

> **Every Python snippet below has a runnable, executed counterpart in
> [`scripts/`](../scripts/).** The templates here are for reading and adapting;
> the scripts are the proof they work. `python scripts/validate_all.py` runs all
> 13 against the Mixtape's public data and asserts cross-backend agreement.
>
> **Python backend order of preference:** `StatsPAI` (`import statspai as sp`)
> for causal estimators and their diagnostics → `pyfixest` for large
> high-dimensional FE → `linearmodels` for deep IV/panel diagnostics →
> `statsmodels` for plain OLS/GLM. See [`statspai-guide.md`](statspai-guide.md).
>
> R and Stata snippets are provided for cross-language work and have **not** been
> executed in this environment; treat them as syntax references.

---

## §1 OLS / Regression

### Python

```python
import pandas as pd
import statsmodels.formula.api as smf

# Basic OLS with robust SE
model = smf.ols('outcome ~ treatment + control1 + control2', data=df)
results = model.fit(cov_type='HC1')
print(results.summary())

# WLS (weighted least squares)
model = smf.wls('outcome ~ treatment', data=df, weights=df['weight'])
results = model.fit()
```

### R

```r
library(estimatr)

# OLS with robust SE (HC1)
model <- lm_robust(outcome ~ treatment + control1 + control2, data = df, se_type = "HC1")
summary(model)

# Clustered SE
model <- lm_robust(outcome ~ treatment, data = df, clusters = firm_id, se_type = "stata")
```

### Stata

```stata
* Basic OLS with robust SE
reg outcome treatment control1 control2, robust

* Cluster SE
reg outcome treatment control1 control2, cluster(firm_id)

* High-dimensional FE
reghdfe outcome treatment control1, absorb(firm_id year) cluster(firm_id)
```

---

## §2 Difference-in-Differences

### Python

Runnable version: [`scripts/02_did_2x2.py`](../scripts/02_did_2x2.py).

```python
# --- Preferred: StatsPAI -------------------------------------------------
import statspai as sp

df['treat_post'] = df['treated'] * df['post']
r = sp.feols('y ~ treat_post | entity_id + year', data=df, vcov={'CRV1': 'state'})
print(r.summary())
# tidy() is broom-style: columns are term / estimate / std_error
b = r.tidy().set_index('term').loc['treat_post', 'estimate']

# --- statsmodels: saturated 2x2 with NO fixed effects --------------------
import statsmodels.formula.api as smf

model = smf.ols('y ~ treated * post', data=df)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['state']})
did_coef = results.params['treated:post']
print(f"DiD estimate: {did_coef:.4f} (SE: {results.bse['treated:post']:.4f})")
```

> **Trap: do not write `C(treated)*C(post)` alongside entity and year FE.**
> With entity FE the `treated` main effect is collinear; with year FE `post` is
> collinear. statsmodels solves via pseudo-inverse and **silently splits the
> coefficient across the collinear columns instead of dropping one**, so the
> printed table is not interpretable. Stata's `reghdfe` would drop a term and
> tell you; statsmodels will not. With two-way FE, include the **interaction
> only**:

```python
# CORRECT with two-way FE: interaction only, main effects are absorbed
df['treat_post'] = df['treated'] * df['post']
model = smf.ols('y ~ treat_post + control1 + C(entity_id) + C(year)', data=df)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['state']})
```

Verified on `castle.dta` (2006 cohort vs never-treated): the hand-computed 2×2
difference, statsmodels, `pyfixest` and `sp.feols` all return **0.06824**.

### R

```r
library(lfe)

# DiD with two-way FE
model <- felm(y ~ treated:post + controls | entity_id + year | 0 | state, data = df)
summary(model)

# Alternative with fixest
library(fixest)
model <- feols(y ~ treated:post + controls | entity_id + year, data = df, cluster = ~state)
```

### Stata

```stata
* Standard DiD
reg y treated##post, cluster(state)

* With two-way FE
reghdfe y treated_post controls, absorb(entity_id year) cluster(state)

* Triple difference
reghdfe y treated##post##group controls, absorb(entity_id year) cluster(state)
```

---

## §3 Event Study (Dynamic DiD)

### Python

Runnable version with assertions: [`scripts/03_event_study.py`](../scripts/03_event_study.py).

Three things go wrong in the naive loop, and all three produce output that
*looks* fine:

1. **The reference period is not actually dropped.** `range(-4, 0)` is
   `[-4, -3, -2, -1]` — a loop that says "leads −4..−1, excluding −1" excludes
   nothing. Every relative-time indicator plus two-way FE is perfectly
   collinear and there is no normalisation.
2. **Endpoints are not binned.** Units observed outside the window get all-zero
   indicators, which silently treats them as if they were at the reference
   period. Bin them into the terminal leads/lags.
3. **The plot x-axis does not line up.** The reference period must be
   re-inserted as an exact zero with zero standard error.

```python
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

WINDOW, REF = (-5, 4), -1
lo, hi = WINDOW

# Never-treated units get NO event time -- flag them rather than faking one.
df['ever_treated'] = df['treatment_year'].notna().astype(int)
df['rel_time'] = np.where(df['ever_treated'] == 1,
                          df['year'] - df['treatment_year'], np.nan)

# Bin the endpoints, then build indicators for every period EXCEPT the reference.
binned = df['rel_time'].clip(lower=lo, upper=hi)
periods = [k for k in range(lo, hi + 1) if k != REF]      # <-- REF really excluded
names = []
for k in periods:
    name = f'lead{abs(k)}' if k < 0 else f'lag{k}'
    df[name] = ((binned == k) & (df['ever_treated'] == 1)).astype(int)
    names.append(name)

# Assert the normalisation rather than hoping for it.
on = df.loc[df['ever_treated'] == 1, names].sum(axis=1)
at_ref = binned[df['ever_treated'] == 1] == REF
assert (on[at_ref] == 0).all(),  'reference rows must have all-zero dummies'
assert (on[~at_ref] == 1).all(), 'every other treated row needs exactly one dummy'
assert df.loc[df['ever_treated'] == 0, names].to_numpy().sum() == 0

formula = 'y ~ ' + ' + '.join(names) + ' + C(entity_id) + C(year)'
results = smf.ols(formula, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['state']})

# Re-insert the reference period as an EXACT zero so x and y line up.
xs, coefs, ses = [], [], []
for k in range(lo, hi + 1):
    xs.append(k)
    if k == REF:
        coefs.append(0.0); ses.append(0.0); continue
    name = f'lead{abs(k)}' if k < 0 else f'lag{k}'
    coefs.append(results.params[name]); ses.append(results.bse[name])
assert len(xs) == len(coefs) == len(ses) == hi - lo + 1

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(xs, coefs, yerr=[1.96 * s for s in ses], fmt='o-', capsize=3)
ax.axhline(0, color='red', linestyle='--')
ax.axvline(REF + 0.5, color='grey', linestyle='--', alpha=0.5)
ax.set_xlabel(f'Periods relative to treatment (t={REF} normalised to 0)')
ax.set_ylabel('Coefficient estimate')
ax.set_xticks(xs)
plt.tight_layout()
```

### Python (StatsPAI, one call)

```python
import statspai as sp

# Pass the FULL panel.  Never-treated units keep a MISSING first_treat -- they
# are the comparison group.  Dropping them first leaves no clean controls and
# the standard errors blow up by ~1e14.
es = sp.event_study(data=df, y='y', treat_time='first_treat', time='year',
                    unit='entity_id', window=(-5, 4), ref_period=-1, cluster='state')
print(es.summary())
```

Verified on `castle.dta`: the hand-built binned specification and
`sp.event_study` agree to **~1e-16 on every coefficient**.

**Do not stop at "the pre-trends look flat."** Test the leads jointly, report
that test's *power*, and then report an Honest DiD breakdown value — see
[`scripts/11_honest_did.py`](../scripts/11_honest_did.py) and
[`inference-and-standard-errors.md`](inference-and-standard-errors.md) §5.

### R

```r
library(fixest)

# Sun & Abraham (2021) interaction-weighted estimator
model <- feols(y ~ sunab(treatment_year, year) | entity_id + year, data = df, cluster = ~state)
iplot(model, main = "Event Study")
```

### Stata

```stata
* Event study with reghdfe
reghdfe y lead4 lead3 lead2 lag0 lag1 lag2 lag3 lag4, ///
    absorb(entity_id year) cluster(state)

* Plot
coefplot, keep(lead* lag*) vertical yline(0) xline(4.5, lpattern(dash))
```

---

## §4 Staggered DiD / TWFE Issues

Runnable version: [`scripts/04_staggered_did.py`](../scripts/04_staggered_did.py).
Simulation scoring TWFE against a known truth:
[`scripts/simulations/sim_twfe_staggered_bias.py`](../scripts/simulations/sim_twfe_staggered_bias.py).

### Python (StatsPAI — all four repairs)

```python
import statspai as sp

# Callaway-Sant'Anna's convention: never-treated units get g = 0
df['first_treat'] = df['treatment_year'].fillna(0).astype(int)
df['treat'] = ((df['first_treat'] > 0) & (df['year'] >= df['first_treat'])).astype(int)

# 1. DIAGNOSE -- how much weight sits on "already-treated as control"?
bacon = sp.bacon_decomposition(data=df, y='y', treat='treat', time='year', id='unit')
comps = bacon['components']
by_type = (comps.assign(contrib=comps.estimate * comps.weight)
                .groupby('type').agg(weight=('weight', 'sum'), contrib=('contrib', 'sum')))
print(by_type)          # weights sum to 1; sum of contrib reproduces the TWFE coef

# 2. Callaway & Sant'Anna (2021) -- group-time ATTs with clean controls
cs      = sp.callaway_santanna(data=df, y='y', g='first_treat', t='year', i='unit',
                               control_group='nevertreated')
overall = sp.aggte(cs, type='simple',  bstrap=False)
dynamic = sp.aggte(cs, type='dynamic', bstrap=False)

# 3. Sun & Abraham (2021) -- interaction-weighted event study
sa = sp.sun_abraham(data=df, y='y', g='first_treat', t='year', i='unit',
                    control_group='nevertreated')

# 4. Borusyak-Jaravel-Spiess -- imputation estimator
bjs = sp.did_imputation(data=df, y='y', group='unit', time='year',
                        first_treat='first_treat')
```

On `castle.dta` the Bacon weights sum to 1.0000 and their weighted sum
reproduces the TWFE coefficient exactly (`+0.08181`), while Callaway-Sant'Anna
and Sun-Abraham both return `+0.110`. In the controlled simulation, with effects
that grow at 1.0 per period of exposure, the true ATT is 6.832: **TWFE reports
3.173 (−54%) while Callaway-Sant'Anna returns 6.878.**

### Bacon Decomposition (R)

```r
library(bacondecomp)

# Goodman-Bacon (2021) decomposition
bacon_out <- bacon(y ~ treatment, data = df, id_var = "entity_id", time_var = "year")
print(bacon_out)

# Weighted sum = TWFE estimate
# Shows which 2x2 comparisons drive the estimate
# Flags problematic "already-treated vs later-treated" comparisons
```

### Bacon Decomposition (Stata)

```stata
* Install: ssc install bacondecomp
bacondecomp y treatment, ddetail
```

### Callaway & Sant'Anna (R)

```r
library(did)

# Group-time ATT
att_gt <- att_gt(yname = "y", tname = "year", idname = "entity_id",
                 gname = "treatment_year", data = df)
summary(att_gt)
ggdid(att_gt)

# Aggregate to overall ATT
agg <- aggte(att_gt, type = "dynamic")
ggdid(agg)
```

---

## §5 Regression Discontinuity Design

Runnable version with the full robustness package:
[`scripts/05_rdd.py`](../scripts/05_rdd.py).

### Python (StatsPAI — this is the one to report)

```python
import statspai as sp

# 0. Is the design actually SHARP?  Check compliance before assuming it.
print(df.groupby(df.running_var >= cutoff)['treated'].mean())

# 1. MANIPULATION TEST FIRST.  If units sort across the cutoff, no bandwidth
#    choice repairs the design.  Native Python -- rdd::DCdensity is not needed.
print(sp.rddensity(data=df, x='running_var', c=cutoff).summary())   # Cattaneo-Jansson-Ma
print(sp.mccrary_test(data=df, x='running_var', c=cutoff).summary())  # original McCrary

# 2. Estimate: MSE-optimal bandwidth, triangular kernel, robust bias-corrected CI
rd = sp.rdrobust(data=df, y='y', x='running_var', c=cutoff)
print(rd.summary())
sp.rdplot(data=df, y='y', x='running_var', c=cutoff)

# 3. Robustness: bandwidth, polynomial order, placebo cutoffs, covariate balance
for p in (1, 2):                       # stop at 2 -- Gelman & Imbens (2019)
    print(p, float(sp.rdrobust(data=df, y='y', x='running_var', c=cutoff, p=p).estimate))
for c in (0.35, 0.40, 0.60, 0.65):     # placebos should be null
    print(c, float(sp.rdrobust(data=df, y='y', x='running_var', c=c).estimate))
for cov in predetermined_covariates:   # must be CONTINUOUS at the cutoff
    print(cov, float(sp.rdrobust(data=df, y=cov, x='running_var', c=cutoff).estimate))
```

> Put only **predetermined** variables in the balance table. The treatment
> indicator *should* jump at the cutoff — that is the design working, not an
> imbalance.
>
> `sp.rdrobust(h=...)` also changes how the bias-correction bandwidth `b` is
> selected, so a manual sweep will not reproduce the MSE-optimal run at its
> midpoint. Report the MSE-optimal estimate as the headline.

### Python (hand-rolled local linear — a benchmark, NOT the headline)

```python
import statsmodels.formula.api as smf

df['x_centered'] = df['running_var'] - cutoff
df['treated'] = (df['running_var'] >= cutoff).astype(int)

h = 10
for bw in [h/2, h, 2*h]:
    sub = df[abs(df['x_centered']) <= bw]
    m = smf.ols('y ~ treated * x_centered', data=sub).fit(cov_type='HC1')
    print(f"BW={bw:.1f}: effect={m.params['treated']:.3f} (SE={m.bse['treated']:.3f})")
```

This uses a **uniform kernel, no bias correction, and conventional standard
errors**. Calonico-Cattaneo-Titiunik showed the resulting confidence interval
undercovers: the MSE-optimal bandwidth deliberately trades bias against
variance, so first-order bias remains in the point estimate and the naive CI
does not account for it. On `lmb-data.dta` this route wanders from 21.6 to 17.7
across bandwidths while `sp.rdrobust` returns 18.45 with a robust CI.

### R

```r
library(rdrobust)

# Automatic bandwidth selection + local polynomial
rd_result <- rdrobust(y = df$y, x = df$running_var, c = cutoff)
summary(rd_result)

# Plot
rdplot(y = df$y, x = df$running_var, c = cutoff,
       title = "RD Plot", x.label = "Running Variable", y.label = "Outcome")
```

### Stata

```stata
* RD plot
rdplot y running_var, c(cutoff) graph_options(title("RD Plot"))

* RD estimate with rdrobust
rdrobust y running_var, c(cutoff) kernel(triangular) bwselect(mserd)
```

### McCrary Density Test

Use `rddensity` (Cattaneo-Jansson-Ma 2020) in **any** of the three languages —
it supersedes the original McCrary (2008) binned estimator and the long-stale
`rdd` package.

```python
import statspai as sp
sp.rddensity(data=df, x='running_var', c=cutoff)     # Python, native
```

```r
library(rddensity)
rddensity(X = df$running_var, c = cutoff)            # R
```

```stata
rddensity running_var, c(cutoff)                     /* Stata */
```

---

## §6 Instrumental Variables / 2SLS

### Python

Runnable version, including live demonstrations of both classic IV bugs:
[`scripts/06_iv_2sls.py`](../scripts/06_iv_2sls.py).

```python
import numpy as np
import statsmodels.formula.api as smf
import statspai as sp
from linearmodels.iv import IV2SLS

# --- Preferred: StatsPAI prints coefficient AND diagnostics in one call ----
iv = sp.ivreg('y ~ (endog_var ~ instrument1 + instrument2) + control1 + control2',
              data=df, cluster='cluster_var')
print(iv.summary())   # includes First-stage F, partial R2, Hausman p-value

# --- linearmodels, for the deepest IV diagnostic surface -------------------
results = IV2SLS.from_formula(
    'y ~ 1 + control1 + control2 + [endog_var ~ instrument1 + instrument2]',
    data=df
).fit(cov_type='clustered', clusters=df['cluster_var'])
print(results.summary)
print(results.wooldridge_overid)     # over-identification, when n_instruments > 1

# First-stage F.  NOTE: `.first_stage.diagnostics` is a DataFrame, so
# `diagnostics['f.stat']` is a Series -- formatting it with :.1f raises.
# Index the row too, or compute it directly:
fs = smf.ols('endog_var ~ instrument1 + instrument2 + control1 + control2',
             data=df).fit(cov_type='HC1')
fstat = float(np.squeeze(fs.f_test('instrument1 = instrument2 = 0').fvalue))
print(f'First-stage F on the excluded instruments: {fstat:.2f}')
```

**Two thresholds, not one.** `F > 10` (Staiger-Stock) bounds *relative bias*.
Lee, McCrary, Moreira & Porter (2022) show that a valid 5% t-test needs a
first-stage F above roughly **104.7**, or a tF-adjusted critical value. Card
(1995) with `nearc4` has F ≈ 17.5: the point estimate is usable, a conventional
t-test over-rejects. Say which criterion you are invoking.

### Two IV bugs that survive code review

**1. Manual two-step 2SLS gives the wrong standard error.** Regressing `D` on
`Z`, then plugging the fitted values into a second OLS, reproduces the 2SLS
*point estimate exactly* — which is what makes it convincing — and then computes
residuals against `D̂` instead of `D`. On Card (1995) with classical SEs the
manual route is 2.6% off. The direction depends on the data, so you cannot even
sign your own mistake. **Never hand-roll 2SLS to obtain inference.**

**2. The forbidden regression.** When `D` is binary it is tempting to run a
logit/probit first stage and plug the predicted probability in as a regressor:

```python
# WRONG -- inconsistent unless the logit is exactly the right model
probit = smf.logit('D ~ Z + X', data=df).fit()
df['D_hat'] = probit.predict()
smf.ols('y ~ D_hat + X', data=df).fit()          # forbidden regression

# CORRECT -- plain linear 2SLS
IV2SLS.from_formula('y ~ 1 + X + [D ~ Z]', data=df).fit(cov_type='robust')

# ALSO CORRECT (Wooldridge 2010, procedure 21.1) -- use the nonlinear fitted
# value as an INSTRUMENT, never as a regressor.  Consistent for any first-stage
# functional form.  Check its own first-stage F before trusting it.
df['z_hat'] = probit.predict()
IV2SLS.from_formula('y ~ 1 + X + [D ~ z_hat]', data=df).fit(cov_type='robust')
```

On Card (1995) with `D = 1{educ ≥ 13}`, the forbidden version returns **−0.070**
where consistent linear 2SLS returns **+0.785** — different magnitude and
different sign, with no warning from the standard error.

### R

```r
library(AER)

# 2SLS
model <- ivreg(y ~ control1 + control2 + endog_var | control1 + control2 + instrument1 + instrument2,
               data = df)
summary(model, diagnostics = TRUE)  # Includes weak instrument test, Wu-Hausman, Sargan
```

### Stata

```stata
* 2SLS
ivregress 2sls y control1 control2 (endog_var = instrument1 instrument2), first robust

* Post-estimation diagnostics
estat firststage    /* First-stage F */
estat overid        /* Sargan-Hansen test */
estat endogenous    /* Wu-Hausman */
```

---

## §7 Synthetic Control

Runnable version: [`scripts/07_synthetic_control.py`](../scripts/07_synthetic_control.py).

> **The "Python has no synthetic control, use rpy2" advice is obsolete.**
> Native options: `StatsPAI` (`sp.synth`, `sp.sdid`, plus placebo/RMSPE/LOO
> machinery), `scpi` (Cattaneo, Feng, Palomba & Titiunik — the method's own
> authors, with prediction intervals), and `pysyncon`. Keep `rpy2` as a
> fallback for a referee who asks specifically for `Synth`'s output.

### Python (StatsPAI)

```python
import statspai as sp

sc = sp.synth(data=df, outcome='y', unit='unit_id', time='year',
              treated_unit=TREATED, treatment_time=T0,
              placebo=True)      # placebo distribution computed alongside
print(sc.summary())

sc.estimate      # ATT (post-treatment average gap)
sc.pvalue        # PERMUTATION p-value -- this is the inference
sc.model_info['weights']            # (unit_id, weight) pairs -- NOT sc.params
sc.model_info['pre_treatment_rmse'] # the credibility statistic

sd = sp.sdid(data=df, outcome='y', unit='unit_id', time='year',
             treated_unit=TREATED, treatment_time=T0)   # synthetic DiD
```

> **Performance:** passing `covariates=[...]` triggers the nested V-weight
> optimisation. On a 51-unit × 16-year panel that is >2 minutes versus ~5 s
> without. Pre-treatment outcome lags are the classic Abadie-Diamond-Hainmueller
> predictor set — start there.
>
> **`sc.params` is the ATT, not the donor weights.** The weights live at
> `model_info['weights']` as an `(n, 2)` array of `(unit_id, weight)` pairs.
>
> **With one treated unit there is no sampling distribution.** On `texas.dta`,
> `estimate / se` is **7.74** while the permutation p-value is **0.18**. Quoting
> the t-ratio would be a serious over-claim. The permutation rank *is* the test,
> and with ~50 donors its resolution is about 0.02 — do not report more digits.

### Python (rpy2 fallback)

```python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
ro.globalenv['df'] = df
ro.r('''
library(Synth)
dataprep.out <- dataprep(
    foo = df, predictors = c("predictor1", "predictor2"), predictors.op = "mean",
    dependent = "outcome", unit.variable = "unit_id", time.variable = "year",
    treatment.identifier = treated_unit, controls.identifier = control_units,
    time.predictors.prior = pre_period, time.optimize.ssr = pre_period,
    time.plot = full_period)
synth.out <- synth(dataprep.out)
synth.tab(dataprep.res = dataprep.out, synth.res = synth.out)
''')
```

### R (Native)

```r
library(Synth)

dataprep.out <- dataprep(
    foo = df,
    predictors = c("gdp", "trade", "infrate"),
    predictors.op = "mean",
    dependent = "outcome",
    unit.variable = "unit_id",
    time.variable = "year",
    treatment.identifier = 1,
    controls.identifier = c(2:10),
    time.predictors.prior = 1980:1990,
    time.optimize.ssr = 1980:1990,
    time.plot = 1980:2000
)

synth.out <- synth(dataprep.out)
path.plot(synth.res = synth.out, dataprep.res = dataprep.out)
gaps.plot(synth.res = synth.out, dataprep.res = dataprep.out)

# Placebo tests (permutation)
library(SCtools)
placebo <- generate.placebos(dataprep.out, synth.out, Sigf.ipop = 5)
plot_placebos(placebo)
mspe.plot(placebo, discard.extreme = TRUE, mspe.limit = 20)
```

### Stata

```stata
* Synthetic control
synth outcome predictor1 predictor2 outcome(1980) outcome(1985), ///
    trunit(1) trperiod(1990) figure
```

---

## §8 Matching / PSM / IPW / CEM

Runnable version, scored against the NSW experimental benchmark:
[`scripts/08_matching_ipw.py`](../scripts/08_matching_ipw.py).

### Python (Propensity Score + IPW)

```python
import numpy as np
import statsmodels.formula.api as smf
import statspai as sp

# Step 1: propensity score
logit = smf.logit('treated ~ x1 + x2 + x3', data=df).fit()
df['pscore'] = logit.predict()

# Step 2: CHECK OVERLAP BEFORE WEIGHTING.  This is a precondition, not a
# robustness check -- no weighting scheme rescues a design without common
# support, it only hides the extrapolation.
lo, hi = df.loc[df.treated == 1, 'pscore'].agg(['min', 'max'])
inside = df.pscore.between(lo, hi)
print(f'controls on treated support: {(inside & (df.treated == 0)).sum():,}'
      f' of {(df.treated == 0).sum():,}')

# Step 3: weights, and look at the largest one before you use them
df['ipw'] = np.where(df.treated == 1, 1 / df.pscore, 1 / (1 - df.pscore))
print('max weight:', df.ipw.max())     # untrimmed, a pscore of 0.999 gives 1000

# Step 4: DROP off-support units (Crump et al. 2009), then estimate
kept = df[df.pscore.between(0.1, 0.9)]
res  = smf.wls('y ~ treated', data=kept, weights=kept.ipw).fit(cov_type='HC1')

# StatsPAI equivalent, with Hajek weights and bootstrap SEs
r = sp.ipw(data=df, y='y', treat='treated', covariates=['x1', 'x2', 'x3'],
           estimand='ATT', trim=0.1, n_bootstrap=500, seed=0)

# Balance is the ONLY evidence weighting worked
print(sp.balance_diagnostics(data=kept, treatment='treated',
                             covariates=['x1', 'x2', 'x3'],
                             weights=kept.ipw.to_numpy()).summary())
sp.love_plot(data=kept, treatment='treated', covariates=['x1', 'x2', 'x3'])
```

> **"Trim" is not a standardised word — verify what your library does.**
> Some implementations *drop* units outside `[α, 1−α]`; others *clip* the
> propensity score and keep every row. On NSW/CPS the difference is
> **+$1,454 vs −$7,936** against a known truth of **+$1,794**. Verified for
> StatsPAI 1.21: `sp.ipw(trim=α)` **clips**. Check `model_info['n_obs']` and
> `model_info['pscore_min']` to see which one you got.

### Python (Abadie-Imbens bias-corrected matching)

With more than one continuous matching variable, exact matches do not exist, the
match discrepancy shrinks too slowly, and the resulting bias **does not vanish
asymptotically**. The correction regression-adjusts each matched pair for the
residual covariate gap:

```python
import numpy as np, statsmodels.api as sm
from scipy.spatial import cKDTree

treated, control = df[df.treated == 1], df[df.treated == 0]
scale = df[COVS].std(ddof=0).replace(0, 1.0)          # diagonal-Mahalanobis metric
_, idx = cKDTree((control[COVS] / scale).to_numpy()).query(
    (treated[COVS] / scale).to_numpy(), k=1)
idx = idx.reshape(-1, 1)

yt, yc = treated[Y].to_numpy(), control[Y].to_numpy()
naive = np.mean(yt - yc[idx].mean(axis=1))

# mu0(x): outcome model estimated on CONTROLS only
Xc = sm.add_constant(control[COVS].to_numpy())
mu0 = sm.OLS(yc, Xc).fit()
pred_t = mu0.predict(sm.add_constant(treated[COVS].to_numpy(), has_constant='add'))
corrected = np.mean(yt - yc[idx].mean(axis=1) - (pred_t - mu0.predict(Xc)[idx].mean(axis=1)))
```

> **Matching standard errors are not OLS standard errors.** Abadie & Imbens
> (2006, 2011): the naive post-match regression SE is wrong because matching is
> not a sampling scheme OLS knows about. Use the Abadie-Imbens variance, or a
> bootstrap designed for matching — and note that the ordinary nonparametric
> bootstrap is *also* invalid for nearest-neighbour matching (Abadie & Imbens 2008).
>
> **Prefer weighting or covariate matching over propensity-score matching.**
> King & Nielsen (2019): PSM approximates a completely randomised experiment
> rather than a blocked one, so pruning on the score can *increase* imbalance.

### R (MatchIt)

```r
library(MatchIt)   # Zelig is archived from CRAN -- do not use it

m.out <- matchit(treated ~ x1 + x2 + x3, data = df, method = "nearest", ratio = 1)
summary(m.out)
plot(m.out, type = "jitter")

m.data <- match.data(m.out)
# Post-match SEs: use a matching-aware variance, not lm()'s default
model <- lm(y ~ treated + x1 + x2 + x3, data = m.data, weights = weights)
lmtest::coeftest(model, vcov = sandwich::vcovCL, cluster = ~ subclass)
```

### R (IPW)

```r
library(ipw)

# IPW weights
temp <- ipwpoint(
    exposure = treated,
    family = "binomial",
    link = "logit",
    numerator = ~ 1,
    denominator = ~ x1 + x2 + x3,
    data = df
)
df$ipw <- temp$ipw.weights

# Weighted model
library(survey)
design <- svydesign(ids = ~1, weights = ~ipw, data = df)
model <- svyglm(y ~ treated, design = design)
```

### Stata (CEM + teffects)

```stata
* Coarsened Exact Matching
cem x1 (#5) x2 (#3) x3, treatment(treated)
reg y treated [iweight = cem_weights]

* Propensity Score Matching via teffects
teffects psmatch (y) (treated x1 x2 x3), atet

* IPW
teffects ipw (y) (treated x1 x2 x3), atet
```

---

## §9 DAGs and Collider Bias

### Conceptual Framework

```python
# DAGs are primarily conceptual tools
# Use dagitty.net for interactive DAG drawing

# Key rules from Mixtape:
# 1. Condition on confounders (common causes of treatment and outcome)
# 2. Never condition on colliders (common effects of treatment and outcome)
# 3. Never condition on mediators (if estimating total effect)
# 4. Backdoor criterion: block all backdoor paths from treatment to outcome
```

### R (ggdag)

```r
library(ggdag)
library(dagitty)

# Define DAG
dag <- dagitty('dag {
    X -> Y
    Z -> X
    Z -> Y
    M -> X
    M -> Y
}')

# Identify adjustment sets
adjustmentSets(dag, exposure = "X", outcome = "Y")

# Plot
ggdag(dag) + theme_dag()
```

---

## §10 Randomization Inference

### Python

Runnable version: [`scripts/09_randomization_inference.py`](../scripts/09_randomization_inference.py).

RI tests the **Fisher sharp null** `Y_i(1) = Y_i(0)` for every unit — a
different null from Neyman's `E[Y(1) − Y(0)] = 0`. Under the sharp null all
potential outcomes are known, so the randomisation distribution is exact: valid
in small samples, no asymptotics, no cluster-count requirement. That last point
is why RI is the right tool exactly where cluster-robust SEs fail.

```python
import numpy as np

def permutation_test(treatment, outcome, n_permutations=10_000, clusters=None, seed=0):
    """Two-sided RI p-value for the difference in means under the sharp null."""
    rng = np.random.default_rng(seed)
    observed = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()

    draws = np.empty(n_permutations)
    if clusters is None:
        for b in range(n_permutations):
            perm = rng.permutation(treatment)
            draws[b] = outcome[perm == 1].mean() - outcome[perm == 0].mean()
    else:
        # PERMUTE AT THE LEVEL TREATMENT WAS ASSIGNED.  Permuting individuals
        # when groups were assigned understates the null variance and
        # manufactures significance.
        uniq = np.unique(clusters)
        cl_treat = np.array([treatment[clusters == g][0] for g in uniq])
        for b in range(n_permutations):
            mapping = dict(zip(uniq, rng.permutation(cl_treat)))
            perm = np.array([mapping[g] for g in clusters])
            draws[b] = outcome[perm == 1].mean() - outcome[perm == 0].mean()

    # The +1 counts the OBSERVED assignment, which is itself one of the possible
    # randomisations.  Without it the formula can return exactly 0 -- a p-value
    # no permutation test can produce.  Smallest attainable value is 1/(1+B).
    p = (1 + np.sum(np.abs(draws) >= np.abs(observed))) / (1 + n_permutations)
    return float(observed), float(p), draws

obs, p, draws = permutation_test(df['treated'].to_numpy(), df['y'].to_numpy())
print(f"Observed difference: {obs:.4f}, RI p-value: {p:.5f}")
```

```python
import statspai as sp
sp.ri_test(data=df, y='y', treat='treated', stat='diff_means', n_perms=10_000, seed=0)
```

> Verified for StatsPAI 1.21: `sp.ri_test` omits the `+1` and can return
> `p_value = 0.0` exactly. Report `p < 1/(1+B)` rather than `p = 0.000`.

### R

```r
library(ri2)

# Declare randomization procedure
declaration <- declare_ra(N = nrow(df), m = sum(df$treated))

# Conduct randomization inference
ri_out <- conduct_ri(
    y ~ treated,
    declaration = declaration,
    sharp_hypothesis = 0,
    data = df
)
summary(ri_out)
plot(ri_out)
```

### Stata

```stata
* Randomization inference
ritest treated _b[treated], reps(1000): reg y treated
```

---

## §11. Diff-in-Discontinuities (Grembi-Nannicini-Troiano 2016)

Use when an RDD cutoff existed *before* a policy change and you want to isolate the policy's marginal effect at the cutoff. Bennedsen et al. (2022) uses this at Denmark's 35-employee wage-transparency threshold.

### Python

```python
# Diff-in-disc: estimate RDD before policy, after policy, and difference
from rdrobust import rdrobust

# Pre-period RDD
pre = df[df['post'] == 0]
pre_res = rdrobust(y=pre['y'], x=pre['running_var'], c=cutoff, h=bandwidth, p=1)

# Post-period RDD
post = df[df['post'] == 1]
post_res = rdrobust(y=post['y'], x=post['running_var'], c=cutoff, h=bandwidth, p=1)

# Diff-in-disc = post_res.coef - pre_res.coef (test via bootstrap)
```

### R

```r
library(rdrobust)

# Pooled: Y on post × above_cutoff + controls
library(fixest)
feols(y ~ post * above_cutoff + bs(running_var, df=4)
           + post:bs(running_var, df=4),
      cluster = ~ id, data = df)

# Or two separate RDDs and difference
pre_rd  <- rdrobust(df$y[df$post==0], df$running[df$post==0], c=35, h=15)
post_rd <- rdrobust(df$y[df$post==1], df$running[df$post==1], c=35, h=15)
diff    <- post_rd$coef[1] - pre_rd$coef[1]
```

### Stata

```stata
* Pooled diff-in-disc with local linear polynomial
reg y c.post##c.above_cutoff##c.running_dev i.(covs) ///
    if abs(running_dev) <= 15, vce(cluster firmid)

* Or separate RDDs
rdrobust y running_var if post==0, c(35) h(15) p(1)
estimates store pre
rdrobust y running_var if post==1, c(35) h(15) p(1)
estimates store post
```

**Robustness**: sweep placebo cutoffs (Bennedsen swept 15–100 employees excluding 20–50).

---

## §12. Weather/Shift-Share IV for Cash-Flow Shocks

Template from Brown et al. (2021): county-level abnormal snow cover as IV for firm annual cash flow.

### Python

```python
import numpy as np
import pandas as pd
import statspai as sp

# Construct the abnormal-snow IV: deviation from the county's own trailing
# 10-year Q1 mean.  shift(1) is what makes it a LEADING-EDGE deviation rather
# than one that includes the current observation -- drop it and the instrument
# is mechanically correlated with its own denominator.
df = df.sort_values(['county', 'year'])
df['abn_snow_q1'] = df.groupby('county')['snow_q1'].transform(
    lambda s: s - s.shift(1).rolling(10, min_periods=5).mean()
)

# First stage -- absorb FE rather than expanding them as dummies
first = sp.feols(
    'cash_flow ~ abn_snow_q1 + fixed_assets_lag + size_lag + age_lag'
    ' | naics4^yq + county',
    data=df, vcov={'CRV1': 'naics4'},
)
print(first.summary())

# 2SLS with the same absorbed FE
iv = sp.feols(
    'delta_credit_draw ~ fixed_assets_lag + size_lag + age_lag'
    ' | naics4^yq + county | cash_flow ~ abn_snow_q1',
    data=df, vcov={'CRV1': 'naics4'},
)
print(iv.summary())
```

### R

```r
library(fixest)

# First stage
feols(cash_flow ~ abn_snow_q1 + fixed_assets_lag + size_lag + age_lag
      | naics4^yq + county,
      cluster = ~ naics4, data = df)

# 2SLS (fixest IV syntax)
feols(delta_credit_draw ~ fixed_assets_lag + size_lag + age_lag
      | naics4^yq + county
      | cash_flow ~ abn_snow_q1,
      cluster = ~ naics4, data = df) |> summary(stage = 1:2)
```

### Stata

```stata
* Construct abnormal snow IV: deviation from the trailing 10-year county mean.
* Use tsset + lag operators -- an unrolled sum is unreadable and error-prone.
xtset county year
gen snow_ma10 = (L1.snow_q1 + L2.snow_q1 + L3.snow_q1 + L4.snow_q1 + L5.snow_q1 ///
               + L6.snow_q1 + L7.snow_q1 + L8.snow_q1 + L9.snow_q1 + L10.snow_q1) / 10
gen abn_snow_q1 = snow_q1 - snow_ma10

* Or, tolerating incomplete windows (ssc install rangestat):
* rangestat (mean) snow_q1, interval(year -10 -1) by(county)
* gen abn_snow_q1 = snow_q1 - snow_q1_mean

* First-stage check
reghdfe cash_flow abn_snow_q1 fixed_assets_lag size_lag age_lag, ///
    absorb(naics4#yq county) vce(cluster naics4)

* 2SLS
ivreghdfe delta_credit_draw (cash_flow = abn_snow_q1) ///
    fixed_assets_lag size_lag age_lag, ///
    absorb(naics4#yq county) cluster(naics4) first
```

**First-stage F target**: > 10 (Stock-Yogo) or > 104.7 (Lee-McCrary-Moreira-Porter 2022 correction for t-ratio inference).

---

## §13. Saturated Interacted FE (Within-Unit-Time Identification)

Template from Kempf-Tsoutsoura (2021): identify analyst-level effect by absorbing firm × quarter + agency × quarter.

### Python

```python
# --- Preferred: pyfixest / StatsPAI handle interacted FE directly ---------
import pyfixest as pf
pf.feols('rating_adj ~ misaligned | firm^quarter + agency^quarter + analyst',
         data=df, vcov={'CRV1': 'analyst'})

import statspai as sp
sp.feols('rating_adj ~ misaligned | firm^quarter + agency^quarter + analyst',
         data=df, vcov={'CRV1': 'analyst'})

# --- pyhdfe, when you need the residualised matrices themselves ----------
import pyhdfe
import statsmodels.api as sm

# TRAP: drop_singletons=True (the default) REMOVES ROWS.  residualize() then
# returns fewer rows than df, so any cluster/weight vector taken from the
# original frame is misaligned -- silently, by however many singletons existed.
# Drop the singletons from the frame FIRST so one index governs everything.
FE = ['firm_quarter_id', 'agency_quarter_id', 'analyst_id']
algo = pyhdfe.create(df[FE].to_numpy(), drop_singletons=False)

X_dm = algo.residualize(df[['misaligned']].to_numpy())
y_dm = algo.residualize(df[['rating_adj']].to_numpy())
assert len(X_dm) == len(df), 'row counts must match before clustering'

res = sm.OLS(y_dm, X_dm).fit(
    cov_type='cluster', cov_kwds={'groups': df['analyst_id']}
)
```

### R

```r
library(fixest)

# Triple-interacted FE: firm × quarter + agency × quarter + analyst
feols(rating_adj ~ misaligned | firm^quarter + agency^quarter + analyst,
      cluster = ~ analyst + firm, data = df)

# For employer × county × period (Meeuwis-style)
feols(delta_equity ~ republican:post2016
      | employer^county^period + household,
      cluster = ~ zip + employer, data = df)
```

### Stata

```stata
* reghdfe handles arbitrarily high-dimensional FE efficiently
reghdfe rating_adj misaligned, ///
    absorb(firm#quarter agency#quarter analyst) ///
    vce(cluster analyst firm)

* Triple-interaction (Meeuwis)
reghdfe delta_equity c.republican##c.post2016, ///
    absorb(employer#county#period household) ///
    vce(cluster zip employer)
```

**Rule**: cluster at treatment-assignment level (analyst, household), NOT at the highest FE level.

---

## §14. Event Study — Election / FOMC / Platform Shock

Template for financial event studies with asymmetric information release.

### Python

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def compute_car(panel, event_dates, factors, window=(-10, 10),
                est_window=(-250, -21), model_cols=('mktrf', 'smb', 'hml')):
    """Cumulative abnormal returns around firm-specific event dates.

    Parameters
    ----------
    panel : DataFrame with columns [permno, date, ret]  (daily returns)
    event_dates : DataFrame with columns [permno, event_date]
    factors : DataFrame with columns [date, rf, *model_cols]
    est_window, window : (start, end) in TRADING DAYS relative to the event

    Returns one row per (permno, event) with the CAR and the estimation-window
    fit statistics you need in order to drop badly estimated firms.
    """
    df = panel.merge(factors, on='date', how='inner').sort_values(['permno', 'date'])
    df['exret'] = df['ret'] - df['rf']

    out = []
    for permno, ev in event_dates.itertuples(index=False):
        g = df[df['permno'] == permno].reset_index(drop=True)
        if g.empty:
            continue
        pos = g.index[g['date'] >= ev]
        if len(pos) == 0:
            continue
        t0 = int(pos[0])                                   # first trading day >= event

        est = g.iloc[max(0, t0 + est_window[0]): t0 + est_window[1] + 1]
        evt = g.iloc[max(0, t0 + window[0]): t0 + window[1] + 1]
        # Require a usable estimation window AND a complete event window --
        # silently short windows are the classic source of fake CARs.
        if len(est) < 100 or len(evt) < (window[1] - window[0] + 1):
            continue

        fit = sm.OLS(est['exret'], sm.add_constant(est[list(model_cols)])).fit()
        expected = fit.predict(sm.add_constant(evt[list(model_cols)], has_constant='add'))
        ar = evt['exret'].to_numpy() - expected.to_numpy()

        out.append({
            'permno': permno, 'event_date': ev,
            'car': float(ar.sum()),
            'car_se': float(np.sqrt(len(ar)) * fit.mse_resid ** 0.5),  # iid-AR benchmark
            'n_est': len(est), 'r2_est': float(fit.rsquared),
        })
    return pd.DataFrame(out)


# Cross-sectional test: CAR on firm characteristics
res = smf.ols('car ~ emission_intensity + log_size + log_bm',
              data=event_df).fit(cov_type='HC1')
```

> **`car_se` above is the iid benchmark only.** With clustered event dates
> (an FOMC announcement, an election) abnormal returns are cross-sectionally
> correlated and that SE is far too small. Cluster on the event date, or use
> calendar-time portfolios. Overlapping event windows require the same care.

### R

```r
library(eventstudies)

# Event-study abnormal returns around 2016-11-09
es <- eventstudy(
    firm.returns    = returns_matrix,
    event.list      = data.frame(name=c("AAPL","MSFT"), when=as.Date("2016-11-09")),
    event.window    = 10,
    type            = "marketModel",
    to.remap        = TRUE,
    remap           = "cumsum",
    inference       = TRUE,
    inference.strategy = "bootstrap"
)
plot(es)
```

### Stata

```stata
* eventstudy2 package
eventstudy2 permno date using event_dates.dta, ///
    returns(ret) modeltype(FFM) estwin(-250 -20) eventwin(-10 10) ///
    alpha market_ret smb hml

* Cross-sectional test
reg car_10d emission_intensity log_size log_bm, robust
```

**Key robustness**: (1) placebo non-event dates, (2) no overlapping corporate actions, (3) multiple comparison correction if scanning many events, (4) sign-flip test for asymmetric effects.
