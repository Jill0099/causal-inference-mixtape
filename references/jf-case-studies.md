# End-to-End Finance Case Studies

Five worked examples from JF 2021-2024. Each has: identification narrative → Stata + Python + R code skeleton → robustness triad → what to report in the paper.

---

## Case 1 — Bennedsen 2022: Diff-in-Differences + DDD + Diff-in-Discontinuities

**Setting**: Denmark's Act 562 (2006) requires firms with ≥35 employees to publish gender-disaggregated wages.

### Stata
```stata
* DiD (baseline)
use danish_ida.dta, clear
reghdfe log_wage c.treated##c.post, ///
    absorb(i.pid#i.firmid year) ///
    vce(cluster pid firmid)

* DDD with gender
reghdfe log_wage c.treated##c.post##c.male, ///
    absorb(i.pid#i.firmid year) ///
    vce(cluster pid firmid)

* Event study (pre-trends + dynamic effect).
* rel_k is indexed so that rel_2 is k = -1, the OMITTED reference period.
* Omitting k = 0 instead -- an easy off-by-one -- normalises the treatment
* period itself and makes every post coefficient a difference from impact.
forvalues k = -3/2 {
    local j = `k' + 3
    gen rel_`j' = treated * (year == 2006 + `k')   // rel_0=k-3 ... rel_5=k+2
}
reghdfe log_wage rel_0 rel_1 rel_3 rel_4 rel_5, /// rel_2 (k=-1) omitted
    absorb(i.pid#i.firmid year) vce(cluster firmid)

* Placebo thresholds (sweep 15..100 excluding 20..50)
foreach cut of numlist 15 55 60 65 70 75 80 85 90 95 100 {
    gen treat_`cut' = (emp_2005 >= `cut' & emp_2005 <= `cut' + 15)
    reghdfe log_wage c.treat_`cut'##c.post, absorb(i.pid#i.firmid year)
    drop treat_`cut'
}

* Diff-in-discontinuities (Grembi-Nannicini-Troiano 2016)
rdrobust log_wage emp_2005, c(35) h(15) p(1) covs(i.post##c.post)
```

### Python
```python
import pandas as pd, numpy as np
import statsmodels.formula.api as smf

df = pd.read_parquet('danish_ida.parquet')

# DiD with two-way FE.  On an employer-employee panel the C(pid):C(firmid)
# dummy expansion is astronomically large -- use an HDFE backend, not patsy.
import statspai as sp

df['treat_post'] = df['treated'] * df['post']
res = sp.feols('log_wage ~ treat_post | pid^firmid + year',
               data=df, vcov={'CRV1': 'firmid'})

# Event study.  Note range(-3, 3) already contains -1, so the reference period
# must be excluded from BOTH the dummy list and the formula -- which the
# comprehension below does by filtering on k != -1 in one place only.
rel_periods = [k for k in range(-3, 3) if k != -1]
for k in rel_periods:
    df[f'rel_{k}'] = df['treated'] * (df['year'] == 2006 + k).astype(int)
ev_res = sp.feols(
    'log_wage ~ ' + ' + '.join(f'rel_{k}' for k in rel_periods) + ' | pid^firmid + year',
    data=df, vcov={'CRV1': 'firmid'},
)

# Or in one call, letting StatsPAI build and bin the relative-time indicators:
es = sp.event_study(data=df, y='log_wage', treat_time='first_treat', time='year',
                    unit='pid', window=(-3, 2), ref_period=-1, cluster='firmid')
```

### R
```r
library(fixest)
library(rdrobust)

# DiD
feols(log_wage ~ treated:post | pid^firmid + year,
      cluster = ~ pid + firmid, data = df)

# DDD
feols(log_wage ~ treated:post + male:post + treated:post:male | pid^firmid + year,
      cluster = ~ pid + firmid, data = df)

# Event study
feols(log_wage ~ i(year, treated, ref = 2005) | pid^firmid + year,
      cluster = ~ firmid, data = df) |> iplot()

# Diff-in-disc: two RDDs, before and after the policy, then difference them.
# (`covs = model.matrix(~ post*post, df)` is a typo -- post*post collapses to
#  post, and passing the post indicator as a covariate does not difference
#  anything.  Estimate the two regimes separately.)
pre  <- rdrobust(df$log_wage[df$post == 0], df$emp_2005[df$post == 0], c = 35, h = 15)
post <- rdrobust(df$log_wage[df$post == 1], df$emp_2005[df$post == 1], c = 35, h = 15)
diff <- post$coef[1] - pre$coef[1]
# SE of the difference: the two samples are disjoint, so variances add
se   <- sqrt(post$se[1]^2 + pre$se[1]^2)

# Or pooled, which gives the interaction its own standard error directly:
library(fixest)
feols(log_wage ~ post * above_cutoff * running_dev, cluster = ~ firmid,
      data = subset(df, abs(running_dev) <= 15))
```

### What to report
- Table 2: Baseline DiD with 3 specs (no FE → firm FE → individual×firm FE + year)
- Figure 2: Event-study dynamics (flat pre-trends, sharp 2006 break)
- Table 4: DDD decomposition (male vs. female coefficients)
- Table 5: Placebo thresholds (sweep grid, all insignificant)
- Appendix: diff-in-disc estimate as sanity check

---

## Case 2 — Kempf 2021: Saturated Interacted FE + Election Event Study

**Setting**: Do credit analysts' political affiliations bias their ratings? Treatment = analyst's party ≠ sitting president's party.

### Stata
```stata
use analyst_ratings.dta, clear

* Saturated FE — firm × quarter, agency × quarter, analyst FE
reghdfe rating_adj misaligned, ///
    absorb(firmid#yq agency#yq analystid) ///
    vce(cluster analystid firmid)

* 2016 election event study
gen post2016 = (yq >= tq(2016q4))
reghdfe rating_adj c.misaligned##c.post2016, ///
    absorb(firmid#yq agency#yq analystid) ///
    vce(cluster analystid)

* Heterogeneity by cyclicality (placebo in low-cycle sector)
reghdfe rating_adj c.misaligned##c.high_cyclicality, ///
    absorb(firmid#yq agency#yq analystid) vce(cluster analystid)
```

### Python
```python
# Simplest correct route -- interacted FE syntax, no manual demeaning
import statspai as sp
sp.feols('rating_adj ~ misaligned | firm^yq + agency^yq + analyst',
         data=df, vcov={'CRV1': 'analyst'})

# pyhdfe route, when you want the residualised matrices.
# TRAP: pyhdfe.create defaults to drop_singletons=True, which REMOVES ROWS.
# df['analyst_id'] would then be longer than the residualised arrays and the
# clustering silently misaligns.  Turn it off, or drop singletons from the
# frame first so a single index governs both.
import pyhdfe
import statsmodels.api as sm

algo = pyhdfe.create(df[['firm_yq_id', 'agency_yq_id', 'analyst_id']].to_numpy(),
                     drop_singletons=False)
X_demeaned = algo.residualize(df[['misaligned']].to_numpy())
y_demeaned = algo.residualize(df[['rating_adj']].to_numpy())
assert len(X_demeaned) == len(df)

res = sm.OLS(y_demeaned, X_demeaned).fit(
    cov_type='cluster', cov_kwds={'groups': df['analyst_id']}
)
```

### R
```r
library(fixest)
feols(rating_adj ~ misaligned | firm^quarter + agency^quarter + analyst,
      cluster = ~ analyst + firm, data = df)

# Heterogeneity
feols(rating_adj ~ misaligned*cyclicality | firm^quarter + agency^quarter + analyst,
      cluster = ~ analyst, data = df)
```

### What to report
- Table 3: Main effect with progressive FE saturation (none → firm × quarter → + agency × quarter → + analyst)
- Table 5: 2016 election event-study DiD
- Table 6: Heterogeneity by Gallup polarization index
- Table 8: Market response tests (bond yields do not react → bias is not priced)

---

## Case 3 — Brown 2021: Weather-Shock IV

**Setting**: Instrument firm cash flow with abnormal Q1 snow cover.

### Stata
```stata
use y14q_firm_weather.dta, clear

* First stage
reghdfe cash_flow abn_snow_q1 fixed_assets_lag size_lag age_lag, ///
    absorb(naics4#yq county) vce(cluster naics4)
* ^ F-stat should be > 10

* IV: credit line drawdown on cash flow
ivreghdfe delta_credit_draw (cash_flow = abn_snow_q1) ///
    fixed_assets_lag size_lag age_lag, ///
    absorb(naics4#yq county) cluster(naics4)

* Reduced form
reghdfe delta_credit_draw abn_snow_q1 fixed_assets_lag, ///
    absorb(naics4#yq county) vce(cluster naics4)

* Alternative IV: P95 snow cover
ivreghdfe delta_credit_draw (cash_flow = abn_snow_p95), ///
    absorb(naics4#yq county) cluster(naics4)

* Covariate-balance across IV terciles
forvalues t = 1/3 {
    summarize size_lag age_lag lev_lag if snow_tercile == `t'
}
```

### Python
```python
import statspai as sp

# IV with high-dimensional FE.  Putting C(naics4_yq) + C(county) in a patsy
# formula materialises thousands of dummy columns; use fixest-style absorption.
first = sp.feols(
    'cash_flow ~ abn_snow_q1 + fixed_assets_lag + size_lag + age_lag'
    ' | naics4^yq + county',
    data=df, vcov={'CRV1': 'naics4'},
)
print(first.summary())    # report this table; F on abn_snow_q1 should exceed 10

iv = sp.feols(
    'delta_credit_draw ~ fixed_assets_lag + size_lag + age_lag'
    ' | naics4^yq + county | cash_flow ~ abn_snow_q1',
    data=df, vcov={'CRV1': 'naics4'},
)
print(iv.summary())

# Reduced form -- always show it alongside; with one instrument,
# reduced form / first stage reproduces the 2SLS coefficient exactly.
rf = sp.feols(
    'delta_credit_draw ~ abn_snow_q1 + fixed_assets_lag | naics4^yq + county',
    data=df, vcov={'CRV1': 'naics4'},
)
```

### R
```r
library(fixest)

# First stage
feols(cash_flow ~ abn_snow_q1 + fixed_assets_lag | naics4^yq + county,
      cluster = ~ naics4, data = df)

# 2SLS (fixest)
feols(delta_credit_draw ~ fixed_assets_lag | naics4^yq + county
      | cash_flow ~ abn_snow_q1,
      cluster = ~ naics4, data = df)
# Check first-stage F in summary
```

### What to report
- Table 3: First-stage (F ≈ 20, coef on abn_snow_q1)
- Table 4: 2SLS second-stage (credit line drawdown, line size, rate, maturity)
- Table 5: Industry partition (outdoor/transport show effect; services placebo flat)
- Appendix: reduced form + alternative IV (P95 snow)

---

## Case 4 — Barber 2022: Sharp RDD + Outage Natural Experiment

**Setting**: $300M market-cap cutoff for Robinhood's Top Movers list.

### Stata
```stata
use robinhood_daily.dta, clear

* Sharp RDD
rdrobust buy_users marketcap, c(300) h(50) p(1) ///
    covs(abs_ret vol_percentile) vce(cluster permno date)

* Outage DiD: daily attention-herding × outage-day indicator
reghdfe car_20d herd_today c.herd_today#c.outage_today, ///
    absorb(date) vce(cluster permno date)

* Bandwidth robustness
foreach h of numlist 25 50 75 100 {
    rdrobust buy_users marketcap, c(300) h(`h') p(1)
}

* McCrary density test (manipulation at cutoff)
rddensity marketcap, c(300)
```

### Python
```python
import statspai as sp

# Manipulation test first -- market cap is a running variable firms can
# plausibly influence near a threshold that determines retail attention.
print(sp.rddensity(data=df, x='marketcap', c=300).summary())

# Sharp RDD with MSE-optimal bandwidth and robust bias-corrected CI
rd = sp.rdrobust(data=df, y='buy_users', x='marketcap', c=300,
                 covs=['abs_ret', 'vol_pctl'])
print(rd.summary())
sp.rdplot(data=df, y='buy_users', x='marketcap', c=300)

# Bandwidth sweep as robustness (NOT as the headline -- passing h also changes
# how the bias-correction bandwidth b is selected)
for h in (25, 50, 75, 100):
    r = sp.rdrobust(data=df, y='buy_users', x='marketcap', c=300, h=h)
    print(h, float(r.estimate), float(r.se))
```

### R
```r
library(rdrobust)
library(fixest)

# Sharp RDD
rdrobust(y = df$buy_users, x = df$marketcap, c = 300, h = 50, p = 1,
         covs = df[, c("abs_ret", "vol_pctl")])

# McCrary density
library(rddensity)
rddensity(df$marketcap, c = 300)

# Outage natural experiment
feols(car_20d ~ herd*outage | date, cluster = ~ permno + date, data = df)
```

### What to report
- Figure 4: RDD plot (discontinuity at $300M in buy count)
- Table 4: RDD estimate with bandwidth robustness (25/50/75/100)
- Table 6: Outage DiD (attention-herding effect disappears)
- McCrary test: no manipulation evidence at cutoff

---

## Case 5 — Meeuwis 2022: Election-Surprise DiD × Partisan Lean

**Setting**: Unexpected 2016 Republican victory × household ZIP-code partisan lean.

### Stata
```stata
use retirement_accounts.dta, clear

* DiD with employer × county × period FE
reghdfe delta_equity_share c.republican##c.post2016 c.log_wealth c.age, ///
    absorb(employer#county#period household) ///
    vce(cluster zip employer)

* 2012 placebo
gen post2012 = (yq >= tq(2012q4))
reghdfe delta_equity_share c.republican##c.post2012 c.log_wealth, ///
    absorb(employer#county#period household) vce(cluster zip)

* Active-rebalancing only (nets passive price drift)
reghdfe active_rebalance c.republican##c.post2016, ///
    absorb(employer#county#period household) vce(cluster zip)

* Heterogeneity by ex-ante trading activity
reghdfe delta_equity_share c.republican##c.post2016##c.high_trading, ///
    absorb(employer#county#period household) vce(cluster zip)
```

### Python
```python
import statspai as sp

# Triple-interacted FE on a household panel: never expand these as patsy
# dummies -- employer x county x period alone can be millions of columns.
# Use an HDFE backend and two-way clustering.
df['rep_post'] = df['republican'] * df['post2016']
res = sp.feols(
    'delta_equity_share ~ rep_post + log_wealth + age'
    ' | employer^county^period + household',
    data=df, vcov={'CRV1': 'zip'},
)

# 2012 placebo -- same specification, wrong election
df['rep_post2012'] = df['republican'] * df['post2012']
placebo = sp.feols(
    'delta_equity_share ~ rep_post2012 + log_wealth | employer^county^period + household',
    data=df, vcov={'CRV1': 'zip'},
)
```

### R
```r
library(fixest)
feols(delta_equity_share ~ republican*post2016 + log_wealth
      | employer^county^period + household,
      cluster = ~ zip + employer, data = df)

# 2012 placebo
feols(delta_equity_share ~ republican*post2012 + log_wealth
      | employer^county^period + household,
      cluster = ~ zip, data = df)
```

### What to report
- Table 3: DiD estimate with progressive FE (none → county × period → employer × county × period)
- Table 4: 2012 placebo (null result)
- Table 5: Active rebalancing vs. passive drift decomposition
- Figure 3: UMSC belief survey corroboration (macro beliefs diverge by party post-2016)

---

## Cross-Case Checklist Before Submission

| Check | DiD | IV | RDD | Event study |
|-------|-----|----|-----|-------------|
| Pre-trends plot | ✅ | — | — | ✅ |
| Design-appropriate weak-IV diagnostics/inference | — | ✅ | — | — |
| McCrary density | — | — | ✅ | — |
| Placebo (timing/threshold/event) | ✅ | ✅ | ✅ | ✅ |
| Bandwidth/window robustness | — | — | ✅ | ✅ |
| Cluster at treatment-assignment level | ✅ | ✅ | ✅ | ✅ |
| Alternative FE specs | ✅ | ✅ | — | ✅ |
| Economic magnitude (not just t-stat) | ✅ | ✅ | ✅ | ✅ |
