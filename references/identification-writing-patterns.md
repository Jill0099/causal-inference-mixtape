# Identification Writing Patterns (Finance)

Exemplar paragraphs, framing verbs, and robustness checklists extracted from 25 top-tier JF papers (2021-2024). Use when writing an identification section or reviewing a draft for over-claiming.

---

## §1. The Three-Layer Defense (Gold Standard for Policy Thresholds)

Bennedsen et al. 2022 demonstrates the canonical package when a policy threshold drives treatment:

1. **Event-study pre-trends plot** — year-by-year coefficients on `1{Treat} × 1{year=t}` for t ∈ [−k, +k]; show flat pre, sharp post
2. **Placebo thresholds** — sweep the running variable excluding the true cutoff (e.g., placebos at 15, 20, ..., 100 employees excluding the 20–50 band around the true cutoff of 35); all placebo coefficients insignificant
3. **Diff-in-discontinuities** — estimate the RDD before and after the policy change and difference them, netting out any pre-existing discontinuity at the cutoff (Grembi-Nannicini-Troiano 2016)

When you see a threshold-based design *without* all three layers, flag it.

---

## §2. Saturated Interacted FE — When to Use What

| Goal | FE structure | Paper |
|------|--------------|-------|
| Kill firm fundamentals while keeping within-firm-time analyst variation | `firm × quarter + analyst` | Kempf 2021 |
| Kill bank-level time-varying demand; keep branch × county variation | `bank × branch` + county × year | Drechsler 2021 |
| Kill local labor-market / hedging; keep belief-channel variation | `employer × county × period` | Meeuwis 2022 |
| Kill global macro; keep within country-industry-month variation | `country × year-month + industry` | Bolton 2023 |

**Clustering rule**: cluster at the highest level of *treatment assignment*, not FE. (Abadie-Athey-Imbens-Wooldridge 2023.)

---

## §3. Verb-Framing: Causal vs. Spanning vs. Descriptive

| Framing | Verbs | Required evidence | Example |
|---------|-------|-------------------|---------|
| **Causal** | *cause, drive, affect, reduce, increase, lead to* | exogenous variation + exclusion + robustness triad | "The transparency mandate *causes* a 13% narrowing of the gender pay gap" |
| **Spanning** | *capture, account for, explain, price, subsume* | spanning regression + GRS / joint α test | "Factor momentum *explains* individual-stock momentum" |
| **Predictive** | *forecasts, predicts, has predictive ability* | OOS R², Clark-West test | "Anomaly returns *predict* the market excess return out-of-sample" |
| **Associational** | *is associated with, is correlated with, covaries* | within-unit FE + validation | "Climate exposure *is associated with* green-patent growth" |
| **Descriptive** | *we document, we characterize, we show that* | summary statistics, time series | "We document a 50% rise in ESG fund ownership" |

**Over-claiming red flags**: using "cause / drive" without exogenous variation; using "explain" when you mean "correlate with"; conflating OOS predictive power with economic mechanism.

---

## §4. Exemplar Identification Paragraphs (Verbatim)

### 4.1 DiD — Bennedsen 2022
> "We estimate a difference-in-differences (diff-in-diff) model with treated employees being those working in firms that employ 35 to 50 employees prior to the introduction of the law and control employees being those working in firms with 20 to 34 workers… By including individual × firm fixed effects, we control for time-invariant person characteristics, time-invariant firm characteristics, and the match between firms and workers, which allows us to compare the same employee at the same firm before and after the regulation change."

### 4.2 Saturated FE — Kempf 2021
> "By comparing rating actions of analysts who rate the same firm at the same point in time, we ensure our results cannot be driven by differences in the fundamentals of rated firms… we focus on how the behavior of analysts changes depending on whether their preferred party is in power, as opposed to static differences between Democratic and Republican analysts."

### 4.3 IV — Brown 2021
> "Our identifying assumption is that severe winter weather affects corporate liquidity management only through its effect on cash flows. The temporary nature of our severe winter weather measure makes this assumption plausible. Unlike highly destructive natural disaster events such as hurricanes or earthquakes, abnormal snow cover is unlikely to affect investment opportunities or access to capital, except through its effect on the cash flows of current projects."

### 4.4 Sharp RDD — Barber 2022
> "Robinhood requires stocks above $300 million in market capitalization to be displayed on the Top Movers list. We use a sharp regression discontinuity design to show that Robinhood users are more likely to buy stocks with market capitalization between $300 million and $350 million than stocks with similar absolute returns but market capitalization between $250 million and $300 million."

### 4.5 Within-Unit Subunit Variation — Drechsler 2021
> "This estimation uses only differences in deposit rates across branches of the same bank. It thus removes time-varying bank characteristics (e.g., loan demand), giving us a clean measure of local market power."

### 4.6 Honest Limitation Framing — Bolton 2023
> "As is well known, cross-country studies are beset by endogeneity and identification challenges, as country-level variation can be driven by many different sources. In this study, we can to some extent overcome these challenges by exploiting rich country-, industry-, and firm-level variation in carbon emissions and other characteristics to identify the different sources of transition risk."

### 4.7 Non-Causal Disclaimer — Biais 2023
> "The calibrated coefficients imply required returns that increase in transactional costs and decrease in transactional benefits. That said, we do not claim statistical significance or estimation of population parameters because of the relatively small size of our sample and the likely non-stationarity of our variables."

### 4.8 Prediction, Not Structure — Kelly 2024
> "We prove that expected out-of-sample forecast accuracy and portfolio performance are strictly increasing in model complexity when appropriate shrinkage is applied... The interpretation is not necessarily that asset returns are subject to a large number of fundamental driving forces. Rather, even when the driving variables have low dimension, complex models better leverage the information content of G_t."

---

## §5. Robustness Checklist by Design

### 5.1 DiD
- [ ] Event-study plot (leads + lags with t=−1 as reference)
- [ ] Parallel-trends test (joint significance of leads)
- [ ] Placebo treatment timing
- [ ] Alternative control group (matched / synthetic)
- [ ] Staggered-DiD: Bacon decomposition OR Callaway-Sant'Anna OR Sun-Abraham
- [ ] Clustering at treatment-assignment level
- [ ] Triple-difference (DDD) when available (Bennedsen)
- [ ] Diff-in-disc if there is a threshold (Bennedsen)

### 5.2 IV / 2SLS
- [ ] First-stage F > 10 (Stock-Yogo) or > 104 (Lee et al. 2022 correction)
- [ ] Exclusion-restriction narrative (why the IV affects Y only through D)
- [ ] Over-identification test (Hansen J) if multiple IVs
- [ ] Reduced-form + first-stage visualization
- [ ] LATE vs. ATE interpretation (who are the compliers?)
- [ ] Reverse-causality placebo

### 5.3 RDD
- [ ] McCrary density test at the cutoff
- [ ] Covariate-balance at the cutoff
- [ ] Bandwidth robustness (half, double, CCT-optimal, Imbens-Kalyanaraman)
- [ ] Polynomial-order robustness (linear, quadratic; Gelman-Imbens caution vs. quartic+)
- [ ] Placebo cutoffs
- [ ] Fuzzy-RDD first-stage check

### 5.4 Event Study (Financial Event)
- [ ] Clean event window (no overlapping announcements)
- [ ] Multiple-comparison correction if scanning many events
- [ ] Pre-event return normal
- [ ] Placebo "non-event" dates
- [ ] Sign-flip test (for 2016 election: effect reverses if "treatment" is flipped)

### 5.5 Structural / Calibration
- [ ] Parameter sensitivity analysis
- [ ] Out-of-sample moment fit
- [ ] One-friction-at-a-time counterfactual
- [ ] External validation on reduced-form moment

### 5.6 Non-Causal Framing (Asset Pricing)
- [ ] Factor-model horse race (CAPM → FF3 → Carhart 4 → FF5 → HXZ q-factor → Stambaugh-Yuan)
- [ ] GRS joint α test
- [ ] OOS test (Clark-West for nested)
- [ ] Newey-West SE with appropriate lags
- [ ] Subsample stability
- [ ] Explicit disclaimer of non-causal scope

---

## §6. Common Over-Claim Pitfalls (Finance)

1. **"Exogenous" without exclusion argument** — labeling a regressor "exogenous" doesn't make it so. Require the exclusion restriction to be explicit.
2. **Cross-country panel with many FE ≠ causal** — Bolton 2023 is honest about this. Rich FE reduce confounds but don't produce exogenous variation.
3. **Event study with endogenous timing** — e.g., firms self-select into M&A announcement dates. Use propensity-score weighting or narrow windows.
4. **RDD with manipulated running variable** — always run McCrary (2008) or Cattaneo-Jansson-Ma (2020).
5. **OOS predictive power ≠ economic cause** — Kelly 2024 is explicit: "not that returns are subject to many fundamental forces."
6. **Saturated FE absorbs treatment variation** — check that the FE structure does not soak up the treatment itself (e.g., `county × year` FE absorbs a state-level policy).
7. **p-hacked anomaly** — Jensen 2023 framework: apply Benjamini-Yekutieli FDR or Bayesian theme-shrinkage before trusting any single factor α.
