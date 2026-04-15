# Prompt: Finance Application of a Causal Method

Copy-paste this to start a new implementation on a finance dataset.

---

```
I need to implement [METHOD: DiD / IV / RDD / Event Study / Synthetic Control]
on a finance dataset.

**Setting**:
- Outcome Y: [e.g., firm investment, stock CAR, loan spread]
- Treatment D: [e.g., policy shock, natural experiment, discontinuity at X]
- Unit i: [firm / bank / analyst / household / portfolio]
- Time t: [daily / quarterly / annual, 20XX–20YY]
- Data source: [Compustat, CRSP, Call Reports, 13F, TAQ, Y-14Q, ...]

**Identification narrative** (2–3 sentences):
[Source of exogenous variation — the shock, the cutoff, the IV]

**Task**:
1. Generate [Python / R / Stata] code using the template from
   `references/method-patterns.md` §[N] and finance case from
   `references/jf-case-studies.md` Case [X].
2. Fixed effects should be: [firm + year / firm × quarter + ... / bank × branch]
3. Cluster standard errors at: [the treatment-assignment level]
4. Add robustness checks from `references/identification-writing-patterns.md` §5:
   - [ ] Pre-trends / event study plot
   - [ ] Placebo treatment timing or threshold
   - [ ] Alternative FE specification
   - [ ] Bandwidth / window robustness (if applicable)
   - [ ] First-stage F > 10 (if IV)
   - [ ] McCrary density (if RDD)
5. Write the identification paragraph following the pattern in
   `references/identification-writing-patterns.md` §4.[matching case].

**Output format**:
- Main estimation code block
- Robustness code blocks
- Draft identification paragraph (≤120 words) using appropriate causal verbs
  (avoid "capture / span / predict" unless the design is non-causal)
```

---

## Filled example

```
I need to implement a **weather-shock IV** on a firm panel.

**Setting**:
- Outcome Y: Δ credit-line drawdown
- Treatment D: firm quarterly cash flow (EBITDA/assets)
- Instrument: abnormal Q1 snow cover (county-deviation from 10-year average)
- Unit i: U.S. corporate borrower
- Time t: 2012–2016 quarterly
- Data source: Fed Y-14Q + NOAA county snow cover

**Identification narrative**: Following Brown et al. (2021), severe winter
snow cover causes short-run cash-flow disruption but is unlikely to affect
long-run investment opportunities or credit supply — generating arguably
exogenous variation in firm liquidity.

**Task**:
1. Generate Stata code from §6 (IV) of method-patterns and Case 3 of
   jf-case-studies.
2. FE: NAICS-4 × year-quarter + county.
3. Cluster at NAICS-4 industry.
4. Robustness: P95 snow alternative IV, industry partition, covariate balance
   across snow terciles, first-stage F.
5. Draft identification paragraph.
```
