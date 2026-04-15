# Finance Applications — Method Catalog

Indexed by identification strategy. Each row is a real Journal of Finance (2021-2024) application — Y, D, unit, FE, clustering, data source — so you can adapt the template to your own finance setting.

---

## §1. Difference-in-Differences (DiD)

### 1.1 Bennedsen et al. 2022 — Gender Pay Gap Transparency

- **Shock**: Denmark's 2006 Act 562 (firms ≥35 employees must publish gender-disaggregated wages)
- **Y**: log(annual wage); log(hourly); bonus; firm-level pay gap; productivity; profits
- **D**: 1{firm had 35–50 employees in 2005} × 1{year ≥ 2006}
- **Unit**: worker × firm × year
- **FE**: individual × firm + year (+ firm × year in saturated spec)
- **Clustering**: two-way, individual × firm
- **Data**: Danish IDA matched employer-employee
- **Key spec**:
  ```
  log(wage)_{i,j,t} = α_{i,j} + α_t + β₁·(Treat_j × Post_t) + β₂·(Post_t × Male_i)
                   + δ·(Treat_j × Post_t × Male_i) + X'γ + ε
  ```
- **Robustness triad** (gold standard): (1) event-study pre-trends plot; (2) placebo thresholds swept 15→100 employees; (3) diff-in-discontinuities nets pre-existing cutoff

### 1.2 Kempf & Tsoutsoura 2021 — Partisan Analysts

- **Shock**: U.S. presidential party turnover (2000, 2008, 2016 etc.) interacted with analyst voter registration
- **Y**: quarterly rating adjustment (notches), downgrade indicator, bond yield, firm investment
- **D**: 1{analyst party ≠ sitting-president party}
- **Unit**: analyst a × firm f × quarter
- **FE**: **firm × quarter + agency × quarter + analyst** (saturated — kills firm fundamentals)
- **Clustering**: analyst (and firm)
- **Data**: hand-matched S&P/Moody's/Fitch analysts × voter registration
- **Key spec**:
  ```
  RatingAdj_{a,f,q} = β·Misaligned_{a,q} + μ_{f,q} + μ_{g,q} + μ_a + ε_{a,f,q}
  ```
- **Identification claim**: "By comparing rating actions of analysts who rate the same firm at the same point in time, we ensure our results cannot be driven by differences in the fundamentals of rated firms."

### 1.3 Meeuwis et al. 2022 — Belief Disagreement × 2016 Election

- **Shock**: unexpected Republican 2016 victory as common public signal with partisan-specific interpretation
- **Y**: Δequity share, portfolio market beta
- **D**: Republican_i (ZIP-code partisan lean) × Post_t (post-Nov 2016)
- **Unit**: retirement household
- **FE**: **employer × county × period** (kills hedging/income-risk); + household
- **Clustering**: ZIP code (and employer)
- **Data**: proprietary retirement-account panel + UMSC beliefs survey + Betfair election prices
- **Placebo**: 2012 election (no partisan differential expected — confirmed)

---

## §2. Instrumental Variables (IV / 2SLS)

### 2.1 Brown et al. 2021 — Weather Shocks as IV for Cash Flow

- **IV**: abnormal winter snow cover (county-quarter Q1 deviation from 10-year average)
- **Exclusion**: "severe winter weather is unlikely to affect investment or credit supply except through short-run disruption of current cash flows"
- **Y**: ΔCredit line drawdown, Δcredit line size, interest rate, maturity
- **D**: Annual cash flow (EBITDA/assets), instrumented
- **FE**: NAICS-4 × year-quarter + county (firm FE in robustness)
- **Clustering**: NAICS-4 industry
- **First-stage F ≈ 20** (no weak-instrument concern)
- **Data**: Fed Y-14Q Schedule H.1 + NOAA county-day snow cover
- **Why this generalizes**: any weather/geographic arguably-exogenous shock that affects firm short-run liquidity without affecting long-run fundamentals

### 2.2 Falato et al. 2022 — Pledged Patents as IV for Intangible Collateral Capacity

- **IV**: Mann (2018) pledged-patents share (exogenous cross-firm variation in intangible pledgeability)
- **Y**: cash-to-assets, leverage, investment rate
- **D**: firm intangible intensity
- **FE**: firm + year (or industry × year)
- **Clustering**: firm

### 2.3 Wang et al. 2022 — BLP-Style IVs for Bank Loan/Deposit Demand

- **IV**: rival-bank characteristics / Hausman-type local-market instruments
- **2SLS on**: `log(s_{j,t}) − log(s_{0,t}) = α·r^{dep}_{j,t} + X'β + ξ_{j,t}`
- **Embedded in**: 2-stage structural estimation (BLP demand → SMD on dynamic moments)

---

## §3. Regression Discontinuity (RDD)

### 3.1 Barber et al. 2022 — Sharp RDD at $300M Market Cap

- **Cutoff**: Robinhood's Top Movers list eligibility requires market cap ≥ $300M
- **Y**: Robinhood buy-count; 20-day DGTW-adjusted CAR
- **Running variable**: market cap
- **Bandwidth**: $250M ($250-300 control vs. $300-350 treated)
- **FE**: day FE; polynomial in market cap
- **Clustering**: stock and date (two-way)
- **Complement**: outage natural experiment (when RH is down, effect disappears)
- **Data**: Robintrack (scraped RH API), CRSP, TAQ

### 3.2 Bennedsen 2022 — Diff-in-Discontinuities

- **Running variable**: firm size (employees)
- **Cutoff**: 35 employees (triggers wage-transparency mandate)
- **Variant**: diff-in-disc (Grembi-Nannicini-Troiano 2016) — difference the RDD before and after the policy change to net out any pre-existing discontinuity at 35
- **Placebo cutoffs**: 15, 20, 25, ..., 100 (excluding 20–50)

---

## §4. Event Studies

### 4.1 FOMC Event Study — Drechsler et al. 2021

- **Event**: FOMC meeting-day rate surprise
- **Window**: meeting-day trading window
- **Y**: bank stock CAR
- **Cross-sectional mapping**: bank equity return β_i on FOMC-day rate shock → "FOMC-beta"

### 4.2 Election Event Study — Hsu 2023, Kempf 2021

- **Event**: 2016 Trump election (arguably exogenous regulatory-regime shift)
- **Y**: 3-day CAR by firm characteristic bin
- **Use case**: converts a correlational cross-sectional premium into a discount-rate-channel narrative without a full DiD

### 4.3 Platform Outage — Barber 2022

- **Event**: Robinhood server outages (exogenous access shutdown)
- **Within-stock check**: attention-induced herding disappears on outage days

---

## §5. Saturated Interacted FE (The "Within-Unit-Time" Pattern)

**General form**: absorb unit × time FE so identification comes only from within-cell cross-observation variation.

| Paper | FE structure | What it absorbs |
|-------|--------------|-----------------|
| Kempf 2021 | firm × quarter + agency × quarter + analyst | firm fundamentals; agency trends |
| Drechsler 2021 | bank × branch | time-varying bank demand/credit |
| Meeuwis 2022 | employer × county × period | local labor-market/hedging |
| Bolton 2023 | country × year-month + industry | global macro shocks |
| Bennedsen 2022 | individual × firm | job-match |

**Rule of thumb**: cluster at the highest level of treatment assignment, not at the highest FE level.

---

## §6. Structural Estimation (Calibration / SMD / BLP)

### 6.1 Wang et al. 2022 — Two-Stage BLP + SMD

- **Stage 1**: BLP demand — deposit/loan market shares with rival-bank IVs
- **Stage 2**: SMD — match dynamic moments (spreads, leverage, maturity) with frictions parameters
- **Counterfactual**: shut off each friction one-by-one to decompose pass-through

### 6.2 Falato et al. 2022 — Dynamic Q-Model + Calibration

- **Model**: two-asset (tangible/intangible), collateral constraint, costly equity
- **Calibration**: match firm-level cash/investment moments
- **Validation**: reduced-form FE regression + pledged-patents IV

### 6.3 Duarte & Eisenbach 2021 — Stress-Test Counterfactual

- **Identification**: when real identification is infeasible (crisis data), apply standardized hypothetical 1% asset shock each quarter
- **AV index** decomposes additively into bank "systemicness" and multiplicatively into 4 system factors
- **Validation**: out-of-sample predictive power for SRISK, ΔCoVaR, MES 5 years ahead

---

## §7. Non-Causal Asset Pricing (Reference Only — Contrast Cases)

These papers deliberately avoid causal-identification language. Useful template for *how to frame empirical claims without over-claiming*:

| Paper | Method | Writing pattern |
|-------|--------|-----------------|
| Liu 2022 (Crypto) | 3-factor model + quintile sorts | "three factors capture the cross-section" |
| Biais 2023 (Bitcoin) | Structural GE + calibration | explicit disclaimer: "we do not claim statistical significance of population parameters" |
| Kelly 2024 (Complexity) | Ridgeless high-dim + OOS Sharpe | "not that returns are subject to many fundamental forces — complex models better leverage info" |
| Dong 2022 (Anomalies) | Shrinkage + Clark-West OOS | mechanism story (MCP) deferred: "risk-based framework beyond scope" |
| Ehsani 2022 (Momentum) | Spanning + GRS joint α | "factor momentum subsumes stock momentum" (spanning ≠ causation) |
| Barberis 2021 (Prospect) | Structural model + calibration | quantitative horse-race across 23 anomalies, no causal claim |

**Lesson for writing**: use "capture / account for / span / price" — not "cause / drive / affect" — when there is no exogenous variation.

---

## §8. Measurement-First Papers

### 8.1 Sautner 2023 — Firm Climate Exposure (NLP Measure)

- **Measure**: King-Lam-Roberts ML bigram discovery on earnings-call transcripts
- **Validation ladder**: face validity (human-audit fragments) → variance decomp (70-96% firm-level) → correlation with off-the-shelf (EGKLS) → out-of-sample predictive power (green patents, green-tech jobs) → asset-pricing tests
- **FE**: firm + time
- **Honest framing**: associational within-firm, not causal

### 8.2 Boehmer et al. 2021 — Retail Trade Classification

- **Institutional rule as classifier**: Reg NMS permits sub-penny price improvement only for off-exchange (wholesaler) trades → Z_it = 100·mod(P_it, 0.01) identifies retail orders
- **Validation**: NASDAQ TRF audit-trail subset
- **Use**: provides a public-tape proxy for retail activity without proprietary data

---

## §9. Meta-Empirical / Multiple Testing

### 9.1 Jensen-Kelly-Pedersen 2023 — Replication Crisis in Finance

- **Bayesian hierarchical**: α_{c,θ,f} ~ N(μ_θ, ω²) nested in themes θ, countries c
- **Shrinkage**: κ = τ²/(τ² + σ²/T), pulls toward theme mean
- **MT correction**: Benjamini-Yekutieli FDR
- **External validity**: 153 factors × 93 countries OOS replication
- **Alpha-hacking model**: adds u ~ N(ε̄, σ²_u) to account for data-snooping distortion

**Use when**: you have many hypothesis tests and want to avoid the "factor zoo" problem.

---

## Cross-Paper Data Spine (Finance)

| Data source | Used in |
|-------------|---------|
| CRSP | All asset-pricing papers |
| Compustat | Firm fundamentals (most papers) |
| Call Reports + FDIC SOD | Drechsler, Wang |
| Y-14Q Schedule H.1 | Brown (credit lines) |
| 13F (Thomson Reuters) | Lewellen, Dong, Meeuwis (indirect) |
| Ratewatch | Drechsler (branch-level deposits) |
| TAQ / FINRA TRF | Boehmer, Barber |
| Trucost / ASSET4 / S&P Global ESG | Sautner, Bolton, Hsu, Starks |
| NOAA | Brown (weather IV) |
| Ken French factor library | Liu, Welch, Dong, Ehsani |
| Danish IDA | Bennedsen |
| Robintrack (scraped) | Barber, Welch |
