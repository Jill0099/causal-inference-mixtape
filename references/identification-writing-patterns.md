# Identification Writing Patterns (Finance)

Framing verbs, exemplar identification *moves*, and robustness checklists drawn
from 25 top-tier JF papers (2021-2024). Use when writing an identification
section or reviewing a draft for over-claiming.

> §4 describes each paper's rhetorical structure and gives an adaptable
> template. It does not reproduce article text — consult the originals, which
> remain under their publishers' copyright.
>
> Related: [`reporting-checklist.md`](reporting-checklist.md) for what to print
> alongside each estimate, [`design-router.md`](design-router.md) for choosing
> the design in the first place, and
> [`inference-and-standard-errors.md`](inference-and-standard-errors.md) for
> clustering and parallel-trends sensitivity.

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

## §4. Exemplar Identification Moves

Each entry below describes the **rhetorical move** a published identification
paragraph makes, and the template you can adapt. These are paraphrases and
structural summaries, not reproductions — go to the papers themselves for the
authors' own wording.

### 4.1 DiD with a size threshold — Bennedsen et al. (2022), *JF*

**The move:** define treatment and control as narrow bands on either side of the
regulatory threshold, then name exactly what the fixed-effect structure removes.

> *Template.* "Treated units are those with [running variable] in [band just
> above the cutoff] before [policy]; controls are those in [band just below].
> By including [unit × subunit] fixed effects we absorb time-invariant [unit]
> characteristics, time-invariant [subunit] characteristics, and the match
> between them, so the comparison is the same [unit] at the same [subunit]
> before and after the change."

The strength is the second sentence: it names the variation being used, not just
the variation being removed.

### 4.2 Saturated FE — Kempf & Tsoutsoura (2021), *JF*

**The move:** state the comparison in plain language ("the same firm, the same
quarter, different analysts"), and distinguish the *within-unit change* from a
*static between-group difference*.

> *Template.* "By comparing [decisions] made about the same [unit] at the same
> point in time, our results cannot be driven by differences in [unit]
> fundamentals. We study how [agent] behaviour CHANGES with [treatment], not
> static differences between [type A] and [type B] agents."

### 4.3 IV exclusion — Brown, Gustafson & Ivanov (2021), *JF*

**The move:** argue exclusion by contrasting the instrument with a *stronger*
shock that would violate it.

> *Template.* "Our identifying assumption is that [instrument] affects [outcome]
> only through [channel]. The [temporary / localised / mild] nature of the shock
> makes this plausible: unlike [destructive alternative — hurricanes,
> earthquakes], [instrument] is unlikely to affect [alternative channel A] or
> [channel B], except through [the intended channel]."

This is the single most transferable paragraph in the set. The contrast case is
what does the work.

### 4.4 Sharp RDD — Barber et al. (2022), *JF*

**The move:** name the institutional rule that creates the cutoff, then state the
comparison as two adjacent bins that are alike on everything except eligibility.

> *Template.* "[Institution] requires [running variable] above [cutoff] for
> [treatment]. We use a sharp RD design to compare units with [running variable]
> in [cutoff, cutoff+δ] against units with similar [key covariate] but [running
> variable] in [cutoff−δ, cutoff]."

### 4.5 Within-unit subunit variation — Drechsler et al. (2021), *JF*

**The move:** one sentence naming the confound the design removes and the
quantity it therefore isolates.

> *Template.* "This estimation uses only differences across [subunits] of the
> same [unit]. It removes time-varying [unit] characteristics such as
> [confound], giving a clean measure of [target quantity]."

### 4.6 Honest limitation framing — Bolton & Kacperczyk (2023), *JF*

**The move:** concede the design's limits up front, then state precisely how far
the data goes — "to some extent", not "we solve this".

> *Template.* "Cross-[level] studies are beset by endogeneity and identification
> challenges, since [level]-level variation can be driven by many sources. We can
> to some extent address these by exploiting rich [dimension A], [dimension B]
> and [dimension C] variation, but we do not claim [level]-level exogeneity."

Referees reward this. Overclaiming and then being caught costs far more than
conceding scope in the paper.

### 4.7 Non-causal disclaimer — Biais et al. (2023), *JF*

**The move:** report the calibration, then explicitly decline the inferential
claim a reader might otherwise import.

> *Template.* "The calibrated coefficients imply [pattern]. That said, we do not
> claim statistical significance or estimation of population parameters, given
> [small sample / likely non-stationarity]."

### 4.8 Prediction, not structure — Kelly, Malamud & Zhou (2024), *JF*

**The move:** prove the statistical result, then pre-empt the structural
interpretation readers will reach for.

> *Template.* "We show [statistical result about model complexity / predictive
> accuracy]. The interpretation is NOT that [economic quantity] is driven by
> [many fundamental forces]; rather, [alternative mechanical explanation]."

The value here is the explicit negation. If you can predict the misreading, say
it is a misreading.

---

## §5. Robustness Checklist by Design

### 5.1 DiD
- [ ] Event-study plot (leads + lags, t=−1 omitted from the design matrix and
      re-inserted in the plot as an exact zero; endpoints **binned**)
- [ ] Parallel-trends test (joint significance of leads) **and its power**
      (Roth 2022 — a flat plot under a low-power test proves very little)
- [ ] **Honest DiD (Rambachan-Roth 2023)**: robust CIs over an M grid, with the
      breakdown M stated in words. A breakdown below 1 is a fragile result.
- [ ] Placebo treatment timing
- [ ] Alternative control group (matched / synthetic)
- [ ] Staggered-DiD: Bacon decomposition **and** a heterogeneity-robust headline
      estimator (Callaway-Sant'Anna / Sun-Abraham / BJS), TWFE as benchmark only
- [ ] Clustering at treatment-assignment level; report cluster count **and
      treated-cluster count**; wild cluster bootstrap below ~50 clusters
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
