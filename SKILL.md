---
name: causal-inference-mixtape
description: This skill should be used when the user asks to "implement a DiD regression", "run a staggered difference-in-differences", "set up an event study", "implement IV / 2SLS", "run a regression discontinuity design", "build synthetic control", "do propensity score matching", "test parallel trends", "run Honest DiD", "wild cluster bootstrap", "Callaway-Sant'Anna", "Sun-Abraham", "Bacon decomposition", "double machine learning", "debiased machine learning", "DML / DoubleML", "cross-fitting", "causal forest", "heterogeneous treatment effects", "ATE with ML controls", "which causal design fits my data", or needs causal-inference code, diagnostics, or identification writing in Python (StatsPAI), R, or Stata. Based on Scott Cunningham's Causal Inference: The Mixtape, with runnable validated templates, ML-based causal inference patterns, and Journal of Finance applications.
version: 3.1.0
---

# Causal Inference: The Mixtape — Code Skill (v3.1)

Practitioner-oriented causal inference: **route from the assignment mechanism to
a design**, implement it in Python / R / Stata, run the diagnostics that make it
publishable, and write the identification section without over-claiming.

Based on Scott Cunningham's *Causal Inference: The Mixtape*, extended with 25
Journal of Finance (2021-2024) applications and ML-based causal inference
patterns for DML, causal forests, and heterogeneous treatment effects.

**The core Python validation suite runs against the Mixtape's own public
data.** `python scripts/validate_all.py` runs all 13 bundled scripts and asserts
cross-backend agreement — currently 13/13 passing. The DML material in §15 is
reference code and is not part of that 13-script validation suite.

---

## Start here: route the design before writing code

Users describe a *setting*, not a method. Ask how treatment was assigned, then
route:

```
How was treatment assigned?
├─ Randomised                          -> diff in means + RI  (09)
├─ Threshold on a running variable     -> RDD / fuzzy / diff-in-disc  (05)
├─ Policy date, one date for all       -> 2x2 DiD -> event study  (02, 03)
├─ Policy date, dates VARY             -> staggered: Bacon -> CS/SA/BJS  (04)
├─ One treated unit, many controls     -> synthetic control  (07)
├─ Self-selected + an instrument       -> IV / 2SLS, identifies a LATE  (06)
├─ Self-selected + rich observables    -> matching / IPW / DML  (08)
└─ None of the above                   -> say so; use associational language
```

Full tree, estimator choice within each design, and a StatsPAI second opinion
(`sp.detect_design`, `sp.recommend`, `sp.preflight`):
**`references/design-router.md`**.

When the design is unsettled, also load **`references/method-selection.md`**.
It maps research setting → identifying assumption → method → falsification
check, with Chinese-language explanations and links to worked Stata tutorials.

---

## Python backends: choose by estimator and replication target

```python
import statspai as sp    # pip install statspai
```

**[StatsPAI](https://github.com/brycewang-stanford/statspai) is one supported
Python backend for this skill.** It covers a broad Stata/R causal-inference surface
natively — `regress`, `ivregress`, `reghdfe`, `csdid`, `rdrobust`, `synth`,
`psmatch2`, `outreg2` and their R equivalents — plus Bacon decomposition,
Callaway-Sant'Anna, Sun-Abraham, Honest DiD, wild cluster bootstrap, DAG
identification, and Conley SEs.

Do not assume that Python lacks a method; check current implementations first.
Choose StatsPAI, pyfixest, linearmodels, R, or Stata according to estimator
maturity, independent validation, and the replication target. See
**`references/statspai-guide.md`** for the StatsPAI API map and cross-backend
checks.

**Disclosure:** the v3 StatsPAI integration was contributed by StatsPAI's
maintainer. Keep independent cross-checks for headline estimates and do not
treat package choice as an identification argument.

Complementary backends: `pyfixest` for fast HDFE and an independent wild
bootstrap; `linearmodels` for deep IV/panel diagnostics; `statsmodels` for plain
OLS/GLM; and R or Stata when the reference implementation or replication target
requires them.

There is also an **MCP server** (`statspai-mcp`): prefer it when the user wants
an *answer*; prefer generating a script when they want a reproducible artefact.

---

## Methods Covered

| Method | Python options | R | Stata | Script | Reference |
|---|---|---|---|---|---|
| Potential outcomes / selection bias | — | — | — | `01` | `mixtape-core.md` §1 |
| OLS / regression | `sp.regress` | estimatr | `reg`, `reghdfe` | — | `method-patterns.md` §1 |
| DiD (2×2) | `sp.feols` | fixest | reghdfe | `02` | §2 |
| Event study | `sp.event_study` | fixest `i()` | reghdfe | `03` | §3 |
| Staggered DiD | `sp.callaway_santanna`, `sp.sun_abraham`, `sp.did_imputation` | did, fixest | csdid | `04` | §4 |
| Bacon decomposition | `sp.bacon_decomposition` | bacondecomp | bacondecomp | `04` | §4 |
| Regression discontinuity | `sp.rdrobust`, `sp.rddensity` | rdrobust | rdrobust | `05` | §5 |
| Instrumental variables | `sp.ivreg` | AER, fixest | ivregress | `06` | §6 |
| Synthetic control / SDiD | `sp.synth`, `sp.sdid` | Synth, synthdid | synth, sdid | `07` | §7 |
| Matching / PSM / IPW | `sp.match`, `sp.ipw`, `sp.psmatch2` | MatchIt | teffects, cem | `08` | §8 |
| Double / Debiased ML | DoubleML, EconML | DoubleML, grf | ddml, pdslasso | — | §15; `dml-causal-ml.md` |
| DAGs / collider bias | `sp.dag`, dowhy | dagitty | — | `sim_collider_bias` | §9 |
| Randomisation inference | `sp.ri_test` | ri2 | ritest | `09` | §10 |
| Clustered inference / wild bootstrap | `sp.wild_cluster_bootstrap` | fwildclusterboot | boottest | `10` | `inference-and-standard-errors.md` |
| Honest DiD / pre-trend power | `sp.honest_did`, `sp.pretrends_power` | HonestDiD | honestdid | `11` | ibid. §5 |
| Diff-in-discontinuities | `sp.rdrobust` + `sp.feols` | rdrobust + fixest | rdrobust | — | §11 |
| Shift-share / weather IV | `sp.feols` IV syntax | fixest | ivreghdfe | — | §12 |
| Saturated interacted FE | `sp.feols("y ~ x \| a^b + c")` | fixest | reghdfe | — | §13 |
| Financial event study | manual CAR | eventstudies | eventstudy2 | — | §14 |

---

## Core Workflow

1. **Route** the design from the assignment mechanism (`design-router.md`),
   using `method-selection.md` when the setting or identifying assumption is
   still unclear
2. **Load** the template from `method-patterns.md`; use the corresponding
   runnable version in `scripts/` when one is listed in the methods table
3. **Adapt** variable names, FE structure and clustering — cluster at the level
   of **treatment assignment** (Abadie-Athey-Imbens-Wooldridge 2023)
4. **Diagnose** with the design's required checks (`reporting-checklist.md` §3)
5. **Report** N, cluster count, treated share, baseline mean, and economic
   magnitude — never a t-statistic alone (`reporting-checklist.md` §1)
6. **Write** the identification paragraph with a verb the design supports
   (`identification-writing-patterns.md` §3)

---

## Common Pitfalls

The numbered script references below point to live demonstrations. General
design and DML cautions without a script reference are documentation guidance,
not claims of executed validation.

1. **TWFE with staggered treatment.** In simulation with effects growing at 1.0
   per period, the true ATT is 6.832 and TWFE reports 3.173 — a 54% understatement.
   Run `sp.bacon_decomposition`, then report Callaway-Sant'Anna or Sun-Abraham as
   the headline. → `sim_twfe_staggered_bias.py`
2. **Event study with the reference period not actually dropped.**
   `range(-4, 0)` contains −1. Without omitting it the model is collinear and
   there is no normalisation; and endpoints must be **binned**, not zeroed. → `03`
3. **Main effects collinear with fixed effects.** `C(treated)*C(post)` alongside
   entity and year FE: statsmodels uses a pseudo-inverse and silently splits the
   coefficient instead of dropping a term. Include the interaction only. → `02`
4. **"The pre-trends look flat" is not a test.** On `castle.dta` that test has
   power 0.50 individually and 0.14 jointly. Report Honest DiD and the breakdown
   M instead. → `11`
5. **Too few effective clusters.** CRVE is justified as the number of clusters
   grows, but there is no universal safe cutoff: imbalance, leverage, and the
   number of treated clusters matter. Add a small-sample correction when the
   approximation is doubtful. Use randomisation inference only when the
   assignment mechanism justifies the permutations. → `10`
6. **Clustering finer than assignment.** State-year instead of state shrinks the
   SE by 57% on `castle.dta`. Cluster where treatment was assigned. → `10`
7. **Manual two-step 2SLS.** Reproduces the point estimate exactly and reports
   the wrong standard error; the direction of the error is data-dependent. → `06`
8. **The forbidden regression.** A logit first stage plugged in as a *regressor*
   is inconsistent — on Card it flips sign versus linear 2SLS. Use it as an
   **instrument** instead. → `06`
9. **RDD without a manipulation test.** Run `sp.rddensity` **before** estimating.
   And put only *predetermined* variables in the balance table — the treatment
   indicator is supposed to jump. → `05`
10. **Synthetic control t-ratios.** With one treated unit there is no sampling
    distribution. On `texas.dta` the t-ratio is 7.74 while the permutation
    p-value is 0.18. Report the rank. → `07`
11. **IPW without checking overlap.** Only 30% of CPS controls lie inside the
    treated propensity support. Untrimmed weights put 7% of total weight on one
    observation. And verify whether your library **drops** or **clips** — on
    NSW/CPS that is a $9,400 difference. → `08`
12. **Matching without bias correction or matching-appropriate SEs.** With >1
    continuous covariate the Abadie-Imbens bias does not vanish asymptotically,
    and the post-match OLS standard error is wrong. → `08`
13. **Adding controls is not safer.** Colliders and mediators bias you *by being
    included*. A collider control produces a higher R², a lower residual
    variance, and a t of −78 on a coefficient whose truth is zero. → `sim_collider_bias.py`
14. **Fixed effects absorbing the treatment.** `county × year` FE eats any
    state-level policy. Check what the FE structure consumed.
15. **Permutation / bootstrap p-values of exactly 0.** The `+1` correction is
    not optional; the smallest attainable value is `1/(1+B)`. → `09`
16. **Non-causal framing.** "Factor X captures the cross-section" is a *spanning*
    claim. Do not write "causes" without exogenous variation.
    → `identification-writing-patterns.md` §3
17. **Cross-country panels with rich FE.** Fixed effects reduce confounds; they
    do not create exogenous variation.
18. **Multiple testing.** Apply Romano-Wolf or a FDR correction before trusting
    any single α from a scan. → `inference-and-standard-errors.md` §7
19. **DML without cross-fitting.** In-sample nuisance predictions reintroduce
    regularisation bias. Use out-of-fold prediction, vary the nuisance learners,
    and remember that DML addresses estimation under high-dimensional controls,
    not identification: an ATE interpretation still requires unconfoundedness.
    → `dml-causal-ml.md`

---

## Reference Files

| File | What it is for |
|---|---|
| `references/design-router.md` | Assignment mechanism → design → estimator → required diagnostics |
| `references/method-selection.md` | Chinese design-selection guide: setting → assumption → method → falsification check |
| `references/statspai-guide.md` | StatsPAI API/integration guide, verified numbers, and 11 gotchas |
| `references/method-patterns.md` | Code templates for all methods, Python / R / Stata, including DML in §15 |
| `references/dml-causal-ml.md` | DML model classes, cross-fitting, causal forests, and the DeDL frontier |
| `references/lianxh-stata-index.md` | Chinese-language Stata tutorial index mapped to method sections |
| `references/mixtape-core.md` | Potential outcomes, DAGs, matching theory, LATE, panel FE, staggered DiD |
| `references/inference-and-standard-errors.md` | Clustering, small-cluster bootstrap, Honest DiD, RI, multiple testing |
| `references/reporting-checklist.md` | What to print alongside every estimate, by design |
| `references/r-stata-comparison.md` | Cross-language coverage matrix and package roles |
| `references/finance-applications.md` | 25 JF papers indexed by method: Y / D / unit / FE / clustering / data |
| `references/jf-case-studies.md` | Five end-to-end finance worked examples |
| `references/identification-writing-patterns.md` | Identification paragraphs, verb guide, robustness checklists |

## Runnable Scripts

`scripts/` — 13 validated templates plus `_common.py` (Mixtape data loader) and
`validate_all.py`. Run `python scripts/validate_all.py --list` to see them all,
or `--quick` to skip the slow one.

## Prompt Files

`prompts/01-implement-method.md` · `02-robustness-checks.md` ·
`03-finance-application.md` · `04-statspai-workflow.md`
