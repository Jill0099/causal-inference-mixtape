# Changelog

## v3.1.0 — DML and Chinese method-selection references

- Added §15 reference templates for Double/Debiased ML in Python, R, and Stata.
- Added DML/causal-forest concepts, a Chinese design-selection guide, and a
  public 连享会 Stata tutorial index.
- Scoped DML to settings with defensible identification and overlap; clarified
  that it does not automatically correct measurement error or create exogenous
  variation.
- Kept the v3 core validation boundary explicit: the 13 bundled scripts remain
  validated, while §15 uses optional dependencies and is reference-only.

## v3.0.0 — Runnable, validated, multi-backend templates

The skill previously shipped inert markdown templates. Several contained bugs
that would not run or would silently produce wrong estimates, and the Python
ecosystem guidance was several years out of date. v3 makes the code executable
and checks it.

### Added — runnable, validated templates

- `scripts/` — 13 executable templates against the Mixtape's own public data,
  each asserting its own invariants (cross-backend agreement, algebraic
  identities, design sanity checks). **13/13 passing in 104s.**
- `scripts/_common.py` — Mixtape data loader with disk cache, backend version
  report, tidy-frame normaliser across pyfixest/StatsPAI schemas, and
  `design_facts()` for the numbers every table should carry.
- `scripts/validate_all.py` — one-command regression sweep (`--quick`, `--list`).
- `scripts/simulations/` — collider bias and staggered-TWFE bias, both scored
  against a known truth rather than asserted.
- `scripts/requirements.txt` and `requirements.lock.txt` — direct validated
  versions plus the fully pinned, hashed transitive environment.

### Added — references

- `references/statspai-guide.md` — StatsPAI integration guide: API map,
  Stata/R migration table, recommended workflow, verified cross-backend numbers,
  and **11 non-obvious gotchas found by running it**.
- `references/design-router.md` — assignment mechanism → design → estimator →
  required diagnostics, so a user who describes a *setting* gets routed without
  having to name a method first.
- `references/inference-and-standard-errors.md` — cluster level vs assignment
  level, few-cluster wild bootstrap, BDM serial correlation, two-way and Conley
  clustering, Honest DiD and pre-trend power, randomisation inference, multiple
  testing.
- `references/mixtape-core.md` — the conceptual layer: potential outcomes and
  the selection-bias decomposition, DAGs (confounder/collider/mediator),
  matching theory (Abadie-Imbens, King-Nielsen), LATE/monotonicity/compliers and
  the judge-leniency design, panel FE limits and Nickell bias.
- `references/reporting-checklist.md` — what to print alongside every estimate.
- `prompts/04-statspai-workflow.md` — end-to-end pipeline prompt.

### Fixed — template bugs

Each was reproducible and would have produced wrong output:

1. **Event study reference period never dropped.** `range(-4, 0)` includes −1, so
   the loop that claimed to exclude it excluded nothing — a perfectly collinear
   design with no normalisation. Endpoints are now binned rather than zeroed, and
   the normalisation is asserted.
2. **Event-study plot arrays misaligned.** `list(range(-4,0)) + [0] + list(range(0,5))`
   duplicated 0 and returned 10 entries for 9 coefficients.
3. **Main effects collinear with fixed effects.** `C(treated)*C(post)` alongside
   entity and year FE: statsmodels uses a pseudo-inverse and silently splits the
   coefficient rather than dropping a term.
4. **`results.first_stage.diagnostics['f.stat']`** returns a Series, so
   formatting it with `:.1f` raises.
5. **Literal `...` inside Stata code** for a 10-period moving average — not runnable.
6. **`pyhdfe` row misalignment.** `drop_singletons=True` (the default) removes
   rows, so a cluster vector taken from the original frame silently misaligns.
   Appeared twice, once using a private attribute that does not exist.
7. **`model.matrix(~ post*post, df)`** — `post*post` collapses to `post`;
   replaced with a real diff-in-discontinuities estimator.
8. **Stata event-study loop omitted `k = 0`** instead of `k = −1`, normalising
   the treatment period itself.
9. **`compute_car` was a stub** containing `...`; replaced with a working
   implementation including estimation-window guards.
10. **Randomisation-inference p-value missing the `+1`**, allowing an exact 0.
11. **Post-match OLS standard errors** presented as valid; now flagged with
    Abadie-Imbens.

### Fixed — outdated ecosystem guidance

The skill repeatedly told users to leave Python. Every one of these is now wrong:

| Was | Now |
|---|---|
| "Bacon decomposition: Python **None**" | `sp.bacon_decomposition` |
| "Callaway-Sant'Anna: Python **None**" | `sp.callaway_santanna` |
| "Sun-Abraham: Python **None**" | `sp.sun_abraham` |
| "McCrary: Python **None**, use R `rdd`" | `sp.rddensity`, `sp.mccrary_test` |
| "CEM: Python **None**" | `sp.match(method="cem")` |
| "DAGs: Python **None**" | `sp.dag`, `sp.check_identification`, dowhy |
| "Synthetic control: rpy2 only" | `sp.synth`, `sp.sdid`, scpi, pysyncon |

Also resolved a three-way contradiction: `SKILL.md` said Python RDD was
"statsmodels polynomial", `README.md` said McCrary needed R, and
`r-stata-comparison.md` listed `rddensity (py)`. Retired `Zelig` (archived from
CRAN), `rdd::DCdensity`, and `lfe` in favour of maintained packages.

### Added — methods the skill was missing

Honest DiD (Rambachan-Roth), pre-trend power (Roth 2022), wild cluster
bootstrap, small-cluster guidance, two-way and Conley clustering,
Borusyak-Jaravel-Spiess imputation, Anderson-Rubin CIs, and the Lee et al. (2022)
104.7 result for its applicable single-IV t-ratio setting, plus Abadie-Imbens
bias-corrected matching, the forbidden
regression, manual-2SLS standard errors, judge/examiner leniency designs, Nickell
bias, and multiple-testing corrections.

### Changed — attribution and compliance

`identification-writing-patterns.md` §4 previously reproduced verbatim paragraphs
from eight copyrighted Journal of Finance articles under an MIT licence. Replaced
with paraphrased descriptions of each paper's rhetorical *move* plus an adaptable
template. README now states the source repository, that datasets are loaded at
runtime rather than redistributed, the citation for the book, and that the MIT
licence covers only material authored here.

### Fixed — internal consistency

README file structure was three files and one prompt out of date; the trigger
list omitted every finance phrase; the clone URL is now correct.

---

## v2.1.0

Finance applications and identification writing patterns: 25 JF 2021–2024
case sketches, five end-to-end worked examples, verb-framing guide.

## v2.0.0

Initial release: 10 identification strategies in Python, R and Stata.
