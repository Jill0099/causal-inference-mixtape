# Causal Inference: The Mixtape — Claude Code Skill

A [Claude Code](https://claude.ai/code) skill for applied causal inference:
route from the assignment mechanism to a design, implement it, run the
diagnostics that make it publishable, and write the identification section
without over-claiming.

Built from Scott Cunningham's [*Causal Inference: The
Mixtape*](https://mixtape.scunning.com/), extended with 25 Journal of Finance
(2021–2024) applications.

**Languages:** Python (StatsPAI) · R · Stata

---

## What makes v3 different

**The code is executed, not asserted.** Thirteen runnable scripts in
[`scripts/`](scripts/) load the Mixtape's own public datasets, estimate, and
assert their own invariants — cross-backend agreement, algebraic identities,
design sanity checks.

```bash
$ python scripts/validate_all.py
  01_potential_outcomes.py         PASS    1.0s
  02_did_2x2.py                    PASS    9.3s
  03_event_study.py                PASS    9.7s
  04_staggered_did.py              PASS   27.7s
  05_rdd.py                        PASS    2.8s
  06_iv_2sls.py                    PASS    1.7s
  07_synthetic_control.py          PASS   12.9s
  08_matching_ipw.py               PASS   14.8s
  09_randomization_inference.py    PASS    2.7s
  10_inference_clusters.py         PASS   10.6s
  11_honest_did.py                 PASS    1.0s
  simulations/sim_collider_bias.py           PASS  0.9s
  simulations/sim_twfe_staggered_bias.py     PASS  8.5s
  13/13 passed in 103.8s
```

Selected results the scripts verify:

| Check | Result |
|---|---|
| 2×2 DiD on `castle.dta`: hand-computed / statsmodels / pyfixest / StatsPAI | all **0.06824** |
| Event study: hand-built binned dummies vs `sp.event_study` | agree to **~1e-16** |
| Bacon decomposition weighted sum vs TWFE | both **+0.08181**, weights sum to 1.0000 |
| Simulated staggered panel, true ATT **6.832** | TWFE **3.173** (−54%), Callaway-Sant'Anna **6.878** |
| NSW experimental benchmark | **$1,794.34**, matching the published figure |
| `training_example.dta` subclassification | raw **−$26.25**, age-matched **+$1,695** (the book's numbers) |
| Potential-outcomes decomposition identity residual | **2.7e-15** |
| Wild cluster bootstrap: StatsPAI vs pyfixest | p **0.5733** vs **0.5693** |

---

## Python: StatsPAI is the default backend

```bash
pip install statspai
```

[StatsPAI](https://github.com/brycewang-stanford/statspai) covers the Stata/R
causal-inference surface natively. **The "Python can't do X, switch to R"
advice in older cheatsheets is out of date** — every method previously listed as
a Python gap now has a native implementation:

| Old claim | Reality |
|---|---|
| "Bacon decomposition: R or Stata only" | `sp.bacon_decomposition` |
| "Callaway-Sant'Anna: R or Stata only" | `sp.callaway_santanna` + `sp.aggte` |
| "Sun-Abraham: R or Stata only" | `sp.sun_abraham` |
| "McCrary test: R only" | `sp.rddensity`, `sp.mccrary_test` |
| "Synthetic control: use rpy2" | `sp.synth`, `sp.sdid` (also `scpi`, `pysyncon`) |
| "CEM: Stata or R only" | `sp.match(method="cem")` |
| "DAGs: R only" | `sp.dag`, `sp.check_identification` (also `dowhy`) |
| "Wild cluster bootstrap: shell out to `boottest`" | `sp.wild_cluster_bootstrap` |
| "Honest DiD: R only" | `sp.honest_did`, `sp.pretrends_power` |

See [`references/statspai-guide.md`](references/statspai-guide.md) for the API
map, the recommended workflow, and **eleven non-obvious gotchas found by
actually running it** — including one where a single keyword argument moves an
estimate by $9,400 against a known truth.

Fallbacks: `pyfixest` → `linearmodels` → `statsmodels`. R and Stata when a
referee asks for a specific package.

---

## Start from the design, not the method

Users describe a setting, not an estimator:

```
How was treatment assigned?
├─ Randomised                          -> diff in means + randomisation inference
├─ Threshold on a running variable     -> RDD / fuzzy / diff-in-discontinuities
├─ Policy date, one date for all       -> 2x2 DiD -> event study
├─ Policy date, dates VARY             -> staggered: Bacon -> CS / SA / BJS
├─ One treated unit, many controls     -> synthetic control
├─ Self-selected + an instrument       -> IV / 2SLS, identifies a LATE
├─ Self-selected + rich observables    -> matching / IPW / DML
└─ None of the above                   -> say so; use associational language
```

Full tree with estimator selection inside each design:
[`references/design-router.md`](references/design-router.md).

---

## Methods Covered

| Method | Python | R | Stata | Script |
|---|---|---|---|---|
| Potential outcomes / selection bias | — | — | — | `01` |
| DiD (2×2) | `sp.feols` | fixest | reghdfe | `02` |
| Event study | `sp.event_study` | fixest `i()` | reghdfe | `03` |
| Staggered DiD + Bacon | `sp.callaway_santanna`, `sp.bacon_decomposition` | did, bacondecomp | csdid | `04` |
| Regression discontinuity | `sp.rdrobust`, `sp.rddensity` | rdrobust | rdrobust | `05` |
| Instrumental variables | `sp.ivreg` | AER, fixest | ivregress | `06` |
| Synthetic control / SDiD | `sp.synth`, `sp.sdid` | Synth, synthdid | synth, sdid | `07` |
| Matching / PSM / IPW | `sp.match`, `sp.ipw` | MatchIt | teffects, cem | `08` |
| Randomisation inference | `sp.ri_test` | ri2 | ritest | `09` |
| Clustered inference / wild bootstrap | `sp.wild_cluster_bootstrap` | fwildclusterboot | boottest | `10` |
| Honest DiD / pre-trend power | `sp.honest_did` | HonestDiD | honestdid | `11` |
| DAGs / collider bias | `sp.dag`, dowhy | dagitty | — | `sim_collider_bias` |

---

## Trigger Phrases

- `implement a DiD regression` · `run a staggered difference-in-differences`
- `set up an event study` · `test parallel trends` · `run Honest DiD`
- `implement instrumental variables` · `run a regression discontinuity design`
- `build a synthetic control model` · `implement propensity score matching`
- `implement Bacon decomposition` · `wild cluster bootstrap`
- `which causal design fits my data`

---

## Installation

```bash
git clone https://github.com/Jill0099/causal-inference-mixtape.git \
  ~/.claude/skills/causal-inference-mixtape
```

To run the validation scripts:

```bash
pip install -r scripts/requirements.txt
python scripts/validate_all.py
```

---

## File Structure

```
causal-inference-mixtape/
├── SKILL.md                                  # Core skill (auto-loaded when triggered)
├── references/
│   ├── design-router.md                      # Assignment mechanism -> design -> estimator
│   ├── statspai-guide.md                     # The Python backend: API, workflow, gotchas
│   ├── method-patterns.md                    # Code templates, Python / R / Stata
│   ├── mixtape-core.md                       # Potential outcomes, DAGs, LATE, panel FE
│   ├── inference-and-standard-errors.md      # Clustering, bootstrap, Honest DiD, RI
│   ├── reporting-checklist.md                # What to print with every estimate
│   ├── r-stata-comparison.md                 # Cross-language coverage matrix
│   ├── finance-applications.md               # 25 JF papers indexed by method
│   ├── jf-case-studies.md                    # Five end-to-end finance examples
│   └── identification-writing-patterns.md    # Identification paragraphs, verb guide
├── scripts/
│   ├── _common.py                            # Mixtape data loader, backend helpers
│   ├── 01_potential_outcomes.py ... 11_honest_did.py
│   ├── simulations/                          # Collider bias, staggered TWFE bias
│   ├── requirements.txt
│   └── validate_all.py                       # Run everything, report PASS / FAIL
└── prompts/
    ├── 01-implement-method.md
    ├── 02-robustness-checks.md
    ├── 03-finance-application.md
    └── 04-statspai-workflow.md
```

---

## Sources and attribution

Method templates and datasets derive from Scott Cunningham's
[*Causal Inference: The Mixtape*](https://mixtape.scunning.com/) and its
[companion code repository](https://github.com/scunning1975/mixtape)
(58 Python scripts, ~56 R scripts, ~60 Stata `.do` files). Datasets are loaded
at runtime from that public repository; none are redistributed here. Please cite
the book:

> Cunningham, Scott (2021). *Causal Inference: The Mixtape.* Yale University Press.

The finance material summarises the identification strategies of published
Journal of Finance articles (2021–2024) and paraphrases their rhetorical
structure for teaching purposes. No article text is reproduced; consult the
originals, which remain under their publishers' copyright.

## License

MIT — applies to the templates, scripts and documentation authored in this
repository, not to the underlying book, datasets, or cited articles.
