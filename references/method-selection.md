# Method Selection — 什么时候用什么方法

Start from **your data situation**, not from a method you like. This guide maps a research setting → the identifying assumption it can support → which method (`method-patterns.md` §) → the check that challenges it → the Stata command and frontier reference. The public URLs for the Chinese-language 连享会 tutorials are collected in [`lianxh-stata-index.md`](lianxh-stata-index.md); no local archive paths are required.

**The rule that governs everything below**: the method does not create identification — the *data situation* does. Every estimator is only as good as the exogenous variation it exploits. DML/matching fix *estimation*, not *identification*.

---

## Gate 0 — before choosing a design

### 0a. What kind of endogeneity do you have? (内生性四来源 → remedy)
Diagnose the *source* first; the remedy follows from it (Chinese overview:
[内生性之应对（下）](https://www.lianxh.cn/details/1266.html)).

| Source of endogeneity | Typical remedy | § / command |
|---|---|---|
| **遗漏变量** omitted variable | proxy variable · fixed effects (within-group only) · IV | §1 FE `reghdfe` · §6 IV |
| **反向因果** reverse causality | IV; lagged regressor only if not serially correlated | §6 `ivregress`, `ivreghdfe` |
| **测量误差** measurement error | errors-in-variables / SEM; test with `dgmtest` | `eivreg`, `sem` |
| **自选择** self-selection | Heckman · two-part model · endogenous switching | `heckman`, `twopm`, `movestay` |

Fixed effects only remove *time-invariant within-group* confounders and identify *within*-group (not between-group) effects — state that explicitly (lianxh: `用FE能做因果推断吗?`).

### 0b. Is your policy shock even usable? (准自然实验的四个标准)
Before DiD/event-study/IV, a shock must pass four clarity tests. These are **necessary, not sufficient**:
1. **政策事实清楚** — did the shock really happen? who triggered it, when announced/implemented, verifiable by a list/batch/document?
2. **处理边界清楚** — who is the target? does the policy's target unit match your data unit, and does Treat/Control come from the *policy rule* rather than your ex-post choice?
3. **冲击时间清楚** — which of announcement / pilot / local-execution / name-list timing defines `Post`? (China's 中央发文→地方试点→名单→执行 chain makes this decisive.)
4. **数据合并键清楚** — how does the policy target map onto sample units?

If a shock can't be turned into a clean `Treat`/`Post`/`Exposure`, it stays *background*, not *identification*.

---

## Master decision tree

```
Q1. Is there a treatment / policy / shock (that passes Gate 0b)?
├── YES → Q2. Do all treated units get treated at the SAME time?
│         ├── YES → Difference-in-Differences (§2) + Event study (§3)
│         └── NO (staggered) → Callaway–Sant'Anna csdid / did2s (§4)
│                              ⚠ plain TWFE = negative-weight bias
│
├── Assignment by a CUTOFF on a running variable? → RDD (§5)
│      └── cutoff pre-existed, want a later policy's marginal effect → diff-in-disc (§11)
│
├── Endogenous regressor + external instrument? → IV / 2SLS (§6)
│      └── instrument = local shares × national shocks → Shift-Share / Bartik (§12)
│
├── ONE (or few) treated unit + long pre-period + donor pool? → Synthetic control (§7)
│
└── NO clean shock — only observational variation:
       ├── low-dim confounders, want ATE → Matching / PSM / IPW / IPWRA (§8)
       ├── high-dim / nonlinear confounders → DML (§15)
       ├── want τ(x) / heterogeneity → Causal forest, DML-HTE (§15)
       └── panel, unobserved time-varying heterogeneity → fect / gsynth / CIC (§7-adjacent)
```

Treat the **first plausible** branch as a candidate, then compare all designs the
setting can support and choose the one whose assignment mechanism and assumptions
you can defend. Reaching the bottom (matching/DML) is not failure — it is an
honest signal you lack exogenous variation and must defend unconfoundedness
(Gate 0a) and run a sensitivity analysis (see "Selection-on-observables" below).

---

## Finding a plausibly-exogenous shock (external catalog)

Before settling for the observational branch, check whether an exogenous shock already exists for a setting like yours. Sangmin Oh's public ["Plausibly Exogenous Galore"](https://sangmino.notion.site/1a897b8106ca44eeaf31dcd5ae5a61b1?v=ff7dc75862c6427eb4243e91836e077e) catalog is one starting point. Treat it as an idea index, then verify the original paper and whether the assignment mechanism transfers to your setting.

**How to use it (类比应用 — reason by analogy):**
1. Search the catalog by the *kind* of shock your setting might offer — e.g. `regulation`, `pollution`, `weather`, `lottery`, `threshold`, `border`, `reform`, `China`.
2. Find an entry whose **Source of exogenous variation** is structurally similar to your setting, and read its LHS/RHS to see how it was used.
3. Bring the shock's *shape* back to the decision tree above: discrete cross-unit policy → DiD/event-study (§2–§3); staggered pilots → `csdid` (§4); assignment threshold → RDD (§5); shock used as an instrument → IV (§6); local-shares × national-shock → Shift-Share (§12).

The catalog answers *"is there an exogenous shock for my question, and who used it?"*; this guide answers *"given that shock's shape, which estimator?"* — they compose.

---

## Per-method cards (grounded in the archive)

Each: **Use when · Identifying assumption · What kills it · Stata command · Frontier reference · lianxh source**.

### DiD (§2) & Event study (§3)
- **Use when**: common treatment date, treated + control observed before/after.
- **Assumption**: parallel trends (untestable — you only test *pre*-trends).
- **What kills it**: differential pre-trends, anticipation, concurrent shock.
- **Frontier discipline** (Roth 2022; Rambachan–Roth 2023): pre-trend *tests have low power* — non-significant ≠ parallel. Report a **sensitivity** analysis (`honestdid`) and `pretrends` power, not just the pre-trend plot.
- **Commands**: `reghdfe`, `eventdd`/`eventcoefplot`, `honestdid`, `pretrends`.

### Staggered DiD (§4) — the sub-decision that trips most papers
- **Use when**: units adopt at **different** times.
- **What kills it**: plain TWFE under heterogeneous effects → **negative weights** (Goodman-Bacon 2021; de Chaisemartin–d'Haultfœuille 2020) — the coefficient can even flip sign.

  | Situation | Estimator | Command |
  |---|---|---|
  | Diagnose how bad TWFE is | Goodman-Bacon decomposition | `bacondecomp` |
  | Cohort-time ATT, clean controls, doubly robust | Callaway–Sant'Anna | `csdid` (R `did`) |
  | Fast two-stage / imputation | Gardner | `did2s` |
  | Continuous / general treatment, negative-weight diagnostic | de Chaisemartin | `did_multiplegt` |
  | Uncontaminated event-study coefficients | Sun–Abraham interaction-weighted | `eventstudyinteract` |
- **Chinese tutorials**: see the staggered-DiD entries in `lianxh-stata-index.md`.

### RDD (§5)
- **Use when**: treatment assigned by crossing a threshold on a continuous running variable.
- **Assumption**: no precise manipulation at the cutoff → local as-good-as-random.
- **What kills it**: bunching/manipulation; another policy switching at the same cutoff; **high-order polynomials** (Gelman–Imbens 2019 — use local *linear*, not global cubic); discrete running variable (Kolesár–Rothe 2018).
- **Commands**: `rdrobust`, `rddensity`/McCrary, bandwidth h/2·2h robustness.

### IV / 2SLS (§6)
- **Use when**: endogenous regressor + variable that shifts it but affects Y only through it.
- **Assumption**: relevance + exclusion.
- **What challenges it**:
  - Under heterogeneity IV estimates **LATE, not ATE** (compliers only; Mogstad et al. 2021 — monotonicity is fragile with multiple instruments).
  - **F > 10 is not enough** (Keane–Neal 2023): 2SLS has power asymmetry even at large F → use the **Anderson–Rubin test** (`weakiv`, `weakivtest2`), which stays valid under weak IV; report over-ID (`Sargan`/`Hansen`).
  - Just-ID IV is usually reliable and sign-screening the first stage is a "free lunch" (Angrist–Kolesár 2024).
  - Nonlinear first stage / endogenous nonlinearity may require a **control function** with its own assumptions, not an automatic nonlinear first-stage substitution in plain 2SLS.
- **Commands**: `ivregress 2sls`, `ivreghdfe`, `weakivtest2`, `mivreg`.
- **Need an instrument?** Search the public exogenous-variation catalog above for an analogous assignment mechanism, then verify the original paper and defend why that mechanism transfers instead of inventing an instrument from scratch.

### Shift-Share / Bartik IV (§12)
- **Use when**: instrument = local industry shares × national shocks (labor, trade, public finance).
- **Two identification stories**: share-exogeneity (Goldsmith-Pinkham et al. 2020 — each share is an instrument, testable) **vs** shock-exogeneity (Borusyak et al. 2022 — shocks quasi-random, shares may be endogenous).
- **What challenges it**: correlated residuals across regions with similar industry mix can make naive cluster SEs over-reject. Use design-appropriate exposure-robust inference such as AKM procedures (`reg_ss`, `ivreg_ss`, `ssaggregate`) when their assumptions apply.

### Synthetic control (§7)
- **Use when**: one (few) treated unit, long pre-period, clean donor pool.
- **What kills it**: poor pre-treatment fit; no convex donor combination.
- **Commands**: `synth`, `synth_runner`, `synth2`; placebo (in-space/in-time) + leave-one-out. Panel extensions for time-varying heterogeneity include `gsynth`, `fect`, and Change-in-Change `cic`, each with distinct assumptions.

### Matching / PSM / IPW / IPWRA (§8)
- **Use when**: no shock; unconfoundedness conditional on *low-dim observed* covariates.
- **What kills it**: an unobserved confounder (matching cannot fix it); poor overlap.
- **Commands**: `psmatch2`, `teffects psmatch/ipw/ipwra`, `psestimate` for covariate selection; balance table + `pstest`; Rosenbaum bounds.

### DML & Causal ML (§15)
- **Use when**: unconfoundedness and overlap are credible, but observed confounders are high-dimensional/nonlinear.
- **Assumption**: unconfoundedness — DML fixes regularization/overfitting bias, **not** identification.
- **What kills it**: unobserved confounding, poor overlap, unstable nuisance fits, or invalid sample splitting. ML-generated variables require a separate measurement-error analysis; DML does not automatically correct them.
- **Commands**: `ddml` + `pystacked`, `pdslasso`; for dependent data, use cluster-aware folds and inference supported by the chosen implementation. Concept layer: [`dml-causal-ml.md`](dml-causal-ml.md).

---

## Selection-on-observables: bound what you can't fix
When you land in the matching/DML branch, you cannot rule out unobserved confounders — but you can **bound** their potential impact. This is the honest close that referees now expect (lianxh 敏感性分析 series):
- **Oster (2019)** δ / coefficient-stability bounds.
- **`regsensitivity`** — coefficient sensitivity when *control variables* themselves are endogenous.
- **`konfound`** — how strong must an omitted confounder be to overturn the result.
- **`tesensitivity`** — treatment-effect sensitivity to unmeasured confounding.

Report one of these whenever identification rests on selection-on-observables.

---

## Control-variable selection (good vs bad controls)
Adding controls is not free (Chinese overview:
[Good Controls and Bad Controls](https://www.lianxh.cn/details/1037.html); ties to §9 DAG):
- A **bad control** is a post-treatment variable or a collider — controlling for it *induces* bias rather than removing it.
- Rule: control for defensible pre-treatment confounders (common causes), not mediators or descendants of treatment. Draw the DAG (`§9`, `ggdag`) before choosing the control set and justify the adjustment set with the backdoor criterion.

---

## Selection pitfalls (extends SKILL.md § Common Pitfalls)
1. **Staggered timing + plain TWFE** — the #1 error. Use `csdid`/`did2s`; diagnose with `bacondecomp`.
2. **"Pre-trend is flat, so parallel trends holds"** — pre-tests have low power. Report `honestdid`/`pretrends`.
3. **"F > 10, so my IV is fine"** — insufficient (Keane–Neal 2023). Report Anderson–Rubin.
4. **"I ran DML, so it's causal"** — DML needs unconfoundedness exactly as much as PSM; add a sensitivity bound.
5. **Mechanical clustering rules** — start from the treatment-assignment and sampling design, then account for defensible residual dependence; do not cluster automatically at the highest FE level.
6. **Bad controls** — never control for post-treatment variables/colliders.
7. **Reaching for a fancier estimator to rescue a bad design** — no estimator manufactures exogenous variation. Fix the design or state the limitation.

---

*The Chinese tutorial routing is indexed in `lianxh-stata-index.md`. 连享会 content remains © 连享会, All Rights Reserved; this repository links to the original pages and does not reproduce their text. Method claims should be checked against the cited papers and current package documentation.*
