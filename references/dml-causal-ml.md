# DML & ML-Based Causal Inference — Concepts, Model Classes, and the DeDL Frontier

Companion to `method-patterns.md` §15. This file carries the *when-to-use* reasoning, the three DML model classes, cross-fitting mechanics, heterogeneous-effect estimators, and the DeDL (Debiased Deep Learning) frontier extension. Code templates live in §15; this file is the concept layer.

**Provenance**: adapted from Renyu (Philip) Zhang's public
[`AI-PhD-S26`](https://github.com/rphilipzhang/AI-PhD-S26) course repository
(DOTE 6635) and Ye, Zhang, Zhang, Zhang & Zhang's DeDL article
([DOI: 10.1287/mnsc.2024.04625](https://doi.org/10.1287/mnsc.2024.04625)).

---

## When to reach for DML instead of §1–§14

For the PLR/IRM/HTE templates covered here, reach for DML when **both** hold:
1. Identification is by **selection-on-observables** (unconfoundedness) — there is no instrument, discontinuity, or clean natural experiment. If a design in §2–§7 (DiD/IV/RDD/synthetic control) is available, prefer it; DML does not manufacture exogenous variation.
2. The confounders `X` are **high-dimensional or enter nonlinearly**, so `y ~ treat + X` with hand-picked controls would mis-specify the nuisance. In the PLR partialling-out score, DML lets flexible ML estimate `ℓ₀(X)=E[Y|X]` and `m₀(X)=E[D|X]` while targeting a low-dimensional causal parameter under the required assumptions.

If an outcome, treatment, or control is itself ML-generated (for example, an
LLM sentiment score), use sample splitting so the generated variable is built
out of sample and analyse its measurement error explicitly. DML does **not** by
itself correct arbitrary classical or non-classical measurement error; that
requires assumptions and methods tailored to the generated regressor.

---

## The core idea in three sentences

Naively regularizing `X` (to tame its dimension) biases the treatment coefficient, because the regularizer trades variance for bias and that bias leaks into `theta`. DML controls the leak two ways at once: (a) a **Neyman-orthogonal moment** whose derivative w.r.t. the nuisance is zero at the truth, so small nuisance errors are second-order; and (b) **cross-fitting** — predict each observation's nuisance from a model trained on *other* folds — which limits own-observation overfitting. Under the required nuisance-rate, moment, and sampling conditions, the result is a √n-consistent, asymptotically normal `theta` even though ML was used inside.

Cross-fitting mechanics for the PLR partialling-out score: split into K folds; for each fold k, train `ℓ` and `m` on the other K−1 folds and predict on k; form residuals `Ỹ = Y − ℓ̂(X)` and `D̃ = D − m̂(X)`; estimate `theta` from the residual-on-residual moment; aggregate over folds and, when using repeated sample splitting, over repetitions.

---

## Three model classes (map to §15 code)

| Class | Model | Parameter | Use when | §15 template |
|-------|-------|-----------|----------|--------------|
| **PLR** — Partially Linear | `Y = θ·D + g(X) + ε`, `D = m(X) + v` | constant partial effect `θ` | effect assumed homogeneous; D continuous or binary | `DoubleMLPLR` / `ddml partial` |
| **IRM** — Interactive | fully interacts D with X via a doubly-robust (AIPW) score | ATE / ATTE | binary D, want the doubly-robust average effect | `DoubleMLIRM(score='ATE')` |
| **HTE** — Heterogeneous | `τ(x) = E[Y¹−Y⁰ \| X=x]` via forests/DML | the effect *function* | targeting, personalization, effect heterogeneity | `CausalForestDML` / `grf::causal_forest` |

**Causal Forest / honest trees**: a random forest adapted to estimate `τ(x)` rather than `E[Y|X]`, using *honesty* — one subsample chooses splits, a disjoint subsample estimates the leaf effect — to support inference under the forest's regularity conditions. Athey & Imbens (2016); Wager & Athey (2018). Pair with `test_calibration()` (grf) as an omnibus calibration check before interpreting `τ(x)`.

---

## The DeDL frontier (the course's flagship, and the point of the demo)

**DeDL = Debiased Deep Learning** (Ye, Zhang, Zhang, Zhang & Zhang, published online 2025, *Management Science*, "Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments"). The problem: online platforms run experiments over a combinatorial space of treatment combinations (2^k cells) — most combinations are never directly tested, so their causal effects cannot be read off a simple difference in means. DeDL trains a **DNN as the outcome nuisance** over the treatment/feature space, then applies a **doubly-robust (orthogonalized) correction** using known randomization propensities, yielding effect estimates for treatment combinations under the paper's structural and experimental assumptions.

**A practical lesson from the DeDL evidence and accompanying course exercise:**
- When the first-stage DNN is **poorly trained** (high validation MSE, early epochs), DeDL and the plug-in single-deep-learner (SDL) are *worse* than plain linear regression — debiasing a bad nuisance does not help.
- As the DNN **converges** (validation MSE shrinks), DeDL's debiasing kicks in and it beats both SDL and LR on treatment-effect MAPE.
- **Practical rule**: monitor out-of-fold nuisance error as one diagnostic of nuisance quality and report it alongside `theta`. Good predictive performance is not, by itself, proof of the product-rate and regularity conditions needed for DML inference. (See `method-patterns.md` §15 robustness item 3.)

---

## Positioning within this skill

- DML is an **estimation** upgrade for selection-on-observables settings; it is orthogonal to the **identification** designs in §2–§7. State the identifying assumption (unconfoundedness) exactly as forcefully as for any observational design — see `identification-writing-patterns.md` for the causal-vs-descriptive verb guide. Do not let "we used double machine learning" stand in for an identification argument; a referee will read it as estimation hygiene, not exogenous variation.
- For finance/marketing applications with rich firm/customer covariates and no clean shock, DML is a candidate only when unconfoundedness and overlap are credible. Use PLR for a homogeneous partial effect or Causal Forest for heterogeneity, and combine it with sensitivity analysis and the finance robustness norms in `finance-applications.md`.

## References

- Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *Econometrics Journal* 21(1).
- Wager & Athey (2018); Athey & Imbens (2016) — causal forests / honest trees.
- Ye, Zhang, Zhang, Zhang & Zhang (2025). Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments. *Management Science*. https://doi.org/10.1287/mnsc.2024.04625
- Chernozhukov, Hansen, Kallus, Spindler & Syrgkanis — *Applied Causal Inference Powered by ML and AI* (causalml-book.org). Stanford GSB SI Lab ML-CI tutorial.
- Packages: `DoubleML` (Python & R, docs.doubleml.org), `EconML` (Python), `grf` (R), `ddml` + `pystacked` and `pdslasso` (Stata).
