# Inference and Standard Errors

The applied literature has internalised "cluster your standard errors" and
stopped there. Most inference failures in published work happen *after* that
step. This file covers what comes next.

Runnable companions:
[`scripts/10_inference_clusters.py`](../scripts/10_inference_clusters.py),
[`scripts/11_honest_did.py`](../scripts/11_honest_did.py),
[`scripts/09_randomization_inference.py`](../scripts/09_randomization_inference.py).

---

## §1. Cluster at the level of treatment ASSIGNMENT

Abadie, Athey, Imbens & Wooldridge (2023): clustering is a design property, not
a data property. Cluster at the level at which treatment was **assigned**.

- **Finer than assignment → understates uncertainty.** On `castle.dta`,
  clustering by state-year instead of state shrinks the standard error by 57%
  and moves the t-statistic from 0.77 to 1.77.
- **Much coarser than assignment → throws away precision** with no design
  justification, and reintroduces the few-clusters problem below.
- **Nested clusters:** cluster at the coarsest level at which assignment varies.
- **Sampling vs assignment:** if you have the whole population and treatment was
  assigned at the unit level, clustering may not be needed at all — AAIW make
  this argument explicitly.

The instinct to "cluster on the finest identifier because it is safest" is
backwards.

---

## §2. Serial correlation (Bertrand, Duflo & Mullainathan 2004)

DiD on long panels with serially correlated outcomes and standard errors that
ignore the correlation produces **rejection rates around 45% at a nominal 5%
level**. The gap between iid and clustered SEs *is* that serial correlation made
visible: on `castle.dta`, clustering by state inflates the SE by 138% over iid.

Three remedies, in order of preference:

1. **Cluster on the unit.** Handles arbitrary within-unit correlation. This is
   the default and usually sufficient.
2. **Collapse to pre/post.** Average the outcome within unit before and after
   treatment, then run a two-period DiD. Crude, but the serial-correlation
   problem disappears by construction. Useful as a robustness row.
3. **Block bootstrap** at the unit level.

Do not use Newey-West on a short panel: it is a large-T device and you generally
do not have large T.

---

## §3. Too few clusters

Cluster-robust variance is consistent as the **number of clusters** grows, not
as the number of observations grows. With few effective clusters, the CRVE can
be downward biased and conventional reference distributions can over-reject. There is no
universal count at which clusters become "many": cluster-size imbalance,
leverage, treatment balance, and the number of treated clusters all affect the
quality of the approximation.

| Diagnostic situation | What to do |
|---|---|
| Balanced design, low leverage, many treated and control clusters | Report analytic CRVE and document the cluster count. |
| Approximation is doubtful | Add CR2/Satterthwaite or a null-imposed wild cluster bootstrap and report sensitivity. |
| Few treated clusters or severe imbalance | Treat conventional CRVE cautiously; inspect effective clusters and bootstrap support. |
| Known/random assignment mechanism | Randomisation inference may be used with permutations justified by that mechanism. |
| Rademacher support is small | There are only `2^G` sign patterns; enumerate when feasible or consider Webb weights. |

The count that matters is often the number of **treated** clusters, not the
total. Twenty control states and two treated states is a two-cluster problem.

### Wild cluster bootstrap

Cameron, Gelbach & Miller (2008): impose the null, resample Rademacher signs at
the cluster level, read the p-value off the bootstrap distribution.

```python
import statspai as sp

# NOTE: sp.wild_cluster_bootstrap takes a plain list of regressor COLUMNS and
# has no fixed-effect syntax.  Passing only your treatment variable silently
# estimates a model without FE.  Materialise the dummies and assert the result.
fe = pd.get_dummies(df[['unit', 'year']].astype(str), drop_first=True).astype(float)
boot_df = pd.concat([df[['y', 'treat_post', 'unit']].reset_index(drop=True),
                     fe.reset_index(drop=True)], axis=1)

out = sp.wild_cluster_bootstrap(
    data=boot_df, y='y', x=['treat_post', *fe.columns], cluster='unit',
    test_var='treat_post', h0=0.0, n_boot=1999, weight_type='rademacher', seed=0)
assert abs(out['beta_hat'] - analytic_beta) < 1e-6   # same spec, or it means nothing
```

```python
import pyfixest as pf   # independent implementation -- worth cross-checking
fit = pf.feols('y ~ treat_post | unit + year', data=df, vcov={'CRV1': 'unit'})
fit.wildboottest(param='treat_post', reps=1999, seed=0)
```

```stata
reghdfe y treat_post, absorb(unit year) cluster(unit)
boottest treat_post, reps(1999) weight(rademacher) bootcluster(unit)
```

Verified on `castle.dta`: `sp.wild_cluster_bootstrap` gives p = 0.5733 and
`pyfixest.wildboottest` gives p = 0.5693 — two independent implementations
agreeing, against an analytic CRVE p of 0.4436.

> **Bootstrap and permutation p-values need the `+1`.** The correct formula is
> `(1 + #{|t*| ≥ |t|}) / (1 + B)`. StatsPAI 1.21's `ri_test` and
> `wild_cluster_bootstrap` omit it and can return exactly `0.0`, which is not an
> attainable p-value. Report `p < 1/(1+B)`.

---

## §4. Two-way and spatial clustering

**Two-way clustering is not a free upgrade.** It requires *both* dimensions to
have many groups. On an 11-year panel, clustering on year is itself a
few-clusters problem — on `castle.dta` it produces an SE five times smaller than
state clustering and a t-statistic of 3.52 instead of 0.80. If one dimension is
small, use an appropriate small-sample correction and justify the assumed
dependence structure. Randomisation inference is an alternative only when the
assignment mechanism justifies the permutations.

**Conley spatial standard errors** when observations are correlated by
geographic distance rather than by group membership:

```python
sp.conley(result, data=df, lat='latitude', lon='longitude',
          dist_cutoff=100, kernel='uniform')     # cutoff in km
```

The cutoff is a researcher choice — report a sweep, not one value.

---

## §5. Parallel trends: sensitivity, not a pass/fail test

The standard defence — a flat pre-trend plot plus a joint F-test — is weaker
than it looks. Roth (2022):

- The pre-test has **low power** against exactly the violations that would
  overturn the result.
- **Conditioning on having passed** the pre-test distorts the sampling
  distribution of the estimate you then report.
- "We cannot reject parallel trends" is not "parallel trends holds."

On the castle-doctrine event study the individual pre-period test has power
**0.50** and the joint test **0.14** against the trend the design is powered to
detect. Convention wants 0.80. A plot that "looks flat" under that much noise is
close to uninformative.

### What to report instead

Rambachan & Roth (2023) replace the binary test with a sensitivity analysis:
assume the post-treatment violation is at most `M` times the largest violation
observed pre-treatment, then report the **breakdown M** at which the robust
confidence set first contains zero.

```python
cs  = sp.callaway_santanna(data=df, y='y', g='first_treat', t='year', i='unit')
dyn = sp.aggte(cs, type='dynamic', bstrap=False)

sp.pretrends_power(dyn, alpha=0.05)              # power of the conventional test
sp.honest_did(dyn, e=0, m_grid=[0, 0.25, 0.5, 1.0, 2.0],
              method='relative_magnitude')       # robust CI at each M
```

Reading the grid:

| M | Meaning |
|---|---|
| 0 | Exactly parallel trends — the conventional DiD assumption |
| 0.5 | Post-treatment violation up to half the worst pre-period one |
| 1 | As large as the worst pre-period violation |
| 2 | Twice that |

**Breakdown M below 1 means the result cannot survive a violation no larger than
one already visible in your own pre-period.** That is a fragile result no matter
how flat the plot looked. On `castle.dta` the breakdown sits between 0.00 and
0.25 — significant at M = 0 and not at M = 0.25.

Report it as a sentence: *"the estimate remains significant for violations of
parallel trends up to M times the largest pre-treatment violation."*

---

## §6. Randomisation inference

Tests the **Fisher sharp null** (`Y_i(1) = Y_i(0)` for every unit), not
Neyman's average null. Under the sharp null every potential outcome is known, so
the randomisation distribution is exact: valid in finite samples, no asymptotics,
no cluster-count requirement. That last point is why RI is the right tool
precisely where CRVE fails.

Two rules:

1. **Permute at the level treatment was assigned.** Permuting individuals when
   villages were randomised understates the null variance and manufactures
   significance.
2. **Include the observed assignment** in the reference set — the `+1` above.

```python
sp.ri_test(data=df, y='y', treat='treat', stat='diff_means',
           n_perms=10_000, cluster='village', seed=0)
```

```stata
ritest treat _b[treat], reps(10000) cluster(village): reg y treat
```

---

## §7. Multiple testing

Any paper that scans many outcomes, subgroups, or specifications needs a
correction, and which one depends on the dependence structure:

| Situation | Method | Python |
|---|---|---|
| Few independent tests | Bonferroni / Holm | `sp.holm`, `sp.bonferroni` |
| Many tests, arbitrary dependence | Benjamini-Yekutieli FDR | `sp.adjust_pvalues` |
| Many tests, positive dependence | Benjamini-Hochberg FDR | `sp.benjamini_hochberg` |
| Correlated outcomes, want FWER | Romano-Wolf stepdown (bootstrap) | `sp.romano_wolf` |
| Many related factors / hypotheses | Bayesian hierarchical shrinkage | `sp.bcf`, `sp.meta_analysis` |

Romano-Wolf is usually the right default for a table of correlated outcomes: it
controls family-wise error while exploiting the correlation, so it is far less
conservative than Bonferroni.

---

## §8. Checklist

- [ ] Cluster level matches the level of treatment assignment, and is stated
- [ ] Number of clusters **and number of treated clusters** reported
- [ ] Effective cluster count, imbalance and leverage assessed; small-sample
      correction reported when conventional cluster asymptotics are doubtful
- [ ] Randomisation inference used only with an explicit assignment mechanism
- [ ] Two-way clustering only if both dimensions have many groups
- [ ] DiD: event-study plot **plus** its power **plus** an Honest DiD breakdown M
- [ ] RDD: manipulation test before any estimate; robust bias-corrected CI
- [ ] IV: weak-IV diagnostics matched to the design; Lee et al. 104.7 invoked
      only for the applicable single-instrument t-ratio setting; Anderson-Rubin
      or other weak-IV-robust inference reported when needed
- [ ] Synthetic control: permutation rank, not a t-ratio
- [ ] Matching: Abadie-Imbens variance, not the post-match OLS default
- [ ] Multiple testing correction whenever more than a handful of tests are shown
- [ ] Permutation and bootstrap p-values use the `+1` correction
