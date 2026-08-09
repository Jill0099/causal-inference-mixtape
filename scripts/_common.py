"""Shared helpers for the runnable Mixtape templates.

Every script in ``scripts/`` imports from here so that all templates load data
the same way and print results in a comparable format.

Data convention follows the book itself.  *Causal Inference: The Mixtape* ships
its datasets from a public GitHub raw URL and the book's own R helper is::

    read_data <- function(df)
      read_data <- haven::read_dta(paste0(
        "https://raw.github.com/scunning1975/mixtape/", df))

``read_mixtape`` below is the Python equivalent, with a local disk cache so a
validation sweep does not re-download the same file a dozen times.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MIXTAPE_BASE = "https://raw.github.com/scunning1975/mixtape/master/"

#: Where downloaded ``.dta`` files are cached.  Override with ``MIXTAPE_CACHE``.
CACHE_DIR = Path(os.environ.get("MIXTAPE_CACHE", Path.home() / ".cache" / "mixtape-data"))

#: Datasets used by the templates in this repository, with the book chapter that
#: introduces each one.  ``close_college.dta`` is deliberately absent: it 404s on
#: the public mirror, so the IV templates use ``card.dta`` instead.
DATASETS = {
    "yule.dta": "Ch.2  Yule (1899) pauperism regression",
    "training_example.dta": "Ch.4  potential outcomes / selection bias toy data",
    "nsw_mixtape.dta": "Ch.5  LaLonde NSW experimental sample",
    "cps_mixtape.dta": "Ch.5  CPS non-experimental control pool",
    "lmb-data.dta": "Ch.6  Lee-Moretti-Butler RDD (US House)",
    "card.dta": "Ch.7  Card (1995) college-proximity IV",
    "judge_fe.dta": "Ch.7  judge leniency / examiner design",
    "castle.dta": "Ch.9  castle-doctrine staggered DiD",
    "abortion.dta": "Ch.9  Donohue-Levitt abortion DDD",
    "texas.dta": "Ch.10 Texas prison-capacity synthetic control",
    "thornton_hiv.dta": "Ch.4  Thornton HIV-testing RCT",
}

SEED = 20060815  # fixed everywhere so validation output is byte-stable


def read_mixtape(name: str, *, refresh: bool = False) -> pd.DataFrame:
    """Load a Mixtape dataset, caching the download under ``CACHE_DIR``.

    Parameters
    ----------
    name
        File name as it appears in the Mixtape repo, e.g. ``"castle.dta"``.
    refresh
        Re-download even when a cached copy exists.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / name
    if refresh or not local.exists():
        # Cache the raw .dta bytes rather than a converted parquet: several
        # Mixtape files carry Stata value labels that become mixed-type pandas
        # categoricals, which arrow refuses to serialise.
        import urllib.request

        with urllib.request.urlopen(MIXTAPE_BASE + name) as resp:
            payload = resp.read()
        local.write_bytes(payload)
    return pd.read_stata(local)


# --------------------------------------------------------------------------
# Backend availability
# --------------------------------------------------------------------------

def have(module: str) -> bool:
    """True when ``module`` is importable.  Templates degrade instead of crashing."""
    try:
        __import__(module)
        return True
    except Exception:
        return False


def backend_report() -> pd.DataFrame:
    """Version table for every backend the templates can use."""
    import importlib.metadata as md

    rows = []
    for pkg in [
        "statspai",
        "pyfixest",
        "statsmodels",
        "linearmodels",
        "pyhdfe",
        "rdrobust",
        "differences",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
    ]:
        try:
            rows.append((pkg, md.version(pkg)))
        except Exception:
            rows.append((pkg, "not installed"))
    return pd.DataFrame(rows, columns=["package", "version"])


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------

def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 72 - len(title)))


def report(label: str, coef: float, se: float | None = None, extra: str = "") -> None:
    """One-line estimate report.  Keeps every script's output diff-able."""
    if se is None or not np.isfinite(se) or se == 0:
        print(f"  {label:<44} {coef: .5f}{'  ' + extra if extra else ''}")
        return
    t = coef / se
    print(f"  {label:<44} {coef: .5f}  (se {se:.5f}, t {t:6.2f}){'  ' + extra if extra else ''}")


def design_facts(
    df: pd.DataFrame,
    *,
    outcome: str,
    treat_mask: pd.Series | None = None,
    cluster: str | None = None,
) -> None:
    """Print the four numbers every empirical table should carry alongside a beta.

    Reviewers ask for these constantly and they are trivially cheap to produce:
    N, number of clusters, the treated share, and the untreated baseline mean
    that turns a coefficient into an economic magnitude.
    """
    print(f"  N observations                               {len(df):,}")
    if cluster is not None and cluster in df.columns:
        g = df[cluster].nunique()
        flag = "  <-- FEW CLUSTERS: use wild cluster bootstrap" if g < 42 else ""
        print(f"  N clusters ({cluster})".ljust(46) + f"{g:,}{flag}")
    if treat_mask is not None:
        base = df.loc[~treat_mask.astype(bool), outcome].mean()
        print(f"  Treated share                                {treat_mask.mean():.3f}")
        print(f"  Untreated baseline mean of {outcome}".ljust(46) + f"{base: .5f}")


def pick(tidy: pd.DataFrame, term: str) -> tuple[float, float]:
    """Return ``(estimate, std_error)`` for ``term`` from any tidy frame.

    The Python ecosystem has not converged on one tidy schema.  ``pyfixest``
    returns an index of coefficient names with ``Estimate`` / ``Std. Error``
    columns; ``StatsPAI`` follows R's broom with ``term`` / ``estimate`` /
    ``std_error`` columns.  Templates call this instead of hard-coding either.
    """
    frame = tidy
    if "term" in frame.columns:
        frame = frame.set_index("term")
    est_col = next((c for c in ("Estimate", "estimate", "coef") if c in frame.columns), None)
    se_col = next(
        (c for c in ("Std. Error", "std_error", "std.error", "se") if c in frame.columns), None
    )
    if est_col is None or se_col is None:
        raise KeyError(f"cannot locate estimate/se columns in {list(frame.columns)}")
    return float(frame.loc[term, est_col]), float(frame.loc[term, se_col])


class Timer:
    """Context manager used by ``validate_all.py`` to time each template."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False


def require(condition: bool, message: str) -> None:
    """Assertion that prints a readable failure instead of a bare traceback."""
    if not condition:
        print(f"  !! CHECK FAILED: {message}", file=sys.stderr)
        raise AssertionError(message)
