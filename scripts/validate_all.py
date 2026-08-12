"""Run every template in this directory and report PASS / FAIL.

This is what makes the repository's code claims checkable rather than aspirational.
Each script asserts its own invariants via ``_common.require`` -- cross-backend
agreement, algebraic identities, design sanity checks -- so a green sweep means
the templates actually ran against real data and produced the numbers the
documentation quotes.

Usage
-----
    python validate_all.py                 # run everything
    python validate_all.py 03 04           # run only matching scripts
    python validate_all.py --list          # show what would run
    python validate_all.py --quick         # skip the slow ones

Exit status is 0 only if every selected script passed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Scripts that take appreciably longer than the rest; skipped under --quick.
SLOW = {"07_synthetic_control.py"}


def discover() -> list[Path]:
    top = sorted(p for p in HERE.glob("*.py") if p.name[0].isdigit())
    sims = sorted((HERE / "simulations").glob("sim_*.py"))
    return top + sims


def main(argv: list[str]) -> int:
    quick = "--quick" in argv
    listing = "--list" in argv
    filters = [a for a in argv if not a.startswith("--")]

    scripts = discover()
    if filters:
        scripts = [s for s in scripts if any(f in s.name for f in filters)]
    if quick:
        scripts = [s for s in scripts if s.name not in SLOW]

    if listing:
        for s in scripts:
            tag = "  [slow]" if s.name in SLOW else ""
            print(f"{s.relative_to(HERE)}{tag}")
        return 0

    print("=" * 78)
    print(f"Validating {len(scripts)} template scripts with {sys.executable}")
    print("=" * 78)

    results: list[tuple[str, bool, float, str]] = []
    for path in scripts:
        label = str(path.relative_to(HERE))
        sys.stdout.write(f"  {label:<48} ")
        sys.stdout.flush()
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            cwd=path.parent,
        )
        elapsed = time.perf_counter() - t0
        ok = proc.returncode == 0
        # Surface the assertion message rather than the whole traceback.
        detail = ""
        if not ok:
            for line in reversed(proc.stderr.splitlines()):
                if line.strip() and not line.startswith(("  File", "    ", "Traceback")):
                    detail = line.strip()[:100]
                    break
        print(f"{'PASS' if ok else 'FAIL':>4}  {elapsed:6.1f}s  {detail}")
        results.append((label, ok, elapsed, detail))

    n_pass = sum(1 for _, ok, _, _ in results if ok)
    total = sum(t for _, _, t, _ in results)
    print("-" * 78)
    print(f"  {n_pass}/{len(results)} passed in {total:.1f}s")
    if n_pass != len(results):
        print("\n  Failures:")
        for label, ok, _, detail in results:
            if not ok:
                print(f"    {label}: {detail}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
