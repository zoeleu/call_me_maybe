"""Score benchmark runs and compare them side by side.

Usage::

    python bench/score.py RUN_DIR [RUN_DIR ...] [--failures]

For each run directory and each ``<suite>.results.json`` in it, prints the
number of cases, errors, three accuracies, timing and forward passes:

* name accuracy: the function name is right;
* strict accuracy: name right and parameters exactly equal, *including the
  Python type* (``2 != 2.0``), because the real moulinette checks types
  with ``isinstance``;
* lenient accuracy: name right and parameters "the same value": numbers are
  compared as numbers (``2 == 2.0`` and ``"2019" == 2019``), strings after
  trimming, collapsing whitespace and ignoring case.

``--failures`` lists every case that is not strictly right and says the
first level it fails at: ``name``, ``lenient`` or ``strict-only``.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# --- comparison rules (pure functions, unit-tested) ------------------------


def strict_value_equal(expected: Any, got: Any) -> bool:
    """Return True if the values are equal and have the very same type.

    ``type(...) is`` rejects 2 vs 2.0 and also True vs 1 (bool is a
    subclass of int, so isinstance would accept it).
    """
    return type(expected) is type(got) and expected == got


def strict_parameters_equal(
    expected: dict[str, Any], got: dict[str, Any]
) -> bool:
    """Return True if both dicts have the same keys and strict-equal values."""
    if expected.keys() != got.keys():
        return False
    return all(strict_value_equal(expected[k], got[k]) for k in expected)


def as_number(value: Any) -> float | None:
    """Return ``value`` as a float if it is a number or a numeric string.

    bools are not numbers here. Strings are stripped first, so " 42 " is
    42.0. Returns None when ``value`` is not numeric.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def normalize_text(text: str) -> str:
    """Trim, collapse runs of whitespace to one space, and casefold."""
    return " ".join(text.split()).casefold()


def lenient_value_equal(expected: Any, got: Any) -> bool:
    """Return True if the two values mean the same thing.

    * If at least one side is a real number (int or float), both sides are
      compared as numbers, so 2 == 2.0 and "2019" == 2019. ``math.isclose``
      with a tiny relative tolerance absorbs float rounding noise.
    * Two strings are compared with :func:`normalize_text`.
    * A bool only matches the same bool (in Python, ``1 == True``).
    * Anything else must be ``==``.
    """
    if isinstance(expected, bool) or isinstance(got, bool):
        return strict_value_equal(expected, got)
    if isinstance(expected, (int, float)) or isinstance(got, (int, float)):
        left, right = as_number(expected), as_number(got)
        if left is None or right is None:
            return False
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=0.0)
    if isinstance(expected, str) and isinstance(got, str):
        return normalize_text(expected) == normalize_text(got)
    return bool(expected == got)


def lenient_parameters_equal(
    expected: dict[str, Any], got: dict[str, Any]
) -> bool:
    """Return True if both dicts have the same keys and lenient-equal values.

    Keys are compared exactly: a parameter name is never "almost right".
    """
    if expected.keys() != got.keys():
        return False
    return all(lenient_value_equal(expected[k], got[k]) for k in expected)


def failure_level(
    expected: dict[str, Any], got: dict[str, Any] | None
) -> str | None:
    """Classify one case; return None if it is strictly correct.

    ``expected`` and ``got`` are ``{"name": ..., "parameters": ...}``;
    ``got`` is None when the run raised an error. Otherwise returns the
    first level that fails: "name" (wrong name or no answer), "lenient"
    (right name, wrong values) or "strict-only" (right values, wrong types
    or formatting).
    """
    if got is None or got["name"] != expected["name"]:
        return "name"
    if not lenient_parameters_equal(expected["parameters"],
                                    got["parameters"]):
        return "lenient"
    if not strict_parameters_equal(expected["parameters"],
                                   got["parameters"]):
        return "strict-only"
    return None


# --- reading runs and printing tables --------------------------------------


def suite_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the numbers shown in the table for one suite of one run."""
    total = len(results)
    levels = [failure_level(r["expected"], r["got"]) for r in results]
    seconds = [float(r["seconds"]) for r in results]
    passes = [int(r["forward_passes"]) for r in results]

    def share(count: int) -> float:
        """Return ``count / total``, or 0 for an empty suite."""
        return count / total if total else 0.0

    return {
        "cases": total,
        "errors": sum(1 for r in results if r["error"] is not None),
        "name_acc": share(sum(1 for lv in levels if lv != "name")),
        "lenient_acc": share(sum(
            1 for lv in levels if lv in (None, "strict-only"))),
        "strict_acc": share(sum(1 for lv in levels if lv is None)),
        "mean_s": sum(seconds) / total if total else 0.0,
        "total_s": sum(seconds),
        "mean_fwd": sum(passes) / total if total else 0.0,
    }


def load_run(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Return ``{suite name: results}`` for every results file in a run."""
    suffix = ".results.json"
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(run_dir.glob(f"*{suffix}")):
        runs[path.name[:-len(suffix)]] = json.loads(path.read_text())
    return runs


def format_table(rows: list[tuple[str, str, dict[str, float]]]) -> str:
    """Format ``(suite, run, stats)`` rows as a markdown table."""
    header = ("| suite | run | cases | errors | name | lenient | strict "
              "| mean s | total s | mean fwd |")
    lines = [header, "|" + "---|" * 10]
    for suite, run, stats in rows:
        lines.append(
            f"| {suite} | {run} | {stats['cases']:.0f} "
            f"| {stats['errors']:.0f} | {stats['name_acc']:.0%} "
            f"| {stats['lenient_acc']:.0%} | {stats['strict_acc']:.0%} "
            f"| {stats['mean_s']:.2f} | {stats['total_s']:.1f} "
            f"| {stats['mean_fwd']:.0f} |"
        )
    return "\n".join(lines)


def format_failures(run: str, suite: str,
                    results: list[dict[str, Any]]) -> list[str]:
    """Return text lines describing each case that is not strictly right."""
    lines = []
    for r in results:
        level = failure_level(r["expected"], r["got"])
        if level is None:
            continue
        lines.append(f"[{run} / {suite}] {r['id']}: fails at {level}")
        lines.append(f"  prompt:   {r['prompt']}")
        lines.append(f"  expected: {json.dumps(r['expected'])}")
        lines.append(f"  got:      {json.dumps(r['got'])}")
        if r["error"] is not None:
            lines.append(f"  error:    {r['error']}")
    return lines


def main() -> int:
    """Print the score table (and failures); return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--failures", action="store_true",
                        help="list every case that is not strictly right")
    args = parser.parse_args()

    # rows grouped by suite, so runs of the same suite sit next to each
    # other in the table.
    rows: list[tuple[str, str, dict[str, float]]] = []
    failures: list[str] = []
    for run_dir in args.run_dirs:
        run = run_dir.resolve().name
        suites = load_run(run_dir)
        if not suites:
            print(f"warning: no *.results.json in {run_dir}",
                  file=sys.stderr)
        for suite, results in suites.items():
            rows.append((suite, run, suite_stats(results)))
            failures.extend(format_failures(run, suite, results))
    rows.sort(key=lambda row: row[0])  # stable: keeps the run order
    print(format_table(rows))

    counts: dict[str, set[float]] = {}
    for suite, _, stats in rows:
        counts.setdefault(suite, set()).add(stats["cases"])
    for suite, sizes in counts.items():
        if len(sizes) > 1:
            print(f"\nwarning: runs of {suite} have different case counts "
                  f"{sorted(sizes)} (--limit?); they are not comparable.")
    if args.failures:
        print("\n" + "\n".join(failures) if failures else "\nno failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
