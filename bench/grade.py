"""Validate harness outputs against the expected calls in ``bench/cases/``.

Usage::

    python bench/grade.py [RESULTS_DIR] [--failures]

``RESULTS_DIR`` (default ``data/output/xlam``) holds one ``<case>.json``
per case directory in ``bench/cases/``, as written by running the program
on that case (see the Makefile's ``bench`` target)::

    python -m src \\
        --functions_definition bench/cases/xlam-89/functions_definition.json \\
        --input bench/cases/xlam-89/function_calling_tests.json \\
        --output data/output/xlam/xlam-89.json

Each output file must hold exactly one call (each case has one prompt).
Three accuracies are printed:

* name: the function name is right;
* lenient: the parameters "mean the same" (numbers as numbers, strings up
  to spacing and case);
* strict: name right and parameters exactly equal, including the Python
  type (``2 != 2.0``), because the moulinette checks types with
  ``isinstance``.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

CASES_DIR = Path(__file__).resolve().parent / "cases"

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
    ``got`` is None when the run produced no output. Otherwise returns
    the first level that fails: "name" (wrong name or no answer),
    "lenient" (right name, wrong values) or "strict-only" (right values,
    wrong types or formatting).
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


# --- reading cases and outputs ---------------------------------------------


def load_cases() -> list[tuple[str, dict[str, Any]]]:
    """Return ``(case, expected call)`` for every case directory."""
    cases: list[tuple[str, dict[str, Any]]] = []
    if not CASES_DIR.is_dir():
        return cases
    for path in sorted(CASES_DIR.iterdir()):
        expected_path = path / "expected.json"
        if path.is_dir() and expected_path.exists():
            expected: dict[str, Any] = json.loads(expected_path.read_text())
            cases.append((path.name, expected))
    return cases


def load_result(results_dir: Path, case: str) -> dict[str, Any] | None:
    """Return the call the harness wrote for ``case``, or None.

    None also means "the file is missing, empty or holds more than one
    call": the case then fails at the ``name`` level.
    """
    path = results_dir / f"{case}.json"
    try:
        entries = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entries, list) or len(entries) != 1:
        return None
    result: dict[str, Any] = entries[0]
    return result


def format_failure(case: str, expected: dict[str, Any],
                   got: dict[str, Any] | None) -> list[str]:
    """Return the lines describing one failing case."""
    lines = [f"{case}: expected {json.dumps(expected)}"]
    if got is None:
        lines.append(f"{case}: got no result (missing or invalid output "
                     "file)")
    else:
        lines.append(f"{case}: got      {json.dumps(got)}")
    return lines


def main() -> int:
    """Print the per-case verdicts and totals; return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", nargs="?", type=Path,
                        default=Path("data/output/xlam"))
    parser.add_argument("--failures", action="store_true",
                        help="show expected and got for every failure")
    args = parser.parse_args()

    cases = load_cases()
    if not cases:
        print(f"error: no case directories with expected.json in "
              f"{CASES_DIR}", file=sys.stderr)
        return 1
    counts = {"name": 0, "lenient": 0, "strict-only": 0}
    failures: list[str] = []
    for case, expected in cases:
        got = load_result(args.results_dir, case)
        level = failure_level(expected, got)
        if level is None:
            print(f"{case}: ok")
        else:
            if level in counts:
                counts[level] += 1
            failures.extend(format_failure(case, expected, got))
            print(f"{case}: fails at {level}")
    total = len(cases)
    print(f"\n{total} cases: name {total - counts['name']}/{total}, "
          f"lenient {total - counts['name'] - counts['lenient']}/{total}, "
          f"strict {total - sum(counts.values())}/{total}")
    if args.failures:
        print("\n" + "\n".join(failures) if failures else "\nno failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
