"""Build the subject suites from the moulinette shipped in the repository.

Usage::

    python bench/make_suites.py [--out-dir bench/data] [--force]

Writes ``subject_public.json`` and ``subject_private.json``. Cases keep the
moulinette's order (function order, then test order) because the real
moulinette grades answers by position. We only import the moulinette's data
and its ``extract_function_info``; ``Moulinette.prepare_exercises`` is never
called (it deletes its output directory with ``shutil.rmtree``).
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Running "python bench/make_suites.py" puts bench/ on sys.path, not the
# repository root, so add the root to be able to import the bench package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.suite import REPO_ROOT, Suite, write_json_new  # noqa: E402


def build_suite(visibility: str) -> Suite:
    """Build the suite of the moulinette's ``public`` or ``private`` set."""
    # The moulinette is a package inside moulinette/, not installed.
    sys.path.insert(0, str(REPO_ROOT / "moulinette"))
    from moulinette.extract_functions_infos import extract_function_info
    from moulinette.functions_definition import exercises

    functions: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    # dicts keep insertion order, so this is the moulinette's own order.
    for function, data in exercises.items():
        if data["visibility"] != visibility:
            continue
        functions.append(extract_function_info(function).model_dump())
        for test in data["tests"]:
            cases.append({
                "id": f"{visibility}-{len(cases) + 1:02d}",
                "prompt": test["prompt"],
                "expected": {
                    "name": function.__name__,
                    "parameters": dict(test["fn_args"]),
                },
            })
    return Suite.model_validate({
        "name": f"subject_{visibility}",
        "functions": functions,
        "cases": cases,
    })


def main() -> int:
    """Write both suites; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "bench" / "data")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing suite files")
    args = parser.parse_args()
    for visibility in ("public", "private"):
        suite = build_suite(visibility)
        path = args.out_dir / f"{suite.name}.json"
        try:
            write_json_new(path, suite.model_dump(), force=args.force)
        except FileExistsError:
            print(f"error: {path} exists (use --force to overwrite)",
                  file=sys.stderr)
            return 1
        print(f"wrote {path} ({len(suite.cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
