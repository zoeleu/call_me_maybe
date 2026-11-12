"""The suite file format shared by every benchmark script.

A suite is one JSON file::

    {"name": "subject_public",
     "functions": [...],            # optional, shared by all cases
     "cases": [{"id": "...", "prompt": "...",
                "functions": [...], # optional, replaces the shared list
                "expected": {"name": "...", "parameters": {...}}}]}

Function definitions are kept as plain dicts here, in the project's format
(``name``, ``description``, ``parameters``, ``returns``). ``bench/run.py``
validates them with the project's own model, so this file does not depend on
``src`` (which may change, or come from another checkout).
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Path of the repository root (the parent of this bench/ directory).
REPO_ROOT = Path(__file__).resolve().parent.parent


class ExpectedCall(BaseModel):
    """The function call a case should produce."""

    name: str
    parameters: dict[str, Any]


class Case(BaseModel):
    """One prompt to run, with the call we expect for it."""

    id: str
    prompt: str
    functions: list[dict[str, Any]] | None = None
    expected: ExpectedCall


class Suite(BaseModel):
    """A named list of cases, with an optional shared function list."""

    name: str
    functions: list[dict[str, Any]] | None = None
    cases: list[Case]

    def functions_for(self, case: Case) -> list[dict[str, Any]]:
        """Return the function list of ``case``, or the shared one.

        Raises ValueError when neither the case nor the suite has one.
        """
        if case.functions is not None:
            return case.functions
        if self.functions is not None:
            return self.functions
        raise ValueError(f"case {case.id!r} has no function list")


def load_suite(path: Path) -> Suite:
    """Read and validate a suite file."""
    return Suite.model_validate_json(path.read_text())


def write_json_new(path: Path, data: Any, force: bool = False) -> None:
    """Write ``data`` as indented JSON to ``path``.

    Unless ``force`` is True, the file must not exist yet: mode "x" makes
    ``open`` raise FileExistsError instead of overwriting it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "x"
    with open(path, mode, encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
