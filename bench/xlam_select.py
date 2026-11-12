"""Pick about 40 test cases from the xLAM function-calling dataset.

Usage (HF_HOME must point outside the small home directory)::

    HF_HOME=$G/hf python bench/xlam_select.py [--force]

The dataset ``Salesforce/xlam-function-calling-60k`` is gated: the token is
read from the repository's ``.env`` (``HF_TOKEN=...``) unless HF_TOKEN is
already set. Outputs:

* ``bench/data/xlam.json``: the suite (gitignored: rows of a gated dataset
  must not be committed);
* ``bench/data/xlam_review.md``: a readable list of the chosen cases (also
  printed);
* ``bench/xlam_ids.json``: only the chosen ids and their category (small,
  meant to be committed, to know which rows were used).

A row is usable when (see :func:`make_candidate`):

* it has exactly one call, to a tool listed in the row, and tool names in
  the row are unique;
* every parameter of every tool of the row is ``str``, ``int`` or ``float``
  (a trailing ``, optional`` is ignored), mapped to string/integer/number;
* the call gives a value for every parameter of its tool (our decoder
  always writes all of them), the tool has at least one parameter, and each
  value has the declared type (bools never count as numbers);
* no value is an empty string, and no string value is longer than
  ``MAX_STRING_LENGTH`` characters (the decoder stops a value after
  ``max_value_tokens`` tokens, so a very long value would fail for a reason
  unrelated to choosing the right call);
* the row has at most ``MAX_TOOLS`` tools;
* at most one value is missing from the query, and it passes
  :func:`is_small_inference` (rows whose answer contains values nobody
  could guess from the query, like invented IDs, are dropped).

Limits of the conversion: the project's function format has no place for
parameter descriptions, so they are dropped (only the tool's description is
kept); xLAM gives no return type, so ``returns`` is set to string as a
placeholder (the decoder does not show it to the model). Integer values
given for a ``number`` parameter are stored as floats, because the decoder
always produces floats for ``number``.

Selection is deterministic (sorted ids, fixed seed). Each case gets a
category (``many_params`` for 3+ parameters, else ``float``, ``integer`` or
``string``) and a kind: ``findable`` when every value can be found in the
query, ``inference`` otherwise. ``QUOTAS`` gives how many of each to take;
within a quota, tool counts 1 to 6 are taken in turn. Rows 89, 90 and 91 are
considered first.
"""

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.suite import REPO_ROOT, Suite, write_json_new  # noqa: E402

REPO_ID = "Salesforce/xlam-function-calling-60k"
FILENAME = "xlam_function_calling_60k.json"
TYPE_MAP = {"str": "string", "int": "integer", "float": "number"}
SEED = 42
PRIORITY_IDS = (89, 90, 91)
MAX_TOOLS = 6
MAX_STRING_LENGTH = 80
MAX_SINGLE_TOOL = 10
MAX_PER_FUNCTION = 2
MAX_INFERRED_LENGTH = 20
# category -> (findable cases, inference cases). 30 + 10 = 40 cases,
# so a quarter need a small inference.
QUOTAS = {
    "string": (8, 2),
    "integer": (8, 2),
    "float": (7, 3),
    "many_params": (7, 3),
}


@dataclass
class Candidate:
    """A usable dataset row, converted to the project's formats."""

    id: int
    query: str
    functions: list[dict[str, Any]]
    expected: dict[str, Any]
    category: str
    findable: bool

    @property
    def n_tools(self) -> int:
        """Number of functions offered to the model for this case."""
        return len(self.functions)

    @property
    def kind(self) -> str:
        """``findable`` or ``inference``."""
        return "findable" if self.findable else "inference"


def base_type(declared: str) -> str | None:
    """Map an xLAM type like ``"int, optional"`` to a project type.

    Only the part before the first comma counts. Returns None for any
    other type (lists, bools, dicts...), which makes the row unusable.
    """
    return TYPE_MAP.get(declared.split(",")[0].strip())


def convert_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one xLAM tool to the project's function format, or None."""
    parameters: dict[str, Any] = {}
    for name, spec in tool["parameters"].items():
        json_type = base_type(str(spec.get("type", "")))
        if json_type is None:
            return None
        parameters[name] = {"type": json_type}
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": parameters,
        "returns": {"type": "string"},  # placeholder, see module docstring
    }


def value_has_type(value: Any, json_type: str) -> bool:
    """Return True if ``value`` is a valid ``json_type`` value."""
    if isinstance(value, bool):
        return False
    if json_type == "integer":
        return isinstance(value, int)
    if json_type == "number":
        return isinstance(value, (int, float))
    return isinstance(value, str)


def number_text(value: float) -> str:
    """Write a number the way a query would: 3.0 -> "3", 1e-05 -> "0.00001".

    ``Decimal(repr(x))`` keeps the shortest digits Python prints for x, and
    the "f" format writes them without scientific notation.
    """
    value = abs(value)
    if float(value).is_integer():
        return str(int(value))
    return format(Decimal(repr(float(value))), "f")


def value_in_query(value: Any, query: str) -> bool:
    """Return True if ``value`` can be read directly from ``query``.

    Strings: found anywhere, ignoring case. Numbers: their digits appear
    as a whole number (not inside a longer one); thousands commas in the
    query are ignored, so 1000 is found in "1,000".
    """
    if isinstance(value, str):
        return value.casefold() in query.casefold()
    text = re.escape(number_text(value))
    return re.search(rf"(?<![\d.]){text}(?![\d])",
                     query.replace(",", "")) is not None


def is_small_inference(value: Any, query: str) -> bool:
    """Return True if a value missing from the query looks inferable.

    Floats always qualify (unit conversions like 5 micro -> 5e-06). An
    integer or a string must not contain a run of 3+ digits that is absent
    from the query: such values are usually IDs the dataset invented
    ("token456", 7654321), which no model could guess. A string must also
    be short (at most ``MAX_INFERRED_LENGTH`` characters, so no URLs).
    """
    if isinstance(value, float):
        return True
    text = str(value)
    if len(text) > MAX_INFERRED_LENGTH:
        return False
    return all(digits in query for digits in re.findall(r"\d{3,}", text))


def category_of(types: list[str]) -> str:
    """Return the category of a call from its parameter types."""
    if len(types) >= 3:
        return "many_params"
    if "number" in types:
        return "float"
    if "integer" in types:
        return "integer"
    return "string"


def make_candidate(row: dict[str, Any]) -> Candidate | None:
    """Return the row as a Candidate, or None if it is not usable."""
    try:
        answers = json.loads(row["answers"])
        tools = json.loads(row["tools"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    if len(answers) != 1 or not 1 <= len(tools) <= MAX_TOOLS:
        return None
    names = [tool.get("name") for tool in tools]
    if len(set(names)) != len(names):
        return None
    functions = [convert_tool(tool) for tool in tools]
    if any(function is None for function in functions):
        return None
    usable = [f for f in functions if f is not None]  # for mypy
    call = answers[0]
    by_name = {function["name"]: function for function in usable}
    if call.get("name") not in by_name:
        return None
    declared = by_name[call["name"]]["parameters"]
    arguments = call.get("arguments")
    if not declared or not isinstance(arguments, dict):
        return None
    if set(arguments) != set(declared):
        return None
    parameters: dict[str, Any] = {}
    for name, spec in declared.items():
        value = arguments[name]
        if not value_has_type(value, spec["type"]):
            return None
        if isinstance(value, str) and not 0 < len(value) <= \
                MAX_STRING_LENGTH:
            return None
        if spec["type"] == "number":
            value = float(value)  # the decoder writes floats for numbers
        parameters[name] = value
    query = row["query"]
    missing = [v for v in parameters.values()
               if not value_in_query(v, query)]
    # Keep rows where at most one value needs a (small) inference.
    if len(missing) > 1 or (missing and
                            not is_small_inference(missing[0], query)):
        return None
    return Candidate(
        id=int(row["id"]),
        query=query,
        functions=usable,
        expected={"name": call["name"], "parameters": parameters},
        category=category_of([spec["type"] for spec in declared.values()]),
        findable=not missing,
    )


def select(candidates: list[Candidate]) -> list[Candidate]:
    """Choose the cases: priority rows first, then fill ``QUOTAS``.

    Within each (category, kind) quota, tool counts 1, 2, ..., MAX_TOOLS
    are taken in turn so the number of tools is spread; at most
    ``MAX_SINGLE_TOOL`` single-tool cases are taken overall, and at most
    ``MAX_PER_FUNCTION`` cases expect the same function name (the dataset
    repeats some functions, like triangle_area, many times).
    """
    pool = sorted(candidates, key=lambda c: c.id)
    random.Random(SEED).shuffle(pool)
    chosen: list[Candidate] = [c for c in sorted(candidates,
                                                 key=lambda c: c.id)
                               if c.id in PRIORITY_IDS]
    for category, (n_findable, n_inference) in QUOTAS.items():
        for findable, wanted in ((True, n_findable), (False, n_inference)):
            have = sum(1 for c in chosen
                       if c.category == category and c.findable == findable)
            queues = {
                n: [c for c in pool if c.category == category
                    and c.findable == findable and c.n_tools == n
                    and c not in chosen]
                for n in range(1, MAX_TOOLS + 1)
            }
            while have < wanted and any(queues.values()):
                for n, queue in queues.items():
                    if have >= wanted:
                        break
                    singles = sum(1 for c in chosen if c.n_tools == 1)
                    if n == 1 and singles >= MAX_SINGLE_TOOL:
                        queue.clear()  # no more single-tool cases
                    # Skip functions already used MAX_PER_FUNCTION times.
                    while queue and sum(
                            1 for c in chosen
                            if c.expected["name"] == queue[0].expected["name"]
                    ) >= MAX_PER_FUNCTION:
                        queue.pop(0)
                    if queue:
                        chosen.append(queue.pop(0))
                        have += 1
    return chosen


def review_text(chosen: list[Candidate], n_usable: int) -> str:
    """Return the markdown review list of the chosen cases."""
    lines = [f"# xLAM selection ({len(chosen)} cases from {n_usable} "
             "usable rows)", ""]
    for c in chosen:
        lines.append(f"## {c.id}: {c.category}, {c.kind}, "
                     f"{c.n_tools} tool(s)")
        lines.append(f"- query: {c.query}")
        lines.append(f"- expected: `{json.dumps(c.expected)}`")
        lines.append("")
    return "\n".join(lines)


def load_token() -> None:
    """Put HF_TOKEN from the repository's .env into the environment.

    The value is never printed. Nothing happens if HF_TOKEN is already set
    or the file has no such line.
    """
    env_file = REPO_ROOT / ".env"
    if "HF_TOKEN" in os.environ or not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "HF_TOKEN":
            os.environ["HF_TOKEN"] = value.strip().strip("'\"")


def main() -> int:
    """Download, filter, select and write; return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing output files")
    args = parser.parse_args()
    if "HF_HOME" not in os.environ:
        print("error: set HF_HOME (the download is large; keep it out of "
              "the home directory)", file=sys.stderr)
        return 1
    data_dir = REPO_ROOT / "bench" / "data"
    outputs = [data_dir / "xlam.json", REPO_ROOT / "bench" / "xlam_ids.json",
               data_dir / "xlam_review.md"]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.force:
        print(f"error: {', '.join(existing)} exist (use --force)",
              file=sys.stderr)
        return 1
    load_token()
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset")
    rows = json.loads(Path(path).read_text())
    candidates = [c for c in map(make_candidate, rows) if c is not None]
    del rows  # about 60k rows: free them before going on
    chosen = select(candidates)

    suite = Suite.model_validate({
        "name": "xlam",
        "cases": [{"id": f"xlam-{c.id}", "prompt": c.query,
                   "functions": c.functions, "expected": c.expected}
                  for c in chosen],
    })
    ids = [{"id": c.id, "category": c.category, "kind": c.kind}
           for c in chosen]
    review = review_text(chosen, len(candidates))
    # Checked above; "x" mode still refuses if a file appeared since.
    write_json_new(outputs[0], suite.model_dump(), args.force)
    write_json_new(outputs[1], ids, args.force)
    with open(outputs[2], "w" if args.force else "x",
              encoding="utf-8") as file:
        file.write(review)

    print(review)
    print(f"usable rows: {len(candidates)}")
    for category in QUOTAS:
        for kind in ("findable", "inference"):
            available = sum(1 for c in candidates
                            if c.category == category and c.kind == kind)
            taken = sum(1 for c in chosen
                        if c.category == category and c.kind == kind)
            print(f"  {category:12} {kind:10} available {available:6} "
                  f"chosen {taken}")
    tools = sorted({c.n_tools for c in chosen})
    print("  tools per case: " + ", ".join(
        f"{n}: {sum(1 for c in chosen if c.n_tools == n)}" for n in tools))
    return 0


if __name__ == "__main__":
    sys.exit(main())
