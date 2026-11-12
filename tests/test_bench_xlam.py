"""Check the row filter of bench/xlam_select.py on hand-made rows."""

import json
from typing import Any

import pytest

from bench.xlam_select import (
    base_type,
    category_of,
    is_small_inference,
    make_candidate,
    value_in_query,
)


def row(query: str, answers: list[dict[str, Any]],
        tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a dataset row the way xLAM stores it (JSON strings)."""
    return {"id": 1, "query": query, "answers": json.dumps(answers),
            "tools": json.dumps(tools)}


def tool(name: str, **types: str) -> dict[str, Any]:
    """Build an xLAM tool whose parameters have the given types."""
    return {"name": name, "description": f"{name} tool",
            "parameters": {p: {"description": "", "type": t, "default": ""}
                           for p, t in types.items()}}


@pytest.mark.parametrize("declared, mapped", [
    ("str", "string"), ("int, optional", "integer"), ("float", "number"),
    ("str, default='x'", "string"), ("bool", None), ("List[int]", None),
    ("Tuple[float, float]", None),
])
def test_base_type(declared: str, mapped: str | None) -> None:
    """Only str/int/float are kept, optional or not."""
    assert base_type(declared) == mapped


def test_usable_row() -> None:
    """A single valid call is converted; number values become floats."""
    candidate = make_candidate(row(
        "Area of a 3 by 4.5 rectangle",
        [{"name": "area", "arguments": {"w": 3, "h": 4.5}}],
        [tool("area", w="float", h="float"), tool("other", q="str")],
    ))
    assert candidate is not None
    assert candidate.expected == {"name": "area",
                                  "parameters": {"w": 3.0, "h": 4.5}}
    assert isinstance(candidate.expected["parameters"]["w"], float)
    assert candidate.category == "float"
    assert candidate.findable
    assert candidate.n_tools == 2


@pytest.mark.parametrize("answers, tools", [
    # two calls
    ([{"name": "f", "arguments": {"a": "x"}}] * 2, [tool("f", a="str")]),
    # unknown tool
    ([{"name": "g", "arguments": {"a": "x"}}], [tool("f", a="str")]),
    # an optional parameter left out
    ([{"name": "f", "arguments": {"a": "x"}}],
     [tool("f", a="str", b="int, optional")]),
    # a list parameter in ANOTHER tool of the row
    ([{"name": "f", "arguments": {"a": "x"}}],
     [tool("f", a="str"), tool("g", b="List[int]")]),
    # wrong value type (a string for an integer, a bool for an integer)
    ([{"name": "f", "arguments": {"a": "3"}}], [tool("f", a="int")]),
    ([{"name": "f", "arguments": {"a": True}}], [tool("f", a="int")]),
    # empty string
    ([{"name": "f", "arguments": {"a": ""}}], [tool("f", a="str")]),
    # duplicate tool names
    ([{"name": "f", "arguments": {"a": "x"}}],
     [tool("f", a="str"), tool("f", a="str")]),
    # no parameters
    ([{"name": "f", "arguments": {}}], [tool("f")]),
])
def test_unusable_rows(answers: list[dict[str, Any]],
                       tools: list[dict[str, Any]]) -> None:
    """Each broken rule makes the row unusable."""
    assert make_candidate(row("x", answers, tools)) is None


@pytest.mark.parametrize("value, query, found", [
    ("Paris", "weather in paris", True),
    (1000, "a budget of 1,000 dollars", True),
    (5, "about 15 items", False),     # 5 inside 15 does not count
    (2.0, "2 apples", True),
    (0.05, "a 5% rate", False),       # needs an inference
    (1e-05, "a charge of 0.00001 C", True),  # no scientific notation
])
def test_value_in_query(value: Any, query: str, found: bool) -> None:
    """Values are found as text or as whole numbers."""
    assert value_in_query(value, query) is found


@pytest.mark.parametrize("value, small", [
    (0.05, True),                     # numbers: unit or % conversion
    ("2023-04", True),                # "April 2023": digits are there
    ("GBP,JPY", True),
    ("token456", False),              # invented id
    (7654321, False),                 # invented integer id
    (10, True),                       # "ten" -> 10 is fine
    ("https://open.spotify.com/artist/6vWD", False),  # too long
])
def test_is_small_inference(value: Any, small: bool) -> None:
    """Only short values without invented digit runs count."""
    query = "List crimes for April 2023, prices in GBP and JPY, 5% rate"
    assert is_small_inference(value, query) is small


def test_two_missing_values_rejected() -> None:
    """A row needing two inferences is dropped; one is allowed."""
    tools = [tool("f", a="str", b="str")]
    two = [{"name": "f", "arguments": {"a": "x1", "b": "y1"}}]
    one = [{"name": "f", "arguments": {"a": "paris", "b": "y1"}}]
    assert make_candidate(row("weather in Paris", two, tools)) is None
    candidate = make_candidate(row("weather in Paris", one, tools))
    assert candidate is not None and candidate.kind == "inference"


def test_category_of() -> None:
    """3+ parameters wins, then float, then integer, then string."""
    assert category_of(["string", "string", "integer"]) == "many_params"
    assert category_of(["string", "number"]) == "float"
    assert category_of(["integer"]) == "integer"
    assert category_of(["string"]) == "string"
