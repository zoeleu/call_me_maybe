"""Check the comparison rules of bench/grade.py (no model needed)."""

import pytest

from bench.grade import (
    failure_level,
    lenient_value_equal,
    normalize_text,
    strict_parameters_equal,
    strict_value_equal,
)


@pytest.mark.parametrize("expected, got", [
    (2.0, 2.0), (3, 3), ("hello", "hello"),
])
def test_strict_equal(expected: object, got: object) -> None:
    """Same value and same type is strictly equal."""
    assert strict_value_equal(expected, got)


@pytest.mark.parametrize("expected, got", [
    (2.0, 2),        # the moulinette asserts isinstance(a, float)
    (1, True),       # bool is a subclass of int; must not pass
    ("2019", 2019),
    ("Hello", "hello"),
])
def test_strict_rejects(expected: object, got: object) -> None:
    """A different type or a different spelling is not strictly equal."""
    assert not strict_value_equal(expected, got)


@pytest.mark.parametrize("expected, got", [
    (2.0, 2),
    ("2019", 2019),
    (2019, " 2019 "),
    (0.1 + 0.2, 0.3),                  # float rounding noise
    ("New  York ", "new york"),
    ("Straße", "STRASSE"),             # casefold, not lower
])
def test_lenient_accepts(expected: object, got: object) -> None:
    """Numbers compare as numbers; strings ignore case and spacing."""
    assert lenient_value_equal(expected, got)


@pytest.mark.parametrize("expected, got", [
    (2.0, 2.5),
    (5, "five"),
    ("1.0", "1"),     # two strings are compared as text, not numbers
    (1, True),
    ("cat", "dog"),
])
def test_lenient_rejects(expected: object, got: object) -> None:
    """Different values stay different even in lenient mode."""
    assert not lenient_value_equal(expected, got)


def test_normalize_text() -> None:
    """Whitespace is trimmed and collapsed, case is folded."""
    assert normalize_text("  A\t b\n\nC ") == "a b c"


def test_parameters_need_same_keys() -> None:
    """A missing or extra parameter fails, even with equal values."""
    assert not strict_parameters_equal({"a": 1}, {"a": 1, "b": 2})
    assert not strict_parameters_equal({"a": 1}, {"b": 1})


@pytest.mark.parametrize("got, level", [
    (None, "name"),
    ({"name": "fn_other", "parameters": {"a": 2.0}}, "name"),
    ({"name": "fn_add", "parameters": {"a": 3.0}}, "lenient"),
    ({"name": "fn_add", "parameters": {"a": 2}}, "strict-only"),
    ({"name": "fn_add", "parameters": {"a": 2.0}}, None),
])
def test_failure_level(got: dict[str, object] | None,
                       level: str | None) -> None:
    """Each failure is reported at the first level it fails."""
    expected = {"name": "fn_add", "parameters": {"a": 2.0}}
    assert failure_level(expected, got) == level
