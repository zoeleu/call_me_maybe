"""Check the comparison rules of bench/score.py (no model needed)."""

import pytest

from bench.score import (
    failure_level,
    lenient_value_equal,
    normalize_text,
    strict_parameters_equal,
    strict_value_equal,
    suite_stats,
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


def test_suite_stats() -> None:
    """Accuracies count each level; errors and timings are summed."""
    expected = {"name": "f", "parameters": {"x": 1.0}}

    def result(got: dict[str, object] | None, error: str | None = None
               ) -> dict[str, object]:
        """Build one results entry with fixed timing."""
        return {"expected": expected, "got": got, "error": error,
                "seconds": 1.0, "forward_passes": 10}

    stats = suite_stats([
        result({"name": "f", "parameters": {"x": 1.0}}),   # strict
        result({"name": "f", "parameters": {"x": 1}}),     # lenient
        result({"name": "f", "parameters": {"x": 9.0}}),   # name only
        result(None, "RuntimeError: boom"),                # error
    ])
    assert stats["cases"] == 4
    assert stats["errors"] == 1
    assert stats["name_acc"] == 0.75
    assert stats["lenient_acc"] == 0.5
    assert stats["strict_acc"] == 0.25
    assert stats["total_s"] == 4.0
    assert stats["mean_fwd"] == 10


def test_suite_stats_empty() -> None:
    """An empty suite gives zeros instead of dividing by zero."""
    assert suite_stats([])["strict_acc"] == 0.0
