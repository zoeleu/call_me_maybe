"""Check the value state machine of the decoder, without loading the model."""

import pytest

from src.decoder import START, State, _advance


@pytest.mark.parametrize("text", [" 23,", " -7,", " 0,", " -0,"])
def test_integer_accepts(text: str) -> None:
    """Integers, signed or not, are complete once ``end`` is written."""
    assert _advance("integer", ",", START, text) == ("end", 1)


def test_integer_accepts_closing_end() -> None:
    """The last integer closes the call with ``}}``."""
    assert _advance("integer", "}}", START, " 23}}") == ("end", 2)


@pytest.mark.parametrize("text", [" 4.5,", " 4.", " -,", " ,", " +3", ' "3"'])
def test_integer_rejects(text: str) -> None:
    """A period, a lone sign, an empty value or a quote is not an integer."""
    assert _advance("integer", ",", START, text) is None


@pytest.mark.parametrize("text", [" true,", " false,"])
def test_boolean_accepts(text: str) -> None:
    """Both booleans, spelled out, are complete once ``end`` is written."""
    assert _advance("boolean", ",", START, text) == ("end", 1)


def test_boolean_accepts_closing_end() -> None:
    """The last boolean closes the call with ``}}``."""
    assert _advance("boolean", "}}", START, " true}}") == ("end", 2)


@pytest.mark.parametrize(
    "text", [" True,", " 1,", " tru,", " truex,", ' "true"', " true "]
)
def test_boolean_rejects(text: str) -> None:
    """Capitalized, numeric, partial or overlong words are not booleans."""
    assert _advance("boolean", ",", START, text) is None


@pytest.mark.parametrize(
    "text, state",
    [(" t", ("true", 1)), (" tru", ("true", 3)),
     (" f", ("false", 1)), (" fals", ("false", 4))],
)
def test_boolean_partial(text: str, state: State) -> None:
    """A boolean split anywhere still has a valid mid-word state."""
    assert _advance("boolean", ",", START, text) == state


def test_integer_partial() -> None:
    """A sign alone is a valid start that still needs digits."""
    assert _advance("integer", ",", START, " -") == ("sign", 0)


@pytest.mark.parametrize("text", [" 4.5,", " -0.25,", " 23,"])
def test_number_still_accepts_fractions(text: str) -> None:
    """Blocking the period for integers must not affect numbers."""
    assert _advance("number", ",", START, text) == ("end", 1)
