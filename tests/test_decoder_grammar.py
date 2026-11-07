"""Check the value state machine of the decoder, without loading the model."""

import pytest

from src.decoder import START, _advance


@pytest.mark.parametrize("text", [" 23,", " -7,", " 0,"])
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


def test_integer_partial() -> None:
    """A sign alone is a valid start that still needs digits."""
    assert _advance("integer", ",", START, " -") == ("sign", 0)


@pytest.mark.parametrize("text", [" 4.5,", " -0.25,", " 23,"])
def test_number_still_accepts_fractions(text: str) -> None:
    """Blocking the period for integers must not affect numbers."""
    assert _advance("number", ",", START, text) == ("end", 1)
