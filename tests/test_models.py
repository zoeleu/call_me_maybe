"""Tests for the pydantic models."""

import pytest
from pydantic import ValidationError

from src.models import FunctionDefinition


def test_function_definition_valid() -> None:
    """A well-formed definition is accepted."""
    fn = FunctionDefinition.model_validate(
        {
            "name": "fn_greet",
            "description": "Greet someone.",
            "parameters": {"name": {"type": "string"}},
            "returns": {"type": "string"},
        }
    )
    assert fn.name == "fn_greet"


def test_function_definition_missing_field() -> None:
    """A definition without a name is rejected."""
    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate({"description": "x"})
