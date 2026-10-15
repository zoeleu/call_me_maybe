"""Loading of input JSON files and writing of the output file."""

from pathlib import Path

from .models import FunctionCallResult, FunctionDefinition, PromptInput


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    raise NotImplementedError


def load_prompts(path: Path) -> list[PromptInput]:
    """Load and validate test prompts from a JSON file."""
    raise NotImplementedError


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write the function call results to a JSON file."""
    raise NotImplementedError
