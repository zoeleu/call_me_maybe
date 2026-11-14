"""Entry point: ``python -m src``."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import ValidationError
import src
import llm_sdk
from src import load_prompts, write_results
from src.models import FunctionCallResult, PipelineInputs, PromptInput
from src.decoder import GenerationError

T = TypeVar("T")


class CLIError(Exception):
    """An expected pipeline failure with a user-facing message."""


def read_input(path: Path, loader: Callable[[Path], T]) -> T:
    """Load a validated input and attach its path to expected failures."""
    try:
        return loader(path)
    except (OSError, UnicodeError, ValidationError) as error:
        raise CLIError(f"invalid input: {path}: {error}") from error


# ANSI escape codes: gray ("bright black") text, then back to normal.
GRAY = "\033[90m"
RESET = "\033[0m"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LLM function calling")
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json"),
    )
    parser.add_argument(
        "--prompts_config",
        type=Path,
        default=Path("config/prompts.json"),
    )
    return parser.parse_args(argv)


class CallPrinter:
    """Draw the call being written on stderr in gray, then erase it.

    To erase, it moves the cursor back up as many rows as it printed, so it
    counts newlines and the rows the terminal wraps long lines into.
    """

    def __init__(self) -> None:
        """Start at the cursor's current position."""
        self.width = os.get_terminal_size(sys.stderr.fileno()).columns
        self.row = 0
        self.column = 0

    def write(self, text: str) -> None:
        """Print ``text`` in gray and track where the cursor ends up."""
        print(f"{GRAY}{text}{RESET}", end="", file=sys.stderr, flush=True)
        for char in text:
            if char == "\n":
                self.row += 1
                self.column = 0
                continue
            if self.column == self.width:
                # The line is full: the terminal wraps before this char.
                self.row += 1
                self.column = 0
            self.column += 1

    def rewrite(self, text: str) -> None:
        """Replace everything printed so far with ``text``."""
        self.clear()
        self.write(text)

    def clear(self) -> None:
        """Go back to where the text started and erase to the end."""
        # "\033[nA" moves up n rows ("\033[0A" would still move up one);
        # "\033[J" erases from the cursor to the end of the screen.
        up = f"\033[{self.row}A" if self.row else ""
        print(f"\r{up}\033[J", end="", file=sys.stderr, flush=True)
        self.row = 0
        self.column = 0


def format_result(result: FunctionCallResult) -> str:
    """Format a result as ``name(param=value, ...)``."""
    arguments = ", ".join(
        f"{name}={json.dumps(value, indent=2)}"
        for name, value in result.parameters.items()
    )
    return f"{result.name}({arguments})"


def load_inputs(args: argparse.Namespace) -> PipelineInputs:
    """Read and validate every input before loading the model."""
    return PipelineInputs(
        functions=read_input(args.functions_definition, src.load_functions),
        config=read_input(args.prompts_config, src.load_prompt_config),
        prompts=read_input(args.input, load_prompts),
    )


def create_decoder(
    inputs: PipelineInputs, printer: CallPrinter | None,
) -> src.ConstrainedDecoder:
    """Initialize the model and decoder with contextual failure reporting."""
    try:
        model = llm_sdk.Small_LLM_Model(inputs.config.model_name)
        return src.ConstrainedDecoder(
            model=model,
            functions=inputs.functions,
            config=inputs.config,
            on_call_update=printer.rewrite if printer else None,
            on_call_done=printer.clear if printer else None,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CLIError(f"model initialization failed: {error}") from error


def generate_results(
    decoder: src.ConstrainedDecoder, prompts: list[PromptInput],
) -> list[FunctionCallResult]:
    """Generate a complete batch, stopping at the first expected failure."""
    output: list[FunctionCallResult] = []
    for index, prompt in enumerate(prompts, start=1):
        try:
            result = decoder.generate_call(prompt=prompt.prompt)
        except (GenerationError, OSError, RuntimeError) as error:
            raise CLIError(
                f"generation failed for prompt {index}: {error}"
            ) from error
        output.append(result)
        print(format_result(result), file=sys.stderr, flush=True)
    return output


def run(args: argparse.Namespace) -> None:
    """Run each pipeline stage; write only a successfully generated batch."""
    inputs = load_inputs(args)
    printer = CallPrinter() if sys.stderr.isatty() else None
    try:
        decoder = create_decoder(inputs, printer)
        output = generate_results(decoder, inputs.prompts)
    finally:
        if printer:
            printer.clear()
    try:
        write_results(args.output, output)
    except (OSError, ValueError) as error:
        raise CLIError(f"cannot write {args.output}: {error}") from error


def main() -> int:
    """Translate expected pipeline failures into a message and exit status."""
    args = parse_args(sys.argv[1:])
    try:
        run(args)
    except (CLIError, ValidationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
