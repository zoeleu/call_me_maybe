"""Entry point: ``python -m src``."""

import argparse
import json
import os
import sys
from pathlib import Path
import src
import llm_sdk
from src import load_prompts, write_results
from src.models import FunctionCallResult

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


class ThoughtPrinter:
    """Stream the model's thinking to stderr in gray, then erase it.

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

    def clear(self) -> None:
        """Go back to where the thinking started and erase to the end."""
        # "\033[nA" moves up n rows ("\033[0A" would still move up one);
        # "\033[J" erases from the cursor to the end of the screen.
        up = f"\033[{self.row}A" if self.row else ""
        print(f"\r{up}\033[J", end="", file=sys.stderr, flush=True)
        self.row = 0
        self.column = 0


def format_result(result: FunctionCallResult) -> str:
    """Format a result as ``name(param=value, ...)``."""
    arguments = ", ".join(
        f"{name}={json.dumps(value)}"
        for name, value in result.parameters.items()
    )
    return f"{result.name}({arguments})"


def main() -> int:
    """Run the pipeline; return a process exit code."""
    args = parse_args(sys.argv[1:])
    model = llm_sdk.Small_LLM_Model()
    functions = src.load_functions(args.functions_definition)
    config = src.load_prompt_config(args.prompts_config)
    printer = ThoughtPrinter() if sys.stderr.isatty() else None
    decoder = src.ConstrainedDecoder(
        model=model,
        functions=functions,
        config=config,
        on_think_token=printer.write if printer else None,
        on_think_done=printer.clear if printer else None,
    )
    input_file = args.input
    output_file = args.output

    input_parsed = [p.prompt for p in load_prompts(Path(input_file))]
    output: list[FunctionCallResult] = []
    try:
        for prompt in input_parsed:
            result = decoder.generate_call(
                prompt=prompt
            )
            output.append(result)
            print(format_result(result), file=sys.stderr, flush=True)
    finally:
        # Also erase the thinking if generation crashes or is interrupted.
        if printer:
            printer.clear()
    write_results(output_file, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
