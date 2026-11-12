"""Loading of input JSON files and writing of the output file."""

from pathlib import Path

from .models import (
    FunctionCallResult,
    FunctionDefinition,
    PromptConfig,
    PromptInput,
    TokenizerFile,
)
from pydantic import TypeAdapter
from typing import List, TypeVar, Callable


T = TypeVar('T')


def __generate_read_helper(adapter: TypeAdapter[T]) -> Callable[[Path], T]:
    """Generates a function that reads TypeAdapter[T] as a json from path."""
    def read_type_to_json(path: Path) -> T:
        contents = path.read_text()
        return adapter.validate_json(contents)
    return read_type_to_json


def __generate_write_helper(
        adapter: TypeAdapter[T]) -> Callable[[Path, T], None]:
    """Generates a function that writes TypeAdapter[T] as a json to path."""
    def write_type_to_json(path: Path, data: T) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(adapter.dump_json(data))
    return write_type_to_json


load_functions = __generate_read_helper(TypeAdapter(List[FunctionDefinition]))


load_prompts = __generate_read_helper(TypeAdapter(List[PromptInput]))


write_results = __generate_write_helper(TypeAdapter(List[FunctionCallResult]))


__load_tokenizer_file = __generate_read_helper(TypeAdapter(TokenizerFile))


def load_tokenizer_vocab(path: Path) -> dict[str, int]:
    """Return the token-to-id table stored in a ``tokenizer.json`` file."""
    return __load_tokenizer_file(path).model.vocab


load_prompt_config = __generate_read_helper(TypeAdapter(PromptConfig))
