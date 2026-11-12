"""Pydantic models for function definitions, prompts and results."""

from typing import Any

from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    """Type description of a single function parameter or return value."""

    type: str


class FunctionDefinition(BaseModel):
    """A callable function the LLM may choose from."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class PromptInput(BaseModel):
    """A natural-language prompt to translate into a function call."""

    prompt: str


class FunctionCallResult(BaseModel):
    """The function call chosen for a prompt, with typed arguments."""

    prompt: str
    name: str
    parameters: dict[str, Any]


class PromptConfig(BaseModel):
    """Model name, prompt texts and limits used by the constrained decoder.

    ``chat_template`` holds ``{system}`` and ``{user}`` and ends where the
    assistant's answer starts; ``answer_prefix`` is the fixed start of that
    answer, up to the function name.
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    system_prompt: str
    chat_template: str
    max_value_tokens: int = Field(gt=0)
    answer_prefix: str
    parameters_prefix: str


class TokenizerModel(BaseModel):
    """The ``model`` part of a ``tokenizer.json`` file."""

    vocab: dict[str, int]


class TokenizerFile(BaseModel):
    """The part of a Hugging Face ``tokenizer.json`` the decoder reads."""

    model: TokenizerModel


if __name__ == "__main__":
    import json

    with open("config/prompts.schema.json", "w") as f:
        json.dump(PromptConfig.model_json_schema(), f, indent=2)
