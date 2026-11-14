"""Pydantic models for function definitions, prompts and results."""

from typing import Any
from string import Formatter

from pydantic import BaseModel, Field, field_validator

# Parameter types the constrained decoder can generate a value for.
# The decoder reads this too, so the two can never disagree.
SUPPORTED_PARAMETER_TYPES: tuple[str, ...] = (
    "string", "number", "integer", "boolean",
)


class ParameterSpec(BaseModel):
    """Type description of a single function parameter or return value."""

    type: str


class FunctionDefinition(BaseModel):
    """A callable function the LLM may choose from."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec

    @field_validator("parameters")
    @classmethod
    def supported_parameters(
        cls, parameters: dict[str, ParameterSpec],
    ) -> dict[str, ParameterSpec]:
        """Reject parameter types the decoder cannot generate."""
        for name, spec in parameters.items():
            if spec.type not in SUPPORTED_PARAMETER_TYPES:
                raise ValueError(
                    f"{name}: unsupported parameter type {spec.type!r}"
                )
        return parameters


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

    @field_validator("chat_template")
    @classmethod
    def valid_chat_template(cls, template: str) -> str:
        """Require a usable template with system and user placeholders."""
        try:
            fields = {
                field for _, field, _, _ in Formatter().parse(template)
                if field is not None
            }
            if fields != {"system", "user"}:
                raise ValueError("expected {system} and {user} placeholders")
            template.format(system="", user="")
        except (ValueError, KeyError, IndexError, AttributeError) as error:
            raise ValueError(f"invalid chat_template: {error}") from error
        return template


class PipelineInputs(BaseModel):
    """Validated inputs required before model initialization."""

    functions: list[FunctionDefinition] = Field(min_length=1)
    config: PromptConfig
    prompts: list[PromptInput]

    @field_validator("functions")
    @classmethod
    def unique_functions(
        cls, functions: list[FunctionDefinition],
    ) -> list[FunctionDefinition]:
        """Prevent duplicate names from silently overwriting functions."""
        names: set[str] = set()
        for function in functions:
            if function.name in names:
                raise ValueError(f"duplicate function name: {function.name!r}")
            names.add(function.name)
        return functions


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
