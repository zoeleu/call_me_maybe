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
    """Prompt texts and limits used by the constrained decoder."""

    system_prompt: str
    chat_template: str
    think: bool = True
    think_prefix: str
    think_end: str
    max_think_tokens: int = Field(gt=0)
    max_value_tokens: int = Field(gt=0)
    force_answer: str
    answer_prefix: str
    parameters_prefix: str
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)
    min_p: float = Field(ge=0, le=1)
    # Prompt-lookup speculation: guessed tokens checked per forward pass
    # (0 disables it), and the longest n-gram looked up to make a guess.
    speculate_tokens: int = Field(default=8, ge=0)
    speculate_ngram: int = Field(default=3, gt=0)


if __name__ == "__main__":
    import json

    with open("config/prompts.schema.json", "w") as f:
        json.dump(PromptConfig.model_json_schema(), f, indent=2)
