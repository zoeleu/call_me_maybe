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
