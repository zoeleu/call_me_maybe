"""Pydantic models for function definitions, prompts and results."""

from typing import Any

from pydantic import BaseModel


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
