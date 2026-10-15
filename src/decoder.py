"""Constrained decoding: mask logits so only valid tokens can be chosen."""

from llm_sdk import Small_LLM_Model

from .models import FunctionCallResult, FunctionDefinition


class ConstrainedDecoder:
    """Drives the LLM token by token to emit a schema-valid function call."""

    def __init__(
        self, model: Small_LLM_Model, functions: list[FunctionDefinition]
    ) -> None:
        """Store the model and the functions the output may refer to."""
        self.model = model
        self.functions = functions

    def generate_call(self, prompt: str) -> FunctionCallResult:
        """Choose a function and its arguments for ``prompt`` using the LLM."""
        raise NotImplementedError
