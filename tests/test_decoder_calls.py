"""Check whole calls written by the decoder, with a fake model.

The fake model uses the real Qwen3 tokenizer (a small download) but no
neural network: it always prefers the tokens that spell a target answer.
So these tests show what the decoder lets through and how it parses it,
not what a real model would choose.
"""

import json
from typing import Any, cast

import pytest
from huggingface_hub import hf_hub_download
from llm_sdk import Small_LLM_Model
from transformers import AutoTokenizer

from src.decoder import ConstrainedDecoder
from src.models import FunctionDefinition, ParameterSpec, PromptConfig

MODEL_NAME = "Qwen/Qwen3-0.6B"
CONFIG = PromptConfig(
    model_name=MODEL_NAME,
    system_prompt="Tools:\n{tools}",
    chat_template="{system}\n{user}\n",
    max_value_tokens=16,
    answer_prefix='{"name": "',
    parameters_prefix=', "arguments": {',
)


class FakeModel:
    """Stands in for ``Small_LLM_Model``; wants to write ``self.target``."""

    def __init__(self) -> None:
        """Load the tokenizer and the text of every token."""
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.target = ""
        self._texts: dict[int, str] = {}

    def encode(self, text: str) -> list[list[int]]:
        """Tokenize ``text`` as a batch of one, like the SDK."""
        ids: list[int] = self._tokenizer.encode(
            text, add_special_tokens=False
        )
        return [ids]

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text."""
        return str(self._tokenizer.decode(ids, skip_special_tokens=True))

    def get_path_to_tokenizer_file(self) -> str:
        """Return the path of the tokenizer.json file."""
        return str(hf_hub_download(MODEL_NAME, "tokenizer.json"))

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Score each token by how much of the rest of the target it spells.

        The answer written so far is the text after ``answer_prefix``. A
        token that continues the target scores its length (so the longest
        one wins); any other token scores -1.
        """
        if not self._texts:
            self._texts = {
                token: self.decode([token])
                for token in range(len(self._tokenizer))
            }
        written = self.decode(input_ids).rsplit(CONFIG.answer_prefix, 1)[1]
        assert self.target.startswith(written), (written, self.target)
        rest = self.target[len(written):]
        logits = [-1.0] * len(self._tokenizer)
        for token, text in self._texts.items():
            if text and rest.startswith(text):
                logits[token] = float(len(text))
        return logits


@pytest.fixture(scope="module")
def fake() -> FakeModel:
    """Build the fake model once for every test in this file."""
    return FakeModel()


def _call(
    fake: FakeModel, types: dict[str, str], arguments: str
) -> dict[str, Any]:
    """Decode a call to ``fn`` whose target arguments JSON is given.

    ``types`` maps each parameter name to its type; ``arguments`` is the
    text the fake model wants between the braces of ``"arguments"``.
    """
    function = FunctionDefinition(
        name="fn",
        description="A test function.",
        parameters={
            name: ParameterSpec(type=kind) for name, kind in types.items()
        },
        returns=ParameterSpec(type="string"),
    )
    other = function.model_copy(update={"name": "fn_other"})
    fake.target = 'fn", "arguments": {' + arguments + "}}"
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, fake),
        functions=[other, function],
        config=CONFIG,
    )
    result = decoder.generate_call("a prompt")
    assert result.name == "fn"
    return result.parameters


@pytest.mark.parametrize("text, value", [("4", 4), ("-12", -12), ("0", 0)])
def test_integer_stays_int(fake: FakeModel, text: str, value: int) -> None:
    """An ``integer`` parameter comes out as a Python int."""
    parameters = _call(fake, {"n": "integer"}, f'"n": {text}')
    assert parameters == {"n": value}
    assert type(parameters["n"]) is int


@pytest.mark.parametrize(
    "text, value", [("4", 4.0), ("-0.25", -0.25), ("1234567.89", 1234567.89)]
)
def test_number_becomes_float(
    fake: FakeModel, text: str, value: float
) -> None:
    """A ``number`` parameter is a float even when written without a dot."""
    parameters = _call(fake, {"a": "number"}, f'"a": {text}')
    assert parameters == {"a": value}
    assert type(parameters["a"]) is float


def test_mixed_types_in_any_order(fake: FakeModel) -> None:
    """The model picks the parameter order; every type is parsed right."""
    parameters = _call(
        fake,
        {"principal": "number", "rate": "number", "years": "integer"},
        '"years": 23, "principal": 1000, "rate": 0.0375',
    )
    assert parameters == {"years": 23, "principal": 1000.0, "rate": 0.0375}
    assert type(parameters["years"]) is int


@pytest.mark.parametrize(
    "value",
    ["hello", 'Say "hello" to {name}', "C:\\Users\\john", "a, b}} c", "\\d+"],
)
def test_string_round_trip(fake: FakeModel, value: str) -> None:
    """Quotes, backslashes, commas and braces survive inside a string."""
    parameters = _call(fake, {"s": "string"}, f'"s": {json.dumps(value)}')
    assert parameters == {"s": value}


def test_integer_cannot_take_a_fraction(fake: FakeModel) -> None:
    """A model that wants ``4.5`` for an integer is stopped at the dot."""
    with pytest.raises(AssertionError):
        _call(fake, {"n": "integer"}, '"n": 4.5')
