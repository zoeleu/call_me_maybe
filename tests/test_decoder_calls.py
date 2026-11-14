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

from src.decoder import (
    ConstrainedDecoder,
    GenerationError,
    State,
    _check_schema,
    _parse_value,
)
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


@pytest.mark.parametrize("text, value", [("true", True), ("false", False)])
def test_boolean_becomes_bool(
    fake: FakeModel, text: str, value: bool,
) -> None:
    """A ``boolean`` parameter comes out as a Python bool."""
    parameters = _call(fake, {"flag": "boolean"}, f'"flag": {text}')
    assert parameters == {"flag": value}
    assert type(parameters["flag"]) is bool


def test_mixed_types_in_any_order(fake: FakeModel) -> None:
    """The model picks the parameter order; every type is parsed right."""
    parameters = _call(
        fake,
        {"principal": "number", "rate": "number",
         "years": "integer", "verified": "boolean"},
        '"years": 23, "verified": true, "principal": 1000, "rate": 0.0375',
    )
    assert parameters == {
        "years": 23, "verified": True, "principal": 1000.0, "rate": 0.0375,
    }
    assert type(parameters["years"]) is int
    assert type(parameters["verified"]) is bool


@pytest.mark.parametrize(
    "value",
    ["hello", 'Say "hello" to {name}', "C:\\Users\\john", "a, b}} c", "\\d+"],
)
def test_string_round_trip(fake: FakeModel, value: str) -> None:
    """Quotes, backslashes, commas and braces survive inside a string."""
    parameters = _call(fake, {"s": "string"}, f'"s": {json.dumps(value)}')
    assert parameters == {"s": value}


def test_boolean_spans_tokens(fake: FakeModel) -> None:
    """``true`` and ``false`` survive being split across tokens.

    After every prefix of either word, at least one token must keep the
    value going, and the finished word must be closeable in one token.
    """
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, fake),
        functions=[FunctionDefinition(
            name="fn", description="A test function.",
            parameters={"flag": ParameterSpec(type="boolean")},
            returns=ParameterSpec(type="string"),
        )],
        config=CONFIG,
    )
    for word in ("true", "false"):
        for matched in range(len(word) + 1):
            state: State = ("space", 0) if not matched else (word, matched)
            table = decoder._value_table("boolean", ",", state)
            assert table, f"no token continues {word[:matched]!r}"
            if matched == len(word):
                assert ("end", 1) in table.values(), (
                    f"{word!r} cannot be closed after a token split"
                )


def test_integer_cannot_take_a_fraction(fake: FakeModel) -> None:
    """A model that wants ``4.5`` for an integer is stopped at the dot."""
    with pytest.raises(AssertionError):
        _call(fake, {"n": "integer"}, '"n": 4.5')


def test_number_overflow_is_rejected_at_parse() -> None:
    """A decimal too big for float64 raises, not a silent ``inf``.

    ``float()`` on a string never raises on overflow: it returns
    infinity, which pydantic would serialize as ``null``.
    """
    with pytest.raises(GenerationError):
        _parse_value("9" * 400, "number")


def test_number_overflow_boundary() -> None:
    """308 nines is the largest all-nines number a float64 can hold."""
    assert _parse_value("9" * 308, "number") == float("9" * 308)
    with pytest.raises(GenerationError):
        _parse_value("9" * 309, "number")


@pytest.mark.parametrize("value_type", ["integer", "number"])
def test_schema_keeps_bools_out_of_numbers(value_type: str) -> None:
    """``bool`` is a subclass of ``int``; it must not pass as a number."""
    with pytest.raises(GenerationError):
        _check_schema("a", True, value_type)


def test_number_overflow_fails_the_call(fake: FakeModel) -> None:
    """A value limit big enough to overflow float64 fails the whole call.

    Reproduction of the reported bug: with ``max_value_tokens=400`` the
    model may write hundreds of digits; parsing them gave ``inf`` and the
    result JSON had ``null`` instead of a number.
    """
    config = CONFIG.model_copy(update={"max_value_tokens": 420})
    function = FunctionDefinition(
        name="fn",
        description="A test function.",
        parameters={"a": ParameterSpec(type="number")},
        returns=ParameterSpec(type="string"),
    )
    fake.target = 'fn", "arguments": {"a": ' + "9" * 420 + "}}"
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, fake),
        functions=[function],
        config=config,
    )
    with pytest.raises(GenerationError):
        decoder.generate_call("a prompt")


@pytest.mark.parametrize(
    "types, arguments",
    [
        ({"s": "string"}, '"s": ""'),
        ({"n": "integer", "s": "string"}, '"n": 5, "s": ""'),
    ],
)
def test_empty_string_round_trip(
    fake: FakeModel, types: dict[str, str], arguments: str,
) -> None:
    """An empty string is a complete value, whether it ends the call.

    As the last value it must be closed by the ``}}`` token, as a
    middle value by the ``,`` token; both have no characters to strip.
    """
    parameters = _call(fake, types, arguments)
    assert parameters["s"] == ""


def test_huge_integer_is_truncated_but_valid(fake: FakeModel) -> None:
    """An integer longer than the token budget is cut, not broken.

    ``max_value_tokens`` counts value tokens: with 16, this vocab
    spends one on the leading space (there is no " 9" token), 15 on
    digits and one on the forced closing ``}}``. The value stays a
    valid int; a huge ``number`` instead fails past float64 (see the
    overflow tests above).
    """
    config = CONFIG.model_copy(update={"max_value_tokens": 16})
    function = FunctionDefinition(
        name="fn",
        description="A test function.",
        parameters={"n": ParameterSpec(type="integer")},
        returns=ParameterSpec(type="string"),
    )
    fake.target = 'fn", "arguments": {"n": ' + "9" * 400 + "}}"
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, fake),
        functions=[function],
        config=config,
    )
    result = decoder.generate_call("a prompt")
    value = result.parameters["n"]
    assert type(value) is int
    assert str(value) == "9" * 15


def test_many_parameters(fake: FakeModel) -> None:
    """Eight parameters of every type, one string holding , and }}.

    Exercises the ``end`` switching between ``,`` and ``}}`` several
    times, and the space before every value but the first.
    """
    parameters = _call(
        fake,
        {f"p{i}": kind for i, kind in enumerate(
            ["integer", "number", "string", "boolean",
             "integer", "string", "number", "boolean"])},
        ", ".join(f'"p{i}": {value}' for i, value in enumerate(
            ["-3", "2.5", '"x"', "true", "0", '"a, b}} c"', "10", "false"])),
    )
    assert parameters == {
        "p0": -3, "p1": 2.5, "p2": "x", "p3": True,
        "p4": 0, "p5": "a, b}} c", "p6": 10.0, "p7": False,
    }
    assert type(parameters["p0"]) is int
    assert type(parameters["p6"]) is float


def test_zero_parameters(fake: FakeModel) -> None:
    """A function without parameters returns an empty argument dict.

    No value is generated, so the closing ``}}`` of the shown JSON is
    never forced; the decoder builds its result by itself, and the
    tokens it generated are display-only.
    """
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, fake),
        functions=[FunctionDefinition(
            name="fn", description="A test function.",
            parameters={}, returns=ParameterSpec(type="string"),
        )],
        config=CONFIG,
    )
    fake.target = 'fn", "arguments": {}}}'
    result = decoder.generate_call("a prompt")
    assert result.name == "fn"
    assert result.parameters == {}


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        'Say "hello" to {name}',
        "{system} and {user}",
        'back\\slash and "quote"',
        "first\nsecond",
    ],
)
def test_prompt_with_special_characters(
    fake: FakeModel, prompt: str,
) -> None:
    """Odd prompts decode normally and stay verbatim in the chat prompt.

    The prompt is a format *value*, never a template: a placeholder
    typed by the user, such as ``{system}``, must not be substituted.
    """
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, fake),
        functions=[FunctionDefinition(
            name="fn", description="A test function.",
            parameters={"s": ParameterSpec(type="string")},
            returns=ParameterSpec(type="string"),
        )],
        config=CONFIG,
    )
    fake.target = 'fn", "arguments": {"s": "hi"}}'
    result = decoder.generate_call(prompt)
    assert result.prompt == prompt
    assert result.parameters == {"s": "hi"}
    assert prompt in decoder._build_prompt(prompt)


@pytest.mark.parametrize(
    "name, arguments, expected",
    [
        ("fn_int", '"n": 7', {"n": 7}),
        ("fn_str", '"n": "hi"', {"n": "hi"}),
    ],
)
def test_ambiguous_choice_keeps_the_chosen_schema(
    fake: FakeModel, name: str, arguments: str, expected: dict[str, Any],
) -> None:
    """Two functions fit the same prompt; the constraints follow the pick.

    They even share the parameter name with different declared types:
    whichever function the model prefers, its own schema is enforced
    and the other's is ignored.
    """
    functions = [
        FunctionDefinition(
            name="fn_int", description="A test function.",
            parameters={"n": ParameterSpec(type="integer")},
            returns=ParameterSpec(type="string"),
        ),
        FunctionDefinition(
            name="fn_str", description="A test function.",
            parameters={"n": ParameterSpec(type="string")},
            returns=ParameterSpec(type="string"),
        ),
    ]
    fake.target = f'{name}", "arguments": {{{arguments}}}}}'
    decoder = ConstrainedDecoder(
        model=cast(Small_LLM_Model, fake), functions=functions,
        config=CONFIG,
    )
    result = decoder.generate_call("a prompt")
    assert result.name == name
    assert result.parameters == expected
    assert type(result.parameters["n"]) is type(expected["n"])
