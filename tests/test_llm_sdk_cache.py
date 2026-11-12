"""Check the SDK's KV-cached logits against the uncached reference.

Loads each supported model in turn (up to about 4 GB of RAM). A model
that cannot be downloaded (Gemma is gated and needs ``HF_TOKEN``) is skipped.
"""

import numpy as np
import pytest
from llm_sdk import Small_LLM_Model

# float32 on CPU: cached and uncached runs differ by ~1e-4 at most.
TOLERANCE = 1e-3


MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B", "google/gemma-3-270m-it"]


@pytest.fixture(scope="module", params=MODELS)
def model(request: pytest.FixtureRequest) -> Small_LLM_Model:
    """Load each model once for every test in this file."""
    try:
        return Small_LLM_Model(request.param)
    except OSError as error:
        pytest.skip(f"{request.param} is not available: {error}")


def _ids(model: Small_LLM_Model, text: str) -> list[int]:
    """Tokenize ``text`` into a flat list of ids."""
    return [int(i) for i in model.encode(text)[0]]


def _assert_close(got: list[float], expected: list[float]) -> None:
    """Fail unless two logit rows match within TOLERANCE."""
    got_array, expected_array = np.array(got), np.array(expected)
    assert np.abs(got_array - expected_array).max() < TOLERANCE
    assert got_array.argmax() == expected_array.argmax()


def test_cached_matches_uncached(model: Small_LLM_Model) -> None:
    """Every way the ids can change between calls gives correct logits."""
    # Long enough to go past Gemma's sliding window (512 tokens).
    system = "You call tools. " * 150
    base = _ids(model, system + "user\nWhat is 2 + 3?\n")
    other = _ids(model, system + "user\nReverse 'hello'\n")
    third = _ids(model, system + "user\nGreet shrek\n")
    extra = _ids(model, "assistant\n{\"name\": \"")
    sequences = [
        base,                       # cold start
        base + extra[:1],           # one new token
        base + extra,               # several new tokens
        base + extra,               # same ids again
        base[:-3],                  # strict prefix of the cache
        other,                      # next prompt, shared start
        other + extra,              # and its answer
        third,                      # a prompt that fits the saved copy
        third + extra[:2],          # appended after a restore
        extra[:1] + other[1:],      # first token differs
        extra[:1],                  # a single token
    ]
    for ids in sequences:
        _assert_close(
            model.get_logits_cached(ids),
            model.get_logits_from_input_ids(ids),
        )


def test_empty_ids_raise(model: Small_LLM_Model) -> None:
    """Empty input ids raise ValueError."""
    with pytest.raises(ValueError):
        model.get_logits_cached([])
