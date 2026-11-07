"""Check the SDK's KV-cached logits against the uncached reference.

Loads Qwen3-0.6B once for the module (about 3 GB of RAM, ~20 s on CPU).
"""

import numpy as np
import pytest
from llm_sdk import Small_LLM_Model

# float32 on CPU: cached and uncached runs differ by ~1e-4 at most.
TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def model() -> Small_LLM_Model:
    """Load the model once for every test in this file."""
    return Small_LLM_Model()


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
    base = _ids(model, "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n")
    other = _ids(model, "<|im_start|>user\nReverse 'hello'<|im_end|>\n")
    extra = _ids(model, "<|im_start|>assistant\n<think>")
    sequences = [
        base,                       # cold start
        base + extra[:1],           # one new token
        base + extra,               # several new tokens
        base + extra,               # same ids again
        base[:-3],                  # strict prefix of the cache
        other,                      # next prompt, shared start
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
