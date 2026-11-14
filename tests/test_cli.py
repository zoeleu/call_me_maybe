"""CLI failures should be readable and must not start unnecessary inference."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src import __main__ as cli
from src.decoder import GenerationError
from src.models import FunctionCallResult


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create small valid inputs and isolate the CLI from the real model."""
    functions = [{
        "name": "hello", "description": "Say hello",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    }]
    config = {
        "system_prompt": "{tools}", "chat_template": "{system}\n{user}",
        "max_value_tokens": 10, "answer_prefix": "", "parameters_prefix": "",
    }
    for name, content in (
        ("functions.json", functions), ("config.json", config),
        ("input.json", [{"prompt": "Say hello to Zoe"}]),
    ):
        (tmp_path / name).write_text(json.dumps(content))
    monkeypatch.setattr(cli.sys, "argv", [
        "src", "--functions_definition", str(tmp_path / "functions.json"),
        "--prompts_config", str(tmp_path / "config.json"),
        "--input", str(tmp_path / "input.json"),
        "--output", str(tmp_path / "output.json"),
    ])
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: False)
    return tmp_path


@pytest.mark.parametrize("filename", [
    "functions.json", "config.json", "input.json",
])
@pytest.mark.parametrize("invalid", [None, "{", "{}", "\xff"])
def test_invalid_file_prevents_model_loading(
    inputs: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], filename: str, invalid: str | None,
) -> None:
    """Missing, malformed, wrong-schema and non-UTF-8 files fail early."""
    path = inputs / filename
    if invalid is None:
        path.unlink()
    else:
        path.write_bytes(invalid.encode("latin1"))
    model = Mock()
    monkeypatch.setattr(cli.llm_sdk, "Small_LLM_Model", model)
    assert cli.main() == 1
    model.assert_not_called()
    assert str(path) in capsys.readouterr().err


@pytest.mark.parametrize("case", ["empty", "duplicate", "type", "template"])
def test_semantic_validation_precedes_model(
    inputs: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    """Valid JSON can still be unusable by the decoder."""
    path = inputs / "functions.json"
    functions = json.loads(path.read_text())
    if case == "empty":
        functions = []
    elif case == "duplicate":
        functions *= 2
    elif case == "type":
        functions[0]["parameters"]["name"]["type"] = "array"
    else:
        config_path = inputs / "config.json"
        config = json.loads(config_path.read_text())
        config["chat_template"] = "{unknown}"
        config_path.write_text(json.dumps(config))
    path.write_text(json.dumps(functions))
    model = Mock()
    monkeypatch.setattr(cli.llm_sdk, "Small_LLM_Model", model)
    assert cli.main() == 1
    model.assert_not_called()


@pytest.mark.parametrize("stage", ["model", "decoder", "generation", "write"])
def test_pipeline_failures(
    inputs: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], stage: str,
) -> None:
    """Expected runtime failures return one and leave existing output alone."""
    model = Mock()
    decoder = Mock()
    decoder.generate_call.return_value = FunctionCallResult(
        prompt="Say hello to Zoe", name="hello", parameters={"name": "Zoe"},
    )
    factory = Mock(return_value=decoder)
    monkeypatch.setattr(cli.llm_sdk, "Small_LLM_Model", model)
    monkeypatch.setattr(cli.src, "ConstrainedDecoder", factory)
    output = inputs / "output.json"
    if stage == "model":
        model.side_effect = OSError("model unavailable")
    elif stage == "decoder":
        factory.side_effect = ValueError("invalid tokenizer")
    elif stage == "generation":
        output.write_text("previous results")
        decoder.generate_call.side_effect = GenerationError("no valid value")
    else:
        output.mkdir()
    assert cli.main() == 1
    assert "error:" in capsys.readouterr().err
    if stage == "generation":
        assert output.read_text() == "previous results"


def test_success(inputs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete batch still writes the expected JSON and returns zero."""
    result = FunctionCallResult(prompt="hi", name="hello", parameters={})
    monkeypatch.setattr(cli.llm_sdk, "Small_LLM_Model", Mock())
    monkeypatch.setattr(
        cli.src, "ConstrainedDecoder",
        Mock(return_value=Mock(generate_call=Mock(return_value=result))),
    )
    assert cli.main() == 0
    assert json.loads((inputs / "output.json").read_text()) == [
        result.model_dump(),
    ]


def test_cleanup_precedes_error(
    inputs: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Terminal cleanup must not erase the final error message."""
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True)
    printer = Mock()
    printer.clear.side_effect = lambda: print("cleared", file=cli.sys.stderr)
    monkeypatch.setattr(cli, "CallPrinter", Mock(return_value=printer))
    monkeypatch.setattr(cli.llm_sdk, "Small_LLM_Model", Mock())
    decoder = Mock()
    decoder.generate_call.side_effect = GenerationError("bad value")
    monkeypatch.setattr(
        cli.src, "ConstrainedDecoder", Mock(return_value=decoder),
    )
    assert cli.main() == 1
    message = capsys.readouterr().err
    assert message.index("cleared") < message.index("error:")
    assert "prompt 1" in message


def test_unexpected_bug_is_not_hidden(
    inputs: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Programming errors should not be mislabeled as expected failures."""
    monkeypatch.setattr(cli.llm_sdk, "Small_LLM_Model", Mock())
    decoder = Mock()
    decoder.generate_call.side_effect = TypeError("programming bug")
    monkeypatch.setattr(
        cli.src, "ConstrainedDecoder", Mock(return_value=decoder),
    )
    with pytest.raises(TypeError, match="programming bug"):
        cli.main()
