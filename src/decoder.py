"""Constrained decoding: mask logits so only valid tokens can be chosen."""

import json
from pathlib import Path
from typing import Any, Callable, Literal

from llm_sdk import Small_LLM_Model

from .io_utils import load_tokenizer_vocab
from .models import FunctionCallResult, FunctionDefinition, PromptConfig

DIGITS = "0123456789"
JSON_ESCAPABLE = '"\\/bfnrt'

# Where we are in `` <value><end>``: a phase name, and how many characters
# of ``end`` were matched (only used by the "end" phase).
Phase = Literal[
    "start", "space", "string", "escape",
    "sign", "int", "period", "fraction", "end",
]
State = tuple[Phase, int]
START: State = ("start", 0)


def _string_step(phase: Phase, char: str) -> State | None:
    """Advance a JSON string value by one character."""
    if phase in ("start", "space"):
        return ("string", 0) if char == '"' else None
    if phase == "escape":
        return ("string", 0) if char in JSON_ESCAPABLE else None
    if char == "\\":
        return ("escape", 0)
    if char == '"':
        return ("end", 0)
    if ord(char) < 0x20:
        return None
    return ("string", 0)


def _number_step(phase: Phase, char: str, end: str) -> State | None:
    """Advance a number value (``-12``, ``3.5``) by one character."""
    if char in DIGITS:
        if phase in ("start", "space", "sign", "int"):
            return ("int", 0)
        if phase in ("period", "fraction"):
            return ("fraction", 0)
    if char == "-" and phase in ("start", "space"):
        return ("sign", 0)
    if char == "." and phase == "int":
        return ("period", 0)
    if char == end[0] and phase in ("int", "fraction"):
        return ("end", 1)
    return None


def _value_step(
    value_type: str, end: str, state: State, char: str
) -> State | None:
    """Advance `` <value><end>`` by one character.

    Return the new state, or None if ``char`` cannot come next. One space
    is allowed before the value; ``\\uXXXX`` escapes are not supported.
    """
    phase, matched = state
    if phase == "end":
        if matched < len(end) and char == end[matched]:
            return ("end", matched + 1)
        return None
    if phase == "start":
        if char == " ":
            return ("space", 0)
        return None
    if value_type == "string":
        return _string_step(phase, char)
    # An integer is a number that never reaches its fraction part.
    if value_type == "integer" and char == ".":
        return None
    return _number_step(phase, char, end)


def _advance(
    value_type: str, end: str, state: State, text: str
) -> State | None:
    """Feed every character of ``text`` to ``_value_step``."""
    for char in text:
        next_state = _value_step(value_type, end, state, char)
        if next_state is None:
            return None
        state = next_state
    return state


class ConstrainedDecoder:
    """Drives the LLM token by token to emit a schema-valid function call."""

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
        config: PromptConfig,
        on_call_update: Callable[[str], None] | None = None,
        on_call_done: Callable[[], None] | None = None,
    ) -> None:
        """Store the model, the functions and the prompt configuration.

        ``on_call_update``, if given, receives the whole call written so
        far, as ``name(a=1, b``, after each token; ``on_call_done`` is
        called once the call is complete.
        """
        if not functions:
            raise ValueError("at least one function definition is required")
        self.model = model
        self.functions = {function.name: function for function in functions}
        self.config = config
        self.on_call_update = on_call_update
        self.on_call_done = on_call_done
        vocab = load_tokenizer_vocab(Path(model.get_path_to_tokenizer_file()))
        # The vocabulary's keys are not the real text ("Ġ" or "▁" stand for
        # a space), so ask the model for the text of each token instead.
        # Empty tokens (special tokens) are dropped: they would let
        # generation loop without making progress.
        self._token_text: dict[int, str] = {}
        for token in vocab.values():
            text = model.decode([token])
            if text:
                self._token_text[token] = text
        # Cache of _value_table results, filled on first use.
        self._tables: dict[tuple[str, str, State], dict[int, State]] = {}

    def generate_call(self, prompt: str) -> FunctionCallResult:
        """Choose a function and its arguments for ``prompt`` using the LLM."""
        ids = self._encode(
            self._build_prompt(prompt) + self.config.answer_prefix
        )
        function = self.functions[self._choose(
            ids, list(self.functions), show=self._show_call
        )]
        self._show_call(function.name, [])
        ids.extend(self._encode(self.config.parameters_prefix))
        parameters: dict[str, Any] = {}
        # Arguments already written, as "a=1.0", for _show_call.
        shown: list[str] = []
        remaining = list(function.parameters)
        while remaining:
            before = ' "' if parameters else '"'
            name = self._choose(
                ids, remaining, before=before, after='":',
                show=lambda text: self._show_call(
                    function.name, shown + [text]
                ),
            )
            remaining.remove(name)
            # The last value also closes "parameters" and the whole call.
            end = "," if remaining else "}}"
            parameters[name] = self._choose_value(
                ids, function.parameters[name].type, end,
                show=lambda text: self._show_call(
                    function.name, shown + [f"{name}={text}"]
                ),
            )
            shown.append(f"{name}={json.dumps(parameters[name])}")
        self._show_call(function.name, shown, closed=True)
        if self.on_call_done is not None:
            self.on_call_done()
        return FunctionCallResult(
            prompt=prompt, name=function.name, parameters=parameters
        )

    def _encode(self, text: str) -> list[int]:
        """Tokenize ``text`` into a flat list of token ids."""
        return [int(i) for i in self.model.encode(text)[0]]

    def _build_prompt(self, prompt: str) -> str:
        """Fill the chat template with the system message and ``prompt``.

        Functions are listed in the system prompt (at ``{tools}``) as one
        JSON signature per line, the way Qwen's Hermes-style template does.
        The chat template comes from the config, so each model gets its own
        turn markers; it ends where the assistant's answer starts.
        """
        tools = "\n".join(
            json.dumps(self._tool_schema(function))
            for function in self.functions.values()
        )
        system = self.config.system_prompt.replace("{tools}", tools)
        return self.config.chat_template.format(system=system, user=prompt)

    @staticmethod
    def _tool_schema(function: FunctionDefinition) -> dict[str, Any]:
        """Convert ``function`` to the JSON-schema form used for tools."""
        return {
            "type": "function",
            "function": {
                "name": function.name,
                "description": function.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {"type": spec.type}
                        for name, spec in function.parameters.items()
                    },
                    "required": list(function.parameters),
                },
            },
        }

    def _show_call(
        self,
        name: str,
        arguments: list[str] | None = None,
        closed: bool = False,
    ) -> None:
        """Send the call written so far, as ``name(a=1, b``, to the user.

        ``arguments`` is None while the function name is still being
        written; ``closed`` adds the final parenthesis.
        """
        if self.on_call_update is None:
            return
        text = name
        if arguments is not None:
            text += "(" + ", ".join(arguments) + (")" if closed else "")
        self.on_call_update(text)

    def _choose_value(
        self,
        ids: list[int],
        value_type: str,
        end: str,
        show: Callable[[str], None] | None = None,
    ) -> str | float | int:
        """Force the model to write `` <value><end>``; return the value.

        Tokens are checked by their text, character by character, so the
        model may merge the quotes with what surrounds them, as it does in
        normal text: `` "[`` or ``]",`` or ``"}}``. Past
        ``max_value_tokens``, only tokens that close the value are allowed.
        ``show``, if given, receives the value written so far after each
        token, without the space before it or ``end``.
        """
        if value_type not in ("string", "number", "integer"):
            raise ValueError(f"unsupported parameter type {value_type!r}")
        done: State = ("end", len(end))

        def state_after(written: list[int]) -> State:
            state = START
            for token in written:
                state = self._value_table(value_type, end, state)[token]
            return state

        def show_value(written: list[int]) -> None:
            if show is None:
                return
            text = self.model.decode(written)
            phase, matched = state_after(written)
            if phase == "end":
                text = text[:len(text) - matched]
            show(text.strip())

        def allowed_next(written: list[int]) -> set[int]:
            state = state_after(written)
            if state == done:
                return set()
            table = self._value_table(value_type, end, state)
            if len(written) >= self.config.max_value_tokens:
                closing = {t for t, after in table.items()
                           if after[0] == "end"}
                if closing:
                    return closing
            return set(table)

        written = self._generate_with(ids, allowed_next, show_value)
        raw = self.model.decode(written)[:-len(end)].strip()
        if value_type == "string":
            return str(json.loads(raw))
        if value_type == "integer":
            return int(raw)
        return float(raw)

    def _value_table(
        self, value_type: str, end: str, state: State
    ) -> dict[int, State]:
        """Map each token allowed in ``state`` to the state it leads to."""
        key = (value_type, end, state)
        if key not in self._tables:
            table: dict[int, State] = {}
            for token, text in self._token_text.items():
                after = _advance(value_type, end, state, text)
                if after is not None:
                    table[token] = after
            self._tables[key] = table
        return self._tables[key]

    def _choose(
        self,
        ids: list[int],
        candidates: list[str],
        before: str = "",
        after: str = '"',
        show: Callable[[str], None] | None = None,
    ) -> str:
        """Force the model to write ``before + candidate + after``.

        Return the chosen candidate. Encoding the surrounding text with
        the candidate gives the model its usual tokens, such as ``":``.
        ``show``, if given, receives the candidate written so far after each
        token. It strips the characters of ``before`` and ``after`` from
        both ends, so candidates must not start or end with them.
        """

        def show_candidate(written: list[int]) -> None:
            if show is not None:
                show(self.model.decode(written).strip(before + after))

        index = self._choose_tokens(
            ids,
            [self._encode(before + candidate + after)
             for candidate in candidates],
            show_candidate,
        )
        return candidates[index]

    def _choose_tokens(
        self,
        ids: list[int],
        candidates: list[list[int]],
        on_token: Callable[[list[int]], None] | None = None,
    ) -> int:
        """Force the model to write one of ``candidates``; return its index.

        Only the next token of a candidate that matches what was written so
        far is allowed. Duplicate candidates resolve to the first one.
        """
        if not candidates:
            raise ValueError("no candidates to choose from")

        def allowed_next(written: list[int]) -> set[int]:
            position = len(written)
            return {
                tokens[position]
                for tokens in candidates
                if position < len(tokens) and tokens[:position] == written
            }

        return candidates.index(
            self._generate_with(ids, allowed_next, on_token)
        )

    def _generate_with(
        self,
        ids: list[int],
        allowed_next: Callable[[list[int]], set[int]],
        on_token: Callable[[list[int]], None] | None = None,
    ) -> list[int]:
        """Generate tokens restricted by ``allowed_next`` and return them.

        ``allowed_next`` receives the tokens written so far and returns the
        tokens allowed next; an empty set ends generation. When only one
        token is allowed it is appended without asking the model.
        ``on_token``, if given, receives the tokens written so far after
        each token.
        """
        written: list[int] = []
        allowed = allowed_next(written)
        while allowed:
            if len(allowed) == 1:
                token = next(iter(allowed))
            else:
                token = self._best_allowed(ids, allowed)
            ids.append(token)
            written.append(token)
            if on_token is not None:
                on_token(written)
            allowed = allowed_next(written)
        return written

    def _best_allowed(self, ids: list[int], allowed: set[int]) -> int:
        """Return the token of ``allowed`` the model scores highest.

        This is greedy decoding: no randomness, so the same prompt always
        gives the same call.
        """
        logits = self._logits(ids)
        return max(allowed, key=lambda token: logits[token])

    def _logits(self, ids: list[int]) -> list[float]:
        """Return the next-token logits for ``ids``.

        Use the SDK's KV-cached method when it exists (our extended
        llm_sdk), and fall back to the original one otherwise.
        """
        if hasattr(self.model, "get_logits_cached"):
            return self.model.get_logits_cached(ids)
        return self.model.get_logits_from_input_ids(ids)
