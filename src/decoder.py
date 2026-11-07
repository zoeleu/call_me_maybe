"""Constrained decoding: mask logits so only valid tokens can be chosen."""

import json
from pathlib import Path
from typing import Any, Callable, Collection, Literal

import numpy as np
from llm_sdk import Small_LLM_Model

from .io_utils import load_vocab
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
        on_think_token: Callable[[str], None] | None = None,
        on_think_done: Callable[[], None] | None = None,
        on_call_update: Callable[[str], None] | None = None,
        on_call_done: Callable[[], None] | None = None,
    ) -> None:
        """Store the model, the functions and the prompt configuration.

        ``on_think_token``, if given, receives the new thinking text after
        each token. ``on_think_done``, if given, is called once thinking is
        over. ``on_call_update``, if given, receives the whole call written
        so far, as ``name(a=1, b``, after each token; ``on_call_done`` is
        called once the call is complete. With ``config.think`` False the
        prompt holds an empty thinking block, as Qwen3 does when thinking is
        disabled, and the model answers at once.
        """
        if not functions:
            raise ValueError("at least one function definition is required")
        self.model = model
        self.functions = {function.name: function for function in functions}
        self.config = config
        self.on_think_token = on_think_token
        self.on_think_done = on_think_done
        self.on_call_update = on_call_update
        self.on_call_done = on_call_done
        vocab = load_vocab(Path(model.get_path_to_vocab_file()))
        # vocab.json keys are byte-level BPE ("Ġ" for a space), so ask the
        # model for the real text of each token instead. Empty tokens are
        # dropped: they would let generation loop without making progress.
        self._token_text: dict[int, str] = {}
        self._rng = np.random.default_rng(42)
        for token in vocab.values():
            text = model.decode([token])
            if text:
                self._token_text[token] = text
        self._think_end = self._single_token(config.think_end)
        # Cache of _value_table results, filled on first use.
        self._tables: dict[tuple[str, str, State], dict[int, State]] = {}

    def generate_call(self, prompt: str) -> FunctionCallResult:
        """Choose a function and its arguments for ``prompt`` using the LLM."""
        ids = self._encode(self._build_prompt(prompt))
        if self.config.think:
            self._think(ids)
        else:
            # The empty thinking block already ends with the blank line.
            ids.extend(self._encode(self.config.answer_prefix.lstrip("\n")))
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

    def _single_token(self, text: str) -> int:
        """Return the id of ``text``, which must be exactly one token."""
        tokens = self._encode(text)
        if len(tokens) != 1:
            raise ValueError(f"{text!r} must be a single token")
        return tokens[0]

    def _build_prompt(self, prompt: str) -> str:
        """Fill the chat template with the system message and ``prompt``.

        Functions are listed in the system prompt (at ``{tools}``) as one
        JSON signature per line, the way Qwen's Hermes-style template does.
        """
        tools = "\n".join(
            json.dumps(self._qwen_tool(function))
            for function in self.functions.values()
        )
        system = self.config.system_prompt.replace("{tools}", tools)
        text = self.config.chat_template.format(system=system, user=prompt)
        if self.config.think:
            return text + self.config.think_prefix
        return text + "\n\n" + self.config.think_end + "\n\n"

    @staticmethod
    def _qwen_tool(function: FunctionDefinition) -> dict[str, Any]:
        """Convert ``function`` to the JSON-schema form Qwen was trained on."""
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

    def _select(self, logits: list[float], valid: list[int] | None = None) -> int:
        """Implement proper token selection, as Qwen recommends.
        Greedy decoding is explicitly discouraged for Qwen models.
        """
        temperature = self.config.temperature
        top_p = self.config.top_p
        top_k = self.config.top_k
        min_p = self.config.min_p

        scores = np.array(logits)

        # Mask invalid tokens so they are never chosen.
        if valid is not None:
            mask = np.ones_like(scores, dtype=bool)
            mask[valid] = False
            scores[mask] = -np.inf

        scores /= temperature

        # top_k > 0 in the config model, no need to check again
        k = min(top_k, len(scores))
        if k < len(scores):
            top_k_indices = np.argpartition(scores, -k)[-k:]
            mask = np.ones_like(scores, dtype=bool)
            mask[top_k_indices] = False
            scores[mask] = -np.inf

        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()

        if top_p < 1.0:
            # we want a descending order to get only the most probable tokens
            order = np.argsort(probabilities)[::-1]
            # sorted list of the probabilities
            probabilities_sorted = probabilities[order]
            # the running total of probabilities, minus its own probability
            accumulated_ps = np.cumsum(probabilities_sorted) - probabilities_sorted
            # filter out less likely tokens
            probabilities[order[accumulated_ps >= top_p]] = 0.0

        if min_p > 0.0:
            probabilities[probabilities < min_p * probabilities.max()] = 0.0
        
        # normalize remaining values
        probabilities /= probabilities.sum()

        # draw the final token
        return int(self._rng.choice(len(probabilities), p=probabilities))

    def _think(self, ids: list[int]) -> None:
        """Let the model think freely, then open the tool call.

        Generation is greedy and stops at ``think_end``. If the model is
        still thinking after ``max_think_tokens``, ``force_answer`` is
        appended to cut it off. ``think_prefix`` is already in the prompt, so
        it is shown first, as if the model had just written it.
        """
        if self.on_think_token is not None:
            self.on_think_token(self.config.think_prefix)
        thought: list[int] = []
        shown = ""
        for _ in range(self.config.max_think_tokens):
            token = self._select(self.model.get_logits_from_input_ids(ids))
            ids.append(token)
            if token == self._think_end:
                break
            thought.append(token)
            shown = self._report_thought(thought, shown)
        else:
            ids.extend(self._encode(self.config.force_answer))
        if self.on_think_done is not None:
            self.on_think_done()
        ids.extend(self._encode(self.config.answer_prefix))

    def _report_thought(self, thought: list[int], shown: str) -> str:
        """Send the thinking text not shown yet to ``on_think_token``.

        A token can hold part of a multi-byte character, which decodes as
        "\\ufffd"; that text is held back until a later token completes it.
        Return the text shown so far.
        """
        if self.on_think_token is None:
            return shown
        text = self.model.decode(thought)
        if text.endswith("�") or len(text) == len(shown):
            return shown
        self.on_think_token(text[len(shown):])
        return text

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
    ) -> str | float:
        """Force the model to write `` <value><end>``; return the value.

        Tokens are checked by their text, character by character, so the
        model may merge the quotes with what surrounds them, as it does in
        normal text: `` "[`` or ``]",`` or ``"}}``. Past
        ``max_value_tokens``, only tokens that close the value are allowed.
        ``show``, if given, receives the value written so far after each
        token, without the space before it or ``end``.
        """
        if value_type not in ("string", "number"):
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
        """Return the allowed token the model rates highest."""
        logits = self.model.get_logits_from_input_ids(ids)
        return self._select(logits, list(allowed))
