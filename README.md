*This project has been created as part of the 42 curriculum by zleullie.*

# call me maybe

Turn natural-language requests into structured function calls with a small
language model (Qwen/Qwen3-0.6B), using constrained decoding so that every
output is valid, schema-compliant JSON.

## Description

Given a list of available functions and a prompt such as *"What is the sum of
2 and 3?"*, the program does not answer `5`. It returns the function to call
and its typed arguments:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

A 0.6B-parameter model asked to "please answer in JSON" fails often. This
project does not rely on the model behaving. It drives the model one token at
a time and, at every step, only lets it choose among the tokens that keep the
output valid: a real function name, a real parameter name, and a value of the
declared type. The model still makes every *choice* (which function, which
values); the decoder only removes the choices that would break the format.

Overview of the pipeline:

1. Load and validate the function definitions and prompts (pydantic).
2. Build a Qwen3 chat prompt that lists the functions in Qwen's own
   tool-calling format.
3. Optionally let the model think freely (off by default, see
   [Design decisions](#design-decisions)).
4. Generate the call under constraints: function name, then each parameter
   name and value.
5. Write all results to `data/output/function_calls.json`.

## Instructions

### Requirements

- Linux, Python 3.10 to 3.14, and [uv](https://docs.astral.sh/uv/).
- Optional: an AMD GPU with ROCm. The project is set up for two targets:
  `gfx1200` (RX 9060 XT) and `gfx1034` (RX 6500, 42 computers). Without a
  GPU the SDK falls back to the CPU, which works but is much slower.
- The model is downloaded from Hugging Face on the first run. A `HF_TOKEN` is
  not required; if you have one, put `HF_TOKEN=...` in a `.env` file and
  `make run` will load it.

### Install

```bash
make install
```

This runs `uv sync`. The Makefile reads the GPU target from `rocminfo` and
adds the matching extra (`--extra gfx1200` or `--extra gfx1034`). To pick it
yourself:

```bash
make install GPU=gfx1034
```

The reviewer runs plain `uv sync`. It installs the `gfx1200` build of torch,
because `llm-sdk[gfx1200]` is a base dependency in `pyproject.toml`.

### Run

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

| Option | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calls.json` |
| `--prompts_config` | `config/prompts.json` (prompt texts and limits) |

### Makefile

| Rule | What it does |
|---|---|
| `make install` | `uv sync`, with the GPU extra if one is detected |
| `make run` | `python -m src` with the default paths |
| `make debug` | Runs the program under `pdb` |
| `make lint` | `flake8 .` and `mypy` with the flags required by the subject |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make clean` | Removes caches and `data/output/` |

## Example usage

Default paths:

```bash
uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Force the CPU, single-threaded (see [Performance](#performance-analysis)):

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 uv run python -m src
```

Each call is printed on stderr as it is found, and the full result goes to the
output file:

```text
fn_add_numbers(a=2.0, b=3.0)
fn_greet(name="shrek")
fn_reverse_string(s="hello")
fn_get_square_root(a=16.0)
fn_substitute_string_with_regex(source_string="Programming is fun", regex="([aeiou])", replacement="*")
```

```json
[
  {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}},
  {"prompt": "Greet shrek", "name": "fn_greet", "parameters": {"name": "shrek"}}
]
```

When thinking is turned on in `config/prompts.json` and stderr is a terminal,
the model's reasoning streams in gray and is erased once it is done.

## Algorithm explanation

### What the model sees

The prompt follows Qwen3's chat template and its Hermes-style tool format.
Each function from `functions_definition.json` is converted to the JSON
schema Qwen was trained on and listed inside `<tools></tools>`. The
assistant turn is then pre-filled up to the first character the model has to
choose:

```text
<|im_start|>system
You are an expert tool-calling assistant ... <tools>
{"type": "function", "function": {"name": "fn_greet", ...}}
</tools> ...<|im_end|>
<|im_start|>user
Greet shrek<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "
```

The empty `<think></think>` block is what Qwen3 itself writes when thinking
is disabled. Everything up to `{"name": "` is fixed text written by the
program, not by the model.

### One generic constrained loop

All constrained generation goes through one function, `_generate_with(ids,
allowed_next)`. `allowed_next` receives the tokens written so far and returns
the set of tokens allowed next; an empty set means "done".

```text
loop:
    allowed = allowed_next(written)
    if allowed is empty: stop
    if only one token is allowed: append it without calling the model
    else: logits = model(ids); append the allowed token with the highest logit
```

Choosing the best allowed token is the same as setting every other logit to
negative infinity and taking the argmax (greedy decoding). Two kinds of
`allowed_next` are built on top of it.

### Choosing among fixed strings (function and parameter names)

`_choose` turns each candidate into token ids, together with the text around
it (`fn_greet"` for a function name, ` "name":` for a parameter name), so the
model gets the same tokens it would produce naturally, such as `":`. At each
position, the allowed tokens are the next token of every candidate that
matches what was written so far. When only one candidate is left, its
remaining tokens are forced without running the model.

The call is built like this:

1. Choose the function name among all function names.
2. Append `, "arguments": {`.
3. While some parameters are missing: let the model choose the next
   parameter name among the remaining ones, then generate its value. The
   value ends with `,`, or with `}}` for the last one.

So every required parameter is always present, none is repeated, and no
unknown key can appear.

### Generating values with a character state machine

Strings and numbers have no fixed list of candidates. Their format is
described by a small state machine over characters, for the text
`<optional space><value><end>`:

| Phase | Allowed next characters |
|---|---|
| `start` | a space (→ `space`), or the first character of the value |
| string: after the opening `"` | anything but a control character; `\` → `escape`; `"` → `end` |
| string: `escape` | one of `" \ / b f n r t` → back to the string |
| number | an optional leading `-`, digits, at most one `.` followed by digits |
| `end` | the characters of `end` (`,` or `}}`), in order |

A token is allowed if **every character of its text** can be fed to the
state machine from the current state. The token must not run past the end
(`",` is fine for `end = ","`, `", "` is not). This character-level check is
what lets the model use its usual merged tokens, such as ` "[`, `]",` or
`"}}`, instead of being forced to write each quote as a separate token.

For each `(type, end, state)` the set of allowed tokens, and the state each
one leads to, is computed once over the whole vocabulary and cached. Later
steps are dictionary lookups.

To make sure a value always ends, after `max_value_tokens` (64) tokens only
tokens that close the value are allowed. The written text is then parsed:
`json.loads` for strings, `float` for numbers. Numbers are always output as
floats (`2` → `2.0`), as in the subject's examples.

### From token ids to text

`vocab.json` stores tokens in byte-level BPE form (`Ġ` for a space, `Ċ` for a
newline), so its keys are not the real text. At startup, each token id is
decoded once with the SDK's `decode` to get its actual text, and tokens that
decode to an empty string are dropped (they could loop forever without
progress). This takes about 0.4 s.

## Design decisions

- **The program writes the JSON structure, the model fills the gaps.** Rather
  than a full JSON grammar, the fixed parts (`{"name": "`, `, "arguments":
  {`) are appended directly and only the variable parts are constrained.
  This is simpler to check and saves model calls.
- **Qwen's native tool format.** Listing tools the way Qwen3 was trained to
  read them (rendered from the model's own chat template) keeps the prompt
  "in distribution" for the model.
- **Greedy decoding.** Function calling has one right answer, so randomness
  only adds errors. Greedy decoding also makes runs reproducible.
- **Forced tokens skip the model.** When only one token is allowed, no
  forward pass is run. Once a function name is unambiguous, the rest of it
  costs nothing.
- **Thinking is optional and off by default.** With `"think": true` the model
  first reasons freely (up to `max_think_tokens`, then it is told to answer),
  starting from `Okay, the user wants to`, which is how Qwen3 almost always
  starts anyway. It can help on ambiguous prompts but costs hundreds of
  forward passes per prompt, so the default config disables it.
- **Prompts live in `config/prompts.json`.** System prompt, chat template,
  prefixes and limits are data, validated by the `PromptConfig` pydantic
  model, so they can be tuned without touching the decoder.
- **Few-shot examples for units and percentages.** The system prompt shows
  how to turn `8%` into `0.08` and `20kOhm` into `20000`, because these were
  the most common value errors on the benchmark (see below).
- **pydantic everywhere.** Function definitions, prompts, results and the
  prompt config are all pydantic models, and files are read and written
  through `TypeAdapter`.
- **stderr for progress, file for results.** The terminal display (thinking
  in gray, then `name(args)` lines) never mixes with the output file. The
  live thinking display is only enabled when stderr is a terminal, because
  erasing it needs cursor movement escape codes.
- **ROCm through uv.** `llm_sdk` is a uv workspace member with one extra per
  GPU target. Each extra pulls the ROCm build of torch and the matching
  runtime from AMD's package index.

## Performance analysis

### Reliability

The output is valid JSON by construction: the JSON structure is written by
the program, names are chosen from fixed lists, and values can only be
completed along the state machine before being serialised by pydantic.
Every parameter of the chosen function is present exactly once, with the
declared type.

### Accuracy

- **Sample inputs (`data/input`):** 11/11 correct function choices, 10/11
  fully correct calls. The miss: for *"Replace all numbers in ..."* the model
  wrote the regex `34` instead of something like `\d+`. The structure is
  guaranteed; the content of a string is still the model's choice.
- **Berkeley Function Calling Leaderboard (`simple_python`):** during
  development I converted this dataset to the project's format. 302
  questions were usable (98 skipped for types the decoder does not support,
  such as arrays and booleans, or for a mistake in the dataset). On the
  baseline run, before the few-shot examples were added to the prompt:
  - no wrong function choice, no missing or extra parameter, no value of the
    wrong type;
  - 62 of 302 failed a strict exact-value comparison;
  - after reading each failure, about 28 were real mistakes (percent written
    as `5` instead of `0.05`, wrong unit scale, wrong date format, wrong
    choice between options) and the rest were the scorer being stricter than
    the question, so roughly **79 % strict / 91 % lenient**.

  The prompt changes were meant to fix the percent and unit cases. I did not
  record a full re-run after them, so I can't give a number for the current
  prompt.

### Speed

- **GPU (RX 9060 XT, thinking off):** the 11 sample prompts take about 25 s
  end to end, model loading included. The subject's limit is 5 minutes.
- **CPU:** around 2 to 6 s per forward pass on one thread, depending on prompt
  length (about 120 to 410 tokens). The SDK has no KV cache, so every new
  token re-processes the whole sequence; a longer prompt is slower at every
  step. On this machine's torch build, more threads were slower (a 50-token
  pass: 1.8 s on 1 thread, 12.2 s on 6), hence `OMP_NUM_THREADS=1`.
- What keeps it fast: thinking off, forced tokens without a model call, and
  the cached token tables.

### Known limitations

- Only the `string` and `number` parameter types are supported. Any other
  type (for example `boolean` or `integer`) raises an error.
- Numbers cannot use exponent notation (`1e-9`), so the model has to write
  them out in full. Leading zeros (`007`) are accepted and parsed as `7.0`.
- `\uXXXX` escapes are not supported inside strings.
- Missing or invalid input files are not yet caught in `main`, so they end
  with a Python traceback instead of a clear error message.
- The `gfx1034` setup has only been resolved and locked, not run on a 42
  computer, and `uv sync` has not been tested on a fresh clone.

## Challenges faced

- **Installing ROCm torch with uv.** The SDK depends on plain `torch`, and
  the ROCm builds live on AMD's own index. uv only reads indexes from the
  root project, so the index, `index-strategy = "unsafe-best-match"` and one
  extra per GPU had to be declared there. A broken install
  (`_rocm_sdk_libraries/__init__.py` missing from the venv) was fixed by
  reinstalling the package from uv's cache.
- **The vocabulary is not text.** Comparing `vocab.json` keys to characters
  never matched spaces or newlines, because they are stored as `Ġ` and `Ċ`.
  Decoding each token id once gave the real text to work with.
- **Quotes merged with other characters.** The first version only accepted a
  bare `"` to close a string. The model wanted to write `]",` and couldn't,
  so it kept writing the rest of the JSON *inside* the string until the
  token limit, producing values like
  `regex="aeiouAEIOU\", \"replacement\": ..."`. Checking each token's text
  character by character with the state machine fixed it.
- **Slow generation.** It looked like a bug in the decoder, so a minimal
  separate decoder was written to compare. It was just as slow: the cost
  comes from re-running the model on the full prompt at every step. Shorter
  prompts, thinking off and skipping forced tokens were the useful levers.
- **Values that are well-formed but wrong.** Percentages, units and dates
  were the main errors on the benchmark. Constraints can't fix these;
  few-shot examples in the system prompt target them.
- **Showing the thinking live.** One character can be split over several
  tokens (decoding half of it gives `�`), so text is held back until it is
  complete. Erasing the gray text means counting how many rows were printed,
  including lines the terminal wrapped. It also does not work in PyCharm's
  Run console, which is not a terminal unless "Emulate terminal" is on.

## Testing strategy

- **Lint and types:** `make lint` (flake8 and mypy with the subject's flags)
  passes on the project.
- **Fake model during development:** the decoder was checked against a fake
  model that uses the real Qwen tokenizer and always picks the tokens of a
  target call. Cases covered: negative and decimal numbers, strings with
  escaped quotes and backslashes (`\\d+`), merged tokens such as `":` and
  `",`, duplicate candidates, thinking that hits the token limit, and
  multi-byte characters split across tokens. These were throwaway scripts;
  there is no test suite in the repository.
- **End-to-end runs** on `data/input` after each change, checking the
  function names, argument names and types in the output file.
- **Benchmark:** the BFCL `simple_python` run described above, with each
  failure read by hand to separate real mistakes from strict scoring.

Edge cases still worth testing before an evaluation: empty strings, very large
numbers, special characters and quotes in prompts, ambiguous prompts that fit
two functions, functions with many parameters, and missing or malformed input
files.

## Resources

### Function calling and constrained decoding

- [Qwen: Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html):
  Qwen's tool-call prompt format.
- [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B):
  the model, its chat template and thinking mode.
- Willard and Louf, [Efficient Guided Generation for Large Language
  Models](https://arxiv.org/abs/2307.09702) (2023): constrained decoding
  with finite-state machines over the vocabulary.
- [llama.cpp GBNF grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md):
  another practical take on grammar-constrained sampling.
- [Hugging Face: Generation strategies](https://huggingface.co/docs/transformers/generation_strategies):
  greedy decoding, logits processors.
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html):
  the source of the benchmark questions.

### Tokenization and JSON

- Sennrich, Haddow and Birch, [Neural Machine Translation of Rare Words with
  Subword Units](https://arxiv.org/abs/1508.07909) (2016): byte-pair
  encoding.
- [RFC 8259](https://datatracker.ietf.org/doc/html/rfc8259) and
  [json.org](https://www.json.org/json-en.html): the JSON grammar for strings
  and numbers.

### Tools

- [pydantic documentation](https://docs.pydantic.dev/latest/)
- [uv documentation](https://docs.astral.sh/uv/) (workspaces, indexes,
  optional dependencies)
- [ANSI escape codes](https://en.wikipedia.org/wiki/ANSI_escape_code): the
  gray text and the erasing of the thinking display.

### How AI was used

I used Claude Code (Anthropic) throughout the project, over nine sessions on
15 and 16 October 2026. The summary below comes from those session logs.
Everything that stayed in the project, I reviewed and can explain.

**Learning and explanations.** Before writing the decoder I asked how to
approach the project and which vocabulary to look up (logits, greedy
decoding, BPE, logit masking), how generation works with the SDK, what
`encode` returns (a `[1, n]` tensor), whether a KV cache was feasible, why the
CPU was so slow, and what the decoder does and does not enforce (return
types, negative numbers, exponents).

**Environment and tooling.** Claude set up ROCm torch through uv (the AMD
index and the `gfx1200` and `gfx1034` extras), the GPU detection and `.env`
loading in the Makefile, the `accelerate` dependency, typing-only fixes and a
`py.typed` marker in `llm_sdk`, and the flake8 exclude for extra virtual
environments. It also diagnosed a broken ROCm install and PyCharm run
configuration problems.

**Code in `src/`.** I wrote the project structure, the pydantic models, the
file loading helpers and a first version of the decoder: the prompt, two
loops for choosing names, and a first, crude version of the generic
`_generate_with(ids, allowed_next)` loop, which I designed. I then asked
Claude to:

- split the decoder into functions and move the prompts into
  `config/prompts.json`;
- refactor the name choice into `_choose_tokens`, and my
  `_generate_with` loop into its current, cleaner form;
- add value generation for strings and numbers, then replace it with the
  character-level state machine when merged tokens broke string closing.
  This is the core of the constraint logic, and it was written by Claude at
  my request; I reviewed it afterwards;
- add the thinking display (callbacks, gray streaming, erasing),
  `think_prefix` and the `think` config flag;
- switch the prompts to Qwen's tool format, output numbers as floats, and fix
  the default output path to the one in the subject.

**Prompt engineering.** Claude analysed the benchmark failures and suggested
prompt changes. I picked which ones to apply (remove "you may guess
parameters", add few-shot examples for percentages, units and locations).

**Benchmark tooling.** Claude wrote a converter for the BFCL `simple_python`
dataset and a runner that scored outputs against expected answers. These were
development tools and are not part of the submitted project. One lesson from
this: a benchmark run started by Claude overwrote 113 result files from a
25-minute run of mine. The runner was changed to write each run to a new
folder and never delete anything, and I now check what an agent is about to
run before letting it touch outputs.

**Review and documentation.** Claude checked the project against the subject
before delivery (finding the lint failure, the wrong output file name and
this README being a stub) and drafted this README from the subject and the
session logs.

What I did not delegate: deciding the overall design (generic constrained
loop, config file, thinking on or off), noticing that Qwen's thoughts always
start with "Okay, the user wants to", proposing the minimal-prompt
experiment, running the benchmarks, and choosing which suggestions to accept.
