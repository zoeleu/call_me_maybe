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

1. Load and validate the function definitions, the prompts and the prompt
   configuration (pydantic).
2. Build a Qwen3 chat prompt that lists the functions in Qwen's own
   tool-calling format.
3. Optionally let the model think freely (off by default, see
   [Design decisions](#design-decisions)).
4. Generate the call under constraints: function name, then each parameter
   name and value. Among the allowed tokens, the next one is drawn with the
   sampling settings Qwen recommends (temperature, top-k, top-p, min-p).
5. Show the call live on the terminal as it is written, then write all
   results to `data/output/function_calls.json`.

## Instructions

### Requirements

- Linux, Python 3.10 to 3.14, and [uv](https://docs.astral.sh/uv/).
- Optional: an AMD GPU with ROCm. The project is set up for two targets:
  `gfx1200` (RX 9060 XT) and `gfx1034` (RX 6500, 42 computers). Without a
  GPU, the `cpu` extra installs the CPU-only build of torch.
- The model is downloaded from Hugging Face on the first run. A `HF_TOKEN` is
  not required; if you have one, put `HF_TOKEN=...` in a `.env` file and
  `make run` will load it.

### Install

```bash
make install
```

This runs `uv sync`. The Makefile reads the GPU target from `rocminfo` and
adds the matching extra (`--extra gfx1200` or `--extra gfx1034`). When
`rocminfo` is missing or finds no GPU, it uses `--extra cpu`. To pick it
yourself:

```bash
make install GPU=gfx1034
```

On 42 computers, if `~/goinfre` exists, the Makefile puts everything large
there: the uv cache, the virtual environment, uv's Python installs and the
Hugging Face cache (the model). The home directory quota is too small for
torch and the model, and goinfre is fast local storage. It is ephemeral, so
after a reset `make install` simply downloads everything again. Outside 42
nothing changes, and any of these variables can still be overridden from the
environment.

Plain `uv sync` (no extra) installs torch from PyPI. Use `make install`, or
pass `--extra cpu`, `--extra gfx1200` or `--extra gfx1034` yourself. Do not
run a ROCm build without a usable GPU: its CPU fallback uses OpenBLAS and was
about 30 times slower than the CPU build here (35 s against 1 s for one
300-token forward pass).

### Run

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>] [--prompts_config <file>]
```

| Option | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calls.json` |
| `--prompts_config` | `config/prompts.json` (prompt texts, limits, sampling) |

Outside 42, `uv run` works as shown. On a 42 computer, use `make run` so the
goinfre paths above are set.

### Makefile

| Rule | What it does |
|---|---|
| `make install` | `uv sync`, with the GPU extra if one is detected, else `cpu` |
| `make run` | `python -m src` with the default paths |
| `make debug` | Runs the program under `pdb` |
| `make lint` | `flake8 .` and `mypy` with the flags required by the subject |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make clean` | Removes caches and `data/output/` |

### Prompt configuration

`config/prompts.json` holds everything about the prompt and decoding that is
data rather than code: the system prompt, the chat template, the prefixes the
program writes itself, the thinking settings, the token limits and the
sampling settings. It is validated by the `PromptConfig` pydantic model.

`config/prompts.schema.json` is the JSON schema of that model, and
`prompts.json` points to it with `"$schema"`, so an editor such as VS Code can
autocomplete the keys and flag invalid values (for example a negative
`top_k`). After changing `PromptConfig`, regenerate it with:

```bash
uv run python -m src.models
```

## Example usage

Default paths:

```bash
make run
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

When stderr is a terminal, each call is drawn in gray while the model writes
it, redrawn on every token:

```text
fn_
fn_add
fn_add_numbers(
fn_add_numbers(a
fn_add_numbers(a=2
fn_add_numbers(a=2.0, b
fn_add_numbers(a=2.0, b=3.0)
```

(shown here one state per line; on the terminal it is a single line that
updates in place). Once the call is complete the gray line is erased and the
final call is printed. The full result goes to the output file:

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

When thinking is turned on in `config/prompts.json`, the model's reasoning
streams in gray first and is erased once it is done, before the call is
drawn.

## Algorithm explanation

### What the model sees

The prompt follows Qwen3's chat template and its Hermes-style tool format.
Each function from `functions_definition.json` is converted to the JSON
schema Qwen was trained on and listed inside `<tools></tools>`. The
assistant turn is then pre-filled up to the first character the model has to
choose:

```text
<|im_start|>system
You call tools. Tools: <tools>
{"type": "function", "function": {"name": "fn_greet", ...}}
</tools>. To call, start with <tool_call>.
<|im_end|>
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
    else: logits = model(ids); draw one allowed token (see below) and append it
```

Two kinds of `allowed_next` are built on top of it, one for fixed strings
and one for free values. The loop also takes an optional `on_token` callback,
which the live display uses.

### Choosing a token among the allowed ones

`_select(logits, valid)` turns the model's logits into one token, following
the sampling settings Qwen recommends for Qwen3:

1. **Constraint:** every token outside `valid` gets a score of minus
   infinity, so it can never be drawn.
2. **Temperature:** the scores are divided by `temperature` (0.6). Below 1
   this sharpens the distribution towards the likely tokens.
3. **Top-k:** only the `top_k` (20) highest scores are kept.
4. Softmax turns the scores into probabilities.
5. **Top-p (nucleus):** tokens are sorted by probability and kept until
   their running total reaches `top_p` (0.95); the long tail is dropped.
6. **Min-p:** tokens below `min_p` times the best token's probability are
   dropped (0, so disabled in the current config).
7. The remaining probabilities are renormalised and one token is drawn.

The random generator is a numpy `default_rng(42)`, created once per decoder.
Thinking uses the same function without step 1.

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

### Showing the call while it is written

The decoder always knows which part of the call it is writing, so it builds
the `name(a=2.0, b` view directly instead of parsing half-written JSON:

- while choosing a name, the tokens written so far are decoded and the
  surrounding quotes, spaces and `:` are stripped (`_choose`);
- while writing a value, the state machine says how many characters of `end`
  were already written, and those are cut off (`_choose_value`);
- finished arguments are kept as `name=json.dumps(value)`.

After each token the whole line is sent to the `on_call_update` callback,
which erases the previous line and prints the new one.

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
- **Sampling instead of greedy decoding.** The first version always took the
  allowed token with the highest logit. Qwen's documentation explicitly
  discourages greedy decoding for Qwen3, because it can lead to repetition
  and worse answers, and recommends temperature 0.6, top-p 0.95, top-k 20
  and min-p 0. These values are now in `config/prompts.json`. The constraint
  is applied *before* sampling, so randomness can only pick between valid
  tokens; it can never break the format.
- **A fixed seed.** The generator is seeded with 42 so that a run can be
  reproduced. The generator is shared by all prompts, so a prompt's result
  depends on the prompts before it in the input file.
- **Forced tokens skip the model.** When only one token is allowed, no
  forward pass is run. Once a function name is unambiguous, the rest of it
  costs nothing.
- **A minimal system prompt.** The earlier prompt was a long instruction
  block with three few-shot examples (percentages, units, "city, STATE").
  The SDK has no KV cache, so every prompt token is processed again at every
  generation step, and on the CPU that made each step several times slower.
  The current prompt only says that the model calls tools, lists them, and
  says to start with `<tool_call>`; the constrained decoder enforces the rest.
  The trade-off is that the few-shot examples targeted real value errors (see
  [Accuracy](#accuracy)), and their effect is now lost.
- **Thinking is optional and off by default.** With `"think": true` the model
  first reasons freely (up to `max_think_tokens`, then it is told to answer),
  starting from `Okay, the user wants to`, which is how Qwen3 almost always
  starts anyway. It can help on ambiguous prompts but costs hundreds of
  forward passes per prompt, so the default config disables it.
- **Prompts live in `config/prompts.json`, with a JSON schema.** System
  prompt, chat template, prefixes, limits and sampling settings are data,
  validated by the `PromptConfig` pydantic model, so they can be tuned
  without touching the decoder. The schema is generated from the same model,
  so the editor and the program agree on what is valid.
- **pydantic everywhere.** Function definitions, prompts, results and the
  prompt config are all pydantic models, and files are read and written
  through `TypeAdapter`.
- **stderr for progress, file for results.** The terminal display (thinking
  in gray, the call being written, then `name(args)` lines) never mixes with
  the output file. The live display is only enabled when stderr is a
  terminal, because erasing needs cursor movement escape codes.
- **ROCm through uv.** `llm_sdk` is a uv workspace member with one extra per
  GPU target. Each extra pulls the ROCm build of torch and the matching
  runtime from AMD's package index.
- **goinfre on 42 computers.** torch, the ROCm runtime and the model do not
  fit in the home quota, so the Makefile moves uv's cache, the venv, Python
  and the Hugging Face cache to `~/goinfre` when it exists.

## Performance analysis

### Reliability

The output is valid JSON by construction: the JSON structure is written by
the program, names are chosen from fixed lists, and values can only be
completed along the state machine before being serialised by pydantic.
Every parameter of the chosen function is present exactly once, with the
declared type. Sampling does not change this, since it only chooses between
tokens the constraint already allows.

### Accuracy

The numbers below were measured with the **earlier setup: greedy decoding
and the long prompt** (the benchmark baseline was before the few-shot
examples were added). I have not measured the current setup (sampling and
the minimal prompt) yet, so they are a reference point, not the current
accuracy.

- **Sample inputs (`data/input`):** 11/11 correct function choices, 10/11
  fully correct calls. The miss: for *"Replace all numbers in ..."* the model
  wrote the regex `34` instead of something like `\d+`. The structure is
  guaranteed; the content of a string is still the model's choice.
- **Berkeley Function Calling Leaderboard (`simple_python`):** during
  development I converted this dataset to the project's format. 302
  questions were usable (98 skipped for types the decoder does not support,
  such as arrays and booleans, or for a mistake in the dataset). On the
  baseline run:
  - no wrong function choice, no missing or extra parameter, no value of the
    wrong type;
  - 62 of 302 failed a strict exact-value comparison;
  - after reading each failure, about 28 were real mistakes (percent written
    as `5` instead of `0.05`, wrong unit scale, wrong date format, wrong
    choice between options) and the rest were the scorer being stricter than
    the question, so roughly **79 % strict / 91 % lenient**.

### Speed

- **Where the time goes:** almost all of it is the model's forward pass. The
  SDK's `get_logits_from_input_ids` has no KV cache, so every generated token
  runs the model on the whole sequence again, prompt included. The cost of a
  step grows with the prompt length, which is why the prompt was shortened.
  The decoder's own Python work (token tables, masks, sampling over the
  151k-token vocabulary) is in the tens of milliseconds per step.
- **GPU (RX 9060 XT, thinking off, long prompt):** the 11 sample prompts took
  about 25 s end to end, model loading included. The subject's limit is 5
  minutes.
- **CPU:** around 2 to 6 s per forward pass on one thread with the long
  prompt, depending on its length (about 120 to 410 tokens). On my own
  machine's torch build, more threads were slower (a 50-token pass: 1.8 s on
  1 thread, 12.2 s on 6), hence `OMP_NUM_THREADS=1`. On a busy shared
  machine, other programs competing for the cores also slow it down a lot.
- What keeps it fast: a short prompt, thinking off, forced tokens without a
  model call, and the cached token tables.

### Known limitations

- Only the `string` and `number` parameter types are supported. Any other
  type (for example `boolean` or `integer`) raises an error.
- Numbers cannot use exponent notation (`1e-9`), so the model has to write
  them out in full. Leading zeros (`007`) are accepted and parsed as `7.0`.
- `\uXXXX` escapes are not supported inside strings.
- Results are reproducible only for the same input file in the same order,
  because one seeded generator is shared by all prompts.
- The accuracy of the current setup (sampling, minimal prompt) has not been
  measured.
- Missing or invalid input files are not yet caught in `main`, so they end
  with a Python traceback instead of a clear error message.
- The `gfx1034` setup has only been resolved and locked, not run on a 42
  computer's GPU, and `uv sync` has not been tested on a fresh clone.

## Challenges faced

- **Installing ROCm torch with uv.** The SDK depends on plain `torch`, and
  the ROCm builds live on AMD's own index. uv only reads indexes from the
  root project, so the index, `index-strategy = "unsafe-best-match"` and one
  extra per GPU had to be declared there. A broken install
  (`_rocm_sdk_libraries/__init__.py` missing from the venv) was fixed by
  reinstalling the package from uv's cache.
- **Disk space on 42 computers.** The venv, torch and the model do not fit in
  the home quota, which led to moving all caches to goinfre from the
  Makefile.
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
  few-shot examples in the prompt targeted them, at a speed cost that later
  led to the minimal prompt.
- **Showing the thinking and the call live.** One character can be split
  over several tokens (decoding half of it gives `�`), so thinking text is
  held back until it is complete. Erasing the gray text means counting how
  many rows were printed, including lines the terminal wrapped. For the
  call, the hard part was cutting off a partly written end (`"}` of `"}}`)
  without also cutting a `,` that belongs inside a string. It also does not
  work in PyCharm's Run console, which is not a terminal unless "Emulate
  terminal" is on.

## Testing strategy

- **Lint and types:** `make lint` (flake8 and mypy with the subject's flags)
  passes on the project.
- **Fake model during development:** the decoder was checked against a fake
  model that uses the real Qwen tokenizer and always picks the tokens of a
  target call. Cases covered: negative and decimal numbers, strings with
  escaped quotes and backslashes (`\\d+`), merged tokens such as `":` and
  `",`, duplicate candidates, thinking that hits the token limit, and
  multi-byte characters split across tokens. These were throwaway scripts;
  there is no test suite in the repository, and they predate sampling and
  the live call display.
- **End-to-end runs** on `data/input` after each change, checking the
  function names, argument names and types in the output file.
- **Benchmark:** the BFCL `simple_python` run described above, with each
  failure read by hand to separate real mistakes from strict scoring.

Edge cases still worth testing before an evaluation: empty strings, very large
numbers, special characters and quotes in prompts, ambiguous prompts that fit
two functions, functions with many or zero parameters, string values that
contain `,` or `}}`, and missing or malformed input files.

## Resources

The only reference I used directly is Qwen's own documentation:

- [Qwen: Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html):
  the tool-call prompt format.
- [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B):
  the model, its chat template, thinking mode and recommended sampling
  settings.

Everything else (sampling methods, BPE, the JSON grammar, ANSI escape codes,
uv and pydantic details) came from web searches, Wikipedia and AI
explanations (see below), not from specific sources I can cite.

### How AI was used

I used Claude Code (Anthropic) throughout the project: nine sessions on 15
and 16 October 2026, and one more on 7 November 2026. The summary below comes
from those session logs. Everything that stayed in the project, I reviewed
and can explain.

**Learning and explanations.** Before writing the decoder I asked how to
approach the project and which vocabulary to look up (logits, greedy
decoding, BPE, logit masking), how generation works with the SDK, what
`encode` returns (a `[1, n]` tensor), whether a KV cache was feasible, why the
CPU was so slow, and what the decoder does and does not enforce (return
types, negative numbers, exponents). In November I asked again what made CPU
inference slow; Claude read the code and the running process and confirmed
the forward pass without KV cache, on the full prompt, as the main cost.

**Environment and tooling.** Claude set up ROCm torch through uv (the AMD
index and the `gfx1200` and `gfx1034` extras), the GPU detection and `.env`
loading in the Makefile, the `accelerate` dependency, typing-only fixes and a
`py.typed` marker in `llm_sdk`, and the flake8 exclude for extra virtual
environments. It also diagnosed a broken ROCm install and PyCharm run
configuration problems, and in November added my GitHub repository as a
second remote and merged its commits with my uncommitted work.

**Code in `src/`.** I wrote the project structure, the pydantic models, the
file loading helpers and a first version of the decoder: the prompt, two
loops for choosing names, and a first, crude version of the generic
`_generate_with(ids, allowed_next)` loop, which I designed. I also wrote the
sampling (`_select`), the sampling fields in `PromptConfig` and the JSON
schema generation. I asked Claude to:

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
- add the live call display (`_show_call`, the `on_token` hook in
  `_generate_with`, `ThoughtPrinter.rewrite`), following the
  `fn_add_numbers(a=1, b` format I specified;
- switch the prompts to Qwen's tool format, output numbers as floats, and fix
  the default output path to the one in the subject.

**Prompt engineering.** Claude analysed the benchmark failures and suggested
prompt changes. I picked which ones to apply (remove "you may guess
parameters", add few-shot examples for percentages, units and locations).
Replacing that prompt with the minimal one was my decision.

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
session logs, then updated it in November for the new decisions.

What I did not delegate: deciding the overall design (generic constrained
loop, config file, thinking on or off, sampling instead of greedy, the
minimal prompt), noticing that Qwen's thoughts always start with "Okay, the
user wants to", proposing the minimal-prompt experiment, running the
benchmarks, and choosing which suggestions to accept.
