*This project has been created as part of the 42 curriculum by zleullie.*

# call me maybe

Turn natural-language requests into structured function calls with a small
language model (Qwen/Qwen3-0.6B by default; Qwen3.5-0.8B and Gemma 3 270M
also work), using constrained decoding so that every output is valid,
schema-compliant JSON.

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
2. Build a chat prompt for the chosen model that lists the functions in
   Qwen's tool-calling format.
3. Generate the call under constraints: function name, then each parameter
   name and value. Among the allowed tokens, the one the model scores
   highest is taken (greedy decoding).
4. Show the call live on the terminal as it is written, then write all
   results to `data/output/function_calls.json`.

## Instructions

### Requirements

- Linux, Python 3.10 to 3.14, and [uv](https://docs.astral.sh/uv/).
- No GPU is needed or used: the project installs the CPU-only build of
  PyTorch.
- The model is downloaded from Hugging Face on the first run. The default
  model (Qwen3-0.6B) needs no account. Gemma is gated: accept its licence on
  Hugging Face and put `HF_TOKEN=...` in a `.env` file; `make run` loads it.

### Install

```bash
make install
```

This runs `uv sync`. torch comes from PyTorch's CPU index
(`download.pytorch.org/whl/cpu`, about 200 MB) instead of PyPI, whose Linux
wheel also pulls several GB of NVIDIA libraries that a CPU run never uses.

On 42 computers, if `~/goinfre` exists, the Makefile puts everything large
there: the virtual environment, uv's Python installs, the Hugging Face cache
(the models) and the mypy cache, plus the uv cache unless `UV_CACHE_DIR` is
already set. The home directory quota is too small for torch and the models,
and goinfre is fast local storage. It is ephemeral, so after a reset
`make install` simply downloads everything again. Outside 42 nothing changes,
and any of these variables can still be overridden from the environment.

### Run

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>] [--prompts_config <file>]
```

| Option | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calls.json` |
| `--prompts_config` | `config/prompts.json` (model name, prompt texts, limits) |

Outside 42, `uv run` works as shown. On a 42 computer, use `make run` so the
goinfre paths above are set.

### Makefile

| Rule | What it does |
|---|---|
| `make install` | `uv sync` (CPU-only torch) |
| `make run` | `python -m src` with the default paths |
| `make debug` | Runs the program under `pdb` |
| `make test` | Runs the tests in `tests/` with pytest |
| `make lint` | `flake8 .` and `mypy` with the flags required by the subject |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make clean` | Removes caches and `data/output/` |

### Prompt configuration

`config/prompts.json` holds everything about the prompt and decoding that is
data rather than code: the model name, the system prompt, the chat template,
the prefixes the program writes itself and the token limit. It is validated
by the `PromptConfig` pydantic model.

There is one such file per supported model. They differ by `model_name` and
by `chat_template`, which holds each model's own turn markers:

| File | Model |
|---|---|
| `config/prompts.json` (default) | `Qwen/Qwen3-0.6B` |
| `config/qwen3.5-0.8b.json` | `Qwen/Qwen3.5-0.8B` |
| `config/gemma-3-270m-it.json` | `google/gemma-3-270m-it` (gated) |

```bash
make run ARGS="--prompts_config config/qwen3.5-0.8b.json"
```

`config/prompts.schema.json` is the JSON schema of that model, and
`prompts.json` points to it with `"$schema"`, so an editor such as VS Code can
autocomplete the keys and flag invalid values (for example a negative
`max_value_tokens`). After changing `PromptConfig`, regenerate it with:

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

## Algorithm explanation

### What the model sees

The prompt follows the model's chat template (Qwen3's is shown here) and
Qwen's Hermes-style tool format. Each function from
`functions_definition.json` is converted to a JSON schema and listed inside
`<tools></tools>`. The
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
is disabled; it is part of the chat template in the config. Gemma has no
system role and no thinking block: its template puts the system text at the
top of the user turn. Everything up to `{"name": "` is fixed text written by
the program, not by the model.

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

Two kinds of `allowed_next` are built on top of it, one for fixed strings
and one for free values. The loop also takes an optional `on_token` callback,
which the live display uses.

### Choosing a token among the allowed ones

`_best_allowed(ids, allowed)` asks the model for its logits (one score per
token of the vocabulary) and returns the allowed token with the highest
score. Tokens outside `allowed` are never looked at, so they can never be
chosen. This is greedy decoding: there is no randomness, and the same prompt
always gives the same call.

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
| integer | the same as number, but `.` is never allowed |
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
`json.loads` for strings, `float` for numbers, `int` for integers. Numbers
are always output as floats (`2` → `2.0`), as in the subject's examples;
integers stay integers (`7`), since a function declared with `n: int` may
check `isinstance(n, int)` and reject `7.0`.

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

The vocabulary (read from `tokenizer.json`, which every supported model
ships; Gemma has no `vocab.json`) stores tokens in an internal form (`Ġ` or
`▁` for a space, `Ċ` for a newline), so its keys are not the real text. At
startup, each token id is decoded once with the SDK's `decode` to get its
actual text, and tokens that decode to an empty string are dropped (special
tokens; they could loop forever without progress). This takes about 0.5 to
1 s.

### The KV cache in `llm_sdk`

`get_logits_from_input_ids` runs the model on the whole sequence at every
step. `get_logits_cached` (added to the SDK) keeps the keys and values the
model computed for the previous call, and only runs the tokens added since.
Writing one more token then costs one token instead of the whole prompt.

The cache can only move forward. When the next prompt arrives, the ids no
longer continue the previous ones. Plain attention layers could simply drop
the last tokens, but Gemma's sliding-window layers and Qwen3.5's recurrent
layers cannot. So the SDK never removes tokens: the first time a new prompt
arrives, it rebuilds the cache up to what both prompts share (the system
prompt and the tool list) and keeps a copy of that state. Every later prompt
restarts from the copy. The decoder falls back to the uncached method when
the SDK does not have the cached one.

## Design decisions

- **The program writes the JSON structure, the model fills the gaps.** Rather
  than a full JSON grammar, the fixed parts (`{"name": "`, `, "arguments":
  {`) are appended directly and only the variable parts are constrained.
  This is simpler to check and saves model calls.
- **Qwen's native tool format.** Listing tools the way Qwen3 was trained to
  read them (rendered from the model's own chat template) keeps the prompt
  "in distribution" for the model.
- **Greedy decoding, no sampling.** An earlier version drew tokens with the
  sampling settings Qwen recommends (temperature 0.6, top-p 0.95, top-k 20),
  because Qwen's documentation discourages greedy decoding for free text. In
  this project most of the text is fixed or masked, the answers are short,
  and a wrong draw cannot be corrected later. Greedy is simpler, needs no
  seed, and gives the same output on every run. The benchmark below compares
  both.
- **Forced tokens skip the model.** When only one token is allowed, no
  forward pass is run. Once a function name is unambiguous, the rest of it
  costs nothing.
- **A long system prompt with examples.** The prompt has three few-shot
  examples (percentages, units, "city, STATE") that target real value
  errors seen on a benchmark. Without a KV cache every prompt token was
  processed again at every step, which made this prompt too slow on the CPU.
  With the cache it is processed once and reused for every prompt.
- **No thinking.** An earlier version let the model reason freely before
  answering. It cost hundreds of forward passes per prompt and did not pay
  for itself (see the benchmark), so the code was removed. The prompt keeps
  the empty thinking block Qwen3 expects.
- **One answer format for every model.** All three models are made to write
  the same JSON call. This keeps one grammar, but it is not what Qwen3.5 was
  trained to write (its template uses `<function=...>` tags), and Gemma 3
  270M was not trained on any tool format.
- **Prompts live in `config/prompts.json`, with a JSON schema.** System
  prompt, chat template, prefixes and limits are data,
  validated by the `PromptConfig` pydantic model, so they can be tuned
  without touching the decoder. The schema is generated from the same model,
  so the editor and the program agree on what is valid.
- **pydantic everywhere.** Function definitions, prompts, results and the
  prompt config are all pydantic models, and files are read and written
  through `TypeAdapter`.
- **stderr for progress, file for results.** The terminal display (the call
  being written in gray, then `name(args)` lines) never mixes with
  the output file. The live display is only enabled when stderr is a
  terminal, because erasing needs cursor movement escape codes.
- **CPU only, performance cores only.** torch is the CPU build. On a hybrid
  Intel CPU the SDK restricts the process to one logical CPU per
  performance core and uses at most that many threads: a thread that lands
  on an efficiency core makes the others wait (see
  [Speed](#speed)).
- **goinfre on 42 computers.** torch and the models do not fit in the home
  quota, so the Makefile moves the venv, Python, the caches and the Hugging
  Face cache to `~/goinfre` when it exists.

## Performance analysis

### Reliability

The output is valid JSON by construction: the JSON structure is written by
the program, names are chosen from fixed lists, and values can only be
completed along the state machine before being serialised by pydantic.
Every parameter of the chosen function is present exactly once, with the
declared type.

### Accuracy

Measured on 12 November 2026 with the tooling in `bench/` (see
[Benchmark](#benchmark)), on three data sets:

- **public**: the 11 sample prompts of `data/input`, graded by the
  moulinette (it runs the function with the arguments and compares the
  result);
- **private**: the moulinette's 11 private prompts, graded the same way;
- **xLAM**: 37 single-call cases picked from
  [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k),
  each with its own list of 1 to 6 tools. "Strict" means the right function
  and exactly the expected arguments.

| Setup | Model | public | private | xLAM function | xLAM strict |
|---|---|---|---|---|---|
| thinking + sampling (before) | Qwen3-0.6B | 9/11 | 10/11 | 37/37 | 30/37 |
| **greedy, no thinking (default)** | Qwen3-0.6B | 10/11 | 9/11 | 37/37 | 30/37 |
| greedy, no thinking | Qwen3.5-0.8B | 10/11 | 11/11 | 37/37 | 31/37 |
| greedy, no thinking | Gemma 3 270M | 8/11 | 6/11 | 29/37 | 21/37 |

What the failures show (each one was read):

- **Removing thinking and sampling cost nothing measurable.** Both Qwen3
  setups score 49 of 59 over the three sets. The sampled run is a single
  draw with seed 42, so a difference of one case either way is noise. With
  sampling the model wrote `Shrek` for *"Greet shrek"*; greedy copies
  `shrek`.
- **The function is never wrong with the Qwen models.** All their misses
  are argument values.
- **Regexes are the weak spot on the sample prompts**: `34` instead of
  `\d+` for *"replace all numbers"* (Qwen3-0.6B only; Qwen3.5 writes
  `\d+`). `cat` instead of `\bcat\b` and `[aeiou]` without capitals fail
  a strict comparison but give the right result, so the moulinette accepts
  them.
- **Most xLAM misses come from the missing parameter descriptions.** The
  project's function format only has a name and a type per parameter, so
  the model writes `"French"` where the dataset expects `"fr"`, or
  `"2023-01-01"` for `"2023-01"`. One case expects a player id (60) that is
  not in the question at all. These say more about the format than about
  the model.
- **Gemma 3 270M is too small for this prompt**: it picks the wrong function
  in 8 of 37 xLAM cases and 5 of 22 subject prompts, and sometimes fills
  arguments with numbers that are not in the question. Its output is still
  always valid, which is what the decoder guarantees.
- Earlier reference, different data: in October, greedy decoding on 302
  questions of the Berkeley Function Calling Leaderboard (`simple_python`)
  gave about 79 % strict and 91 % after reading the failures by hand. That
  tooling was not kept.

Possible bias to keep in mind: xLAM's expected answers were themselves
generated and checked by language models, and the 37 cases were selected by
rules that favour values copied from the question, so these numbers are
optimistic compared with real requests.

### Speed

Intel i7-12700 (8 performance cores, 4 efficiency cores), CPU only, model
loading excluded (it takes about 2 s once the files are in the disk cache).

| Setup | Model | 11 sample prompts | per xLAM case | forward passes per call |
|---|---|---|---|---|
| thinking + sampling (before) | Qwen3-0.6B | 198 s | 25.0 s | about 115 to 160 |
| **greedy, no thinking (default)** | Qwen3-0.6B | 27 s | 3.3 s | about 10 to 14 |
| greedy, no thinking | Qwen3.5-0.8B | 33 s | 3.5 s | about 10 to 12 |
| greedy, no thinking | Gemma 3 270M | 7 s | 1.5 s | about 8 to 17 |

- **Where the time goes:** almost all of it is the model's forward pass,
  about 0.2 s per call on Qwen3-0.6B (profiled: 26 of 29 s inside torch's
  matrix products). The decoder's own Python work is small.
- **Thinking was the main cost**: about ten times more forward passes per
  call, for the same accuracy. The subject's limit is 5 minutes; the
  thinking setup used 3 min 18 s of it on the samples, the default 27 s.
- **The KV cache** makes each step cost one token instead of the whole
  prompt (about 720 tokens with the tool list).
- **Threads and cores** (11 sample prompts, Qwen3-0.6B): 49 s on 1
  performance core, 36 s on 2, 30 s on 4, 27 s on 8. The same 4 threads on
  the efficiency cores take 45 s, and 20 threads over all cores take 70 s.
  More cores help little because one step mostly reads the model's 2.4 GB
  of weights from memory. The SDK therefore uses the performance cores
  only, one thread each.
- What keeps it fast: no thinking, the KV cache, forced tokens without a
  model call, and the cached token tables.

### Benchmark

The tooling is in `bench/` and is not needed to run the project.

```bash
python bench/make_suites.py      # public and private suites from the moulinette
python bench/xlam_select.py      # xLAM cases (needs HF_TOKEN, gated dataset)
python bench/run.py --suite bench/data/xlam.json --config config/prompts.json --label my-run
python bench/score.py bench/results/<run> [<other run>] [--failures]
```

Each run writes to a new folder in `bench/results/` and never touches an
older one. `bench/data/` and `bench/results/` are not committed: the xLAM
rows are gated, only their ids are in `bench/xlam_ids.json`. Run one model
at a time: each takes 3 to 4 GB of RAM.

### Known limitations

- Only the `string`, `number` and `integer` parameter types are supported.
  Any other type (for example `boolean` or an array) raises an error.
- Numbers cannot use exponent notation (`1e-9`), so the model has to write
  them out in full. Leading zeros (`007`) are accepted and parsed as `7`.
- `\uXXXX` escapes are not supported inside strings.
- Every parameter of the chosen function is always written: there is no way
  to leave out an optional one.
- The function format has no parameter descriptions, so the model only sees
  each parameter's name and type.
- Missing or invalid input files are not yet caught in `main`, so they end
  with a Python traceback instead of a clear error message.
- Core pinning only knows Linux on hybrid Intel CPUs; elsewhere the SDK just
  uses 8 threads.
- `uv sync` has not been tested on a fresh clone outside 42.

## Challenges faced

- **GPU builds of torch.** The project first ran on AMD GPUs through ROCm
  builds of torch, with one uv extra per GPU. It worked on the GPU, but the
  ROCm build's CPU fallback was about 30 times slower than the CPU build,
  and 42 computers have no usable GPU. The whole setup was replaced by the
  CPU-only build.
- **A KV cache that cannot go backwards.** The first cache removed the last
  tokens (`crop`) when a new prompt arrived. That works on Qwen3 but fails
  on Gemma (sliding-window attention) and Qwen3.5 (recurrent layers). The
  cache now restarts from a saved copy instead.
- **More threads, slower run.** With one thread per logical CPU (20), the
  sample run took 70 s instead of 27 s: the 4 efficiency cores hold the
  other threads back. The SDK now pins itself to the performance cores.
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
  came from re-running the model on the full prompt at every step. The KV
  cache and skipping forced tokens were the useful levers.
- **Values that are well-formed but wrong.** Percentages, units and dates
  were the main errors on the benchmark. Constraints can't fix these;
  few-shot examples in the prompt targeted them.
- **Showing the call live.** Erasing the gray text means counting how many
  rows were printed, including lines the terminal wrapped. The hard part was
  cutting off a partly written end (`"}` of `"}}`) without also cutting a
  `,` that belongs inside a string. It also does not work in PyCharm's Run
  console, which is not a terminal unless "Emulate terminal" is on.

## Testing strategy

- **Lint and types:** `make lint` (flake8 and mypy with the subject's flags)
  passes on the project.
- **Unit tests (`make test`):**
  - `tests/test_decoder_grammar.py`: the character state machine, without
    any model (integers, fractions, signs, ends).
  - `tests/test_decoder_calls.py`: whole calls with a fake model that uses
    the real Qwen tokenizer and always prefers the tokens of a target call.
    It checks that integers stay `int`, numbers become `float`, and strings
    with quotes, backslashes, commas and `}}` come back unchanged.
  - `tests/test_llm_sdk_cache.py`: the KV cache gives the same logits as the
    uncached method on all three models, for every way the ids can change
    between calls (a model that cannot be downloaded is skipped).
  - `tests/test_bench_*.py`: the benchmark's scoring and selection rules.
- **End-to-end runs** on `data/input` after each change, checking the
  function names, argument names and types in the output file.
- **Benchmark:** the runs described in [Accuracy](#accuracy), graded by the
  moulinette for the subject's prompts, with each failure read by hand.

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
and 16 October 2026, one on 7 November 2026 and one on 12 November 2026. The summary below comes
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

**Session of 12 November 2026.** I gave Claude a list of changes and it
planned and carried them out, with one subagent for the benchmark tooling.
What I decided: remove thinking, go back to greedy decoding, drop the GPU
setup for CPU-only torch, add Qwen3.5-0.8B and Gemma 3 270M, pin the
performance cores, and which data to benchmark on. What Claude wrote:

- the removal of the thinking and sampling code (including my `_select`),
  and the three-line greedy `_best_allowed`;
- `model_name` in `PromptConfig`, the per-model config files, and reading
  the vocabulary from `tokenizer.json`;
- the rewrite of the SDK's KV cache to restart from a saved copy
  (`_rewind`), after its first version failed on Gemma and Qwen3.5, and the
  performance-core pinning (`_performance_cpus`);
- the CPU-only `pyproject.toml` files and Makefile changes;
- `tests/test_decoder_calls.py` and the extended cache test;
- everything in `bench/` (written by a subagent, reviewed by Claude): the
  runner, the scorer and the xLAM case selection;
- the benchmark runs, the reading of the failures and this README update.

These are parts I still have to review line by line and be able to explain
before an evaluation.

What I did not delegate: deciding the overall design (generic constrained
loop, config file, thinking on or off, sampling instead of greedy, the
minimal prompt), noticing that Qwen's thoughts always start with "Okay, the
user wants to", proposing the minimal-prompt experiment, running the
benchmarks, and choosing which suggestions to accept.
