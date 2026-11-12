"""Run the decoder on benchmark suites and save every answer.

Usage::

    python bench/run.py --suite bench/data/X.json [--suite ...] \\
        --config config/prompts.json --label NAME \\
        [--src-root DIR] [--out-root bench/results] [--limit N]

Each run gets a NEW folder ``<out-root>/<YYYYmmdd-HHMMSS>-<label>/``; if it
already exists the script stops, so a run can never overwrite another one.
In it:

* ``<suite>.results.json``: per case the id, prompt, expected call, the
  call we got (or null), the error (or null), seconds and forward passes;
* ``<suite>.function_calls.json``: the project's output format
  (``prompt``, ``name``, ``parameters``), for the real moulinette; a case
  that raised is written with name "" and parameters {};
* ``meta.json``: what was run, with which code, config and machine.

Both results files are rewritten after every case, so an interrupted run
keeps what it has done. ``--src-root`` lets you benchmark another checkout
of the project: its directory (and its ``llm_sdk/``) is put first on
``sys.path`` before ``src`` and ``llm_sdk`` are imported.

The model is loaded once. A ``ConstrainedDecoder`` is reused as long as
the cases keep the same function list. Only the latest one is kept: each
holds tables over the whole vocabulary (about 100 MB), so keeping one per
function list would eat the RAM on a suite like xLAM.
"""

import argparse
import functools
import importlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.score import failure_level  # noqa: E402
from bench.suite import REPO_ROOT, Suite, load_suite  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", type=Path, action="append", required=True,
                        help="suite JSON file (repeatable)")
    parser.add_argument("--config", type=Path, required=True,
                        help="prompts config of the code being run")
    parser.add_argument("--label", required=True,
                        help="short name for this run (letters, digits, ._-)")
    parser.add_argument("--src-root", type=Path, default=REPO_ROOT,
                        help="checkout whose src/ to benchmark")
    parser.add_argument("--out-root", type=Path,
                        default=REPO_ROOT / "bench" / "results")
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N cases of each suite")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.label):
        parser.error("--label may only use letters, digits, '.', '_', '-'")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def write_json(path: Path, data: Any) -> None:
    """Replace ``path`` with ``data`` as JSON, never leaving half a file.

    The JSON goes to a temporary file first, then ``os.replace`` swaps it
    in in one step, so a Ctrl-C mid-write keeps the previous version.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def import_project(src_root: Path) -> tuple[ModuleType, ModuleType,
                                            ModuleType]:
    """Import ``src``, ``src.models`` and ``llm_sdk`` from ``src_root``.

    Imports are done here, after changing ``sys.path``, so the modules
    come from the checkout being benchmarked and not from this one.
    """
    sys.path.insert(0, str(src_root / "llm_sdk"))
    sys.path.insert(0, str(src_root))
    src = importlib.import_module("src")
    models = importlib.import_module("src.models")
    llm_sdk = importlib.import_module("llm_sdk")
    return src, models, llm_sdk


class CallCounter:
    """Counts calls to the model's logits methods (one per forward pass).

    The wrapper is stored on the model *instance*, so it hides the class
    method of the same name for this object only.
    """

    METHODS = ("get_logits_cached", "get_logits_from_input_ids")

    def __init__(self, model: Any) -> None:
        """Wrap every logits method ``model`` has; start at zero."""
        self.count = 0
        for name in self.METHODS:
            original = getattr(model, name, None)
            if original is not None:
                setattr(model, name, self._wrap(original))

    def _wrap(self, method: Callable[..., Any]) -> Callable[..., Any]:
        """Return ``method`` with a call counter in front of it."""
        @functools.wraps(method)
        def counted(*args: Any, **kwargs: Any) -> Any:
            """Count one call, then do the real work."""
            self.count += 1
            return method(*args, **kwargs)
        return counted


def git_state(directory: Path) -> tuple[str | None, bool | None]:
    """Return (HEAD commit, has uncommitted changes) of ``directory``.

    Both are None when it is not a git checkout or git fails.
    """
    def git(*command: str) -> str:
        """Run a git command in ``directory`` and return its stdout."""
        return subprocess.run(
            ["git", *command], cwd=directory, capture_output=True,
            text=True, check=True,
        ).stdout.strip()

    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def run_suite(suite: Suite, run_dir: Path, limit: int | None,
              model: Any, config: Any, src: ModuleType,
              models: ModuleType, counter: CallCounter) -> None:
    """Run every case of ``suite``, saving progress after each one."""
    decoder: Any = None
    decoder_key = ""  # function list, as JSON, that ``decoder`` was built for
    results: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    cases = suite.cases[:limit]
    for index, case in enumerate(cases, start=1):
        got: dict[str, Any] | None = None
        error: str | None = None
        counter.count = 0
        start = time.perf_counter()
        try:
            functions = suite.functions_for(case)
            key = json.dumps(functions, sort_keys=True)
            if key != decoder_key:
                decoder = None  # free the old one before building
                decoder_key = ""
                decoder = src.ConstrainedDecoder(
                    model=model,
                    functions=[models.FunctionDefinition.model_validate(f)
                               for f in functions],
                    config=config,
                )
                decoder_key = key
            # Start timing and counting here, so building a decoder (done
            # once per function list) is not charged to this case.
            counter.count = 0
            start = time.perf_counter()
            result = decoder.generate_call(case.prompt)
            got = {"name": result.name, "parameters": result.parameters}
        except Exception as exc:  # keep going: one bad case is a result
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - start
        expected = case.expected.model_dump()
        results.append({
            "id": case.id, "prompt": case.prompt, "expected": expected,
            "got": got, "error": error, "seconds": round(seconds, 3),
            "forward_passes": counter.count,
        })
        calls.append({"prompt": case.prompt,
                      "name": got["name"] if got else "",
                      "parameters": got["parameters"] if got else {}})
        write_json(run_dir / f"{suite.name}.results.json", results)
        write_json(run_dir / f"{suite.name}.function_calls.json", calls)
        level = failure_level(expected, got)
        status = "ERROR" if error else ("ok" if level is None
                                        else f"wrong ({level})")
        print(f"[{suite.name} {index}/{len(cases)}] {case.id}: {status} "
              f"{seconds:.2f}s {counter.count} fwd"
              + (f" | {error}" if error else ""),
              file=sys.stderr, flush=True)


def main() -> int:
    """Run the benchmark; return a process exit code."""
    args = parse_args(sys.argv[1:])
    suites = [load_suite(path) for path in args.suite]
    names = [suite.name for suite in suites]
    if len(set(names)) != len(names):
        print(f"error: duplicate suite names {names}", file=sys.stderr)
        return 1
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            print(f"error: bad suite name {name!r}", file=sys.stderr)
            return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / f"{stamp}-{args.label}"
    args.out_root.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()  # fails if it exists: never reuse a run folder
    except FileExistsError:
        print(f"error: {run_dir} already exists", file=sys.stderr)
        return 1
    print(f"run folder: {run_dir}", file=sys.stderr)

    src_root = args.src_root.resolve()
    commit, dirty = git_state(src_root)
    meta: dict[str, Any] = {
        "label": args.label,
        "suites": [str(path) for path in args.suite],
        "limit": args.limit,
        "config_path": str(args.config.resolve()),
        "config": json.loads(args.config.read_text()),
        "src_root": str(src_root),
        "git_commit": commit,
        "git_dirty": dirty,
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "end_time": None,
        "finished": False,
    }
    write_json(run_dir / "meta.json", meta)

    start = time.perf_counter()
    try:
        src, models, llm_sdk = import_project(src_root)
        meta["src_file"] = src.__file__
        meta["llm_sdk_file"] = llm_sdk.__file__
        config = src.load_prompt_config(args.config)
        model_name = getattr(config, "model_name", DEFAULT_MODEL)
        meta["model_name"] = model_name
        load_start = time.perf_counter()
        model = llm_sdk.Small_LLM_Model(model_name)
        meta["model_load_seconds"] = round(
            time.perf_counter() - load_start, 2)
        import torch  # already loaded by llm_sdk; imported for meta only
        meta["torch_version"] = torch.__version__
        meta["torch_threads"] = torch.get_num_threads()
        write_json(run_dir / "meta.json", meta)

        counter = CallCounter(model)
        for suite in suites:
            run_suite(suite, run_dir, args.limit, model, config,
                      src, models, counter)
        meta["finished"] = True
    finally:
        # Also record the end of an interrupted or crashed run.
        meta["end_time"] = datetime.now().isoformat(timespec="seconds")
        meta["total_seconds"] = round(time.perf_counter() - start, 2)
        write_json(run_dir / "meta.json", meta)
    print(f"done: {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
