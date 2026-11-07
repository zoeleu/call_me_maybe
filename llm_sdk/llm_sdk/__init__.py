# ABOUTME: LLM SDK for local model inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model class for loading and running causal
# language models.

# transformers, torch and huggingface_hub are only partially typed.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import os
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    DynamicCache, PreTrainedTokenizer, PreTrainedModel, logging
from huggingface_hub import hf_hub_download


logging.set_verbosity_error()  # keep the console clean


def _shared_prefix(first: list[int], second: list[int]) -> int:
    """Return how many leading ids ``first`` and ``second`` have in common."""
    shared = 0
    limit = min(len(first), len(second))
    while shared < limit and first[shared] == second[shared]:
        shared += 1
    return shared


class Small_LLM_Model:
    """Utility class wrapping a lightweight Hugging Face causal-LM for fast,
    low-memory experimentation.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the HF Hub.
    device: str | None, default=None
        Computation device. If *None* we automatically select ``mps`` when
        available on macOS, ``cuda`` when available, otherwise we fall back
        to ``cpu``.
    dtype: torch.dtype | None, default=None
        Numerical precision. When using a GPU or MPS we default to ``float16``
        to keep memory usage reasonable; on CPU we keep ``float32`` for maximum
        compatibility.
    num_threads: int | None, default=8
        CPU threads used by torch, capped at the number of CPUs. Decoding
        one token is limited by memory bandwidth, not compute: past about
        one thread per physical core, extra threads (hyper-threads,
        efficiency cores) only slow it down. *None* keeps torch's default,
        one thread per logical CPU.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
        num_threads: int | None = 8,
    ) -> None:
        self._model_name = model_name

        if num_threads is not None:
            # Process-wide setting: it affects every model in the process.
            torch.set_num_threads(
                max(1, min(num_threads, os.cpu_count() or 1))
            )

        # Auto-select device with priority: mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in ["cuda", "mps"] else \
                torch.float32
        self._dtype = dtype

        # --- load tokenizer & model ------------------------------------------
        self._tokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            ),
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self._dtype,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=trust_remote_code,
            ),
        )
        self._model.to(self._device)  # pyright: ignore[reportArgumentType]
        self._model.eval()

        # switch to inference-only mode
        for p in self._model.parameters():
            p.requires_grad = False

        # --- KV-cache state for get_logits_cached ----------------------------
        self._cache = DynamicCache(config=self._model.config)
        self._cached_ids: list[int] = []
        self._last_logits: list[float] = []

    def encode(self, text: str) -> torch.Tensor:
        """Tokenise *text* and return a 2-D ``input_ids`` tensor.

        The tensor is placed on the target device.
        """
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        token_ids: list[int]
        if isinstance(ids, torch.Tensor):
            token_ids = cast(list[int], ids.tolist())
        else:
            token_ids = ids
        return cast(
            str, self._tokenizer.decode(token_ids, skip_special_tokens=True)
        )

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return the raw logits (no softmax) for the next token.

        ``input_ids`` is the list of input token ids.
        """
        input_tensor = torch.tensor(
            [input_ids], device=self._device, dtype=torch.long
        )
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        # Logits for the last token of the sequence (batch size 1)
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]

    def get_logits_cached(self, input_ids: list[int]) -> list[float]:
        """Same result as :py:meth:`get_logits_from_input_ids`, reusing work.

        The keys and values computed for the previous ``input_ids`` are kept
        (a KV-cache). Only the tokens after the prefix shared with the
        previous call are run through the model, so appending one token
        costs one token, not the whole sequence. If the new ids diverge
        from the cached ones, the cache is cut back to the shared prefix.
        """
        if not input_ids:
            raise ValueError("input_ids must not be empty")
        if input_ids == self._cached_ids and self._last_logits:
            # Nothing new to feed: the model needs at least one token.
            return list(self._last_logits)
        # The last id is fed again when it is already cached, to get the
        # logits that follow it.
        shared = min(_shared_prefix(input_ids, self._cached_ids),
                     len(input_ids) - 1)
        if shared < len(self._cached_ids):
            # A negative value removes that many tokens from the end;
            # crop(0) would be a no-op, so never pass the target length.
            self._cache.crop(shared - len(self._cached_ids))
        new_tensor = torch.tensor(
            [input_ids[shared:]], device=self._device, dtype=torch.long
        )
        try:
            with torch.no_grad():
                out = self._model(
                    input_ids=new_tensor,
                    past_key_values=self._cache,
                    use_cache=True,
                    # Skip the vocabulary projection for the other
                    # positions: it is a quarter of the model's weights.
                    logits_to_keep=1,
                )
        except BaseException:
            # The cache may be half-updated: start over on the next call.
            self._cache = DynamicCache(config=self._model.config)
            self._cached_ids = []
            self._last_logits = []
            raise
        self._cached_ids = list(input_ids)
        self._last_logits = cast(list[float], out.logits[0, -1].tolist())
        return list(self._last_logits)

    def get_path_to_vocab_file(self) -> str:
        """Download (or find in the cache) the vocabulary file."""
        vocab_file_name = self._tokenizer.vocab_files_names.get(
            'vocab_file', "vocab.json"
        )
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name
        )
        return vocab_path

    def get_path_to_merges_file(self) -> str:
        """Download (or find in the cache) the BPE merges file."""
        merges_file_name = self._tokenizer.vocab_files_names.get(
            'merges_file', "merges.txt"
        )
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name
        )
        return merges_path

    def get_path_to_tokenizer_file(self) -> str:
        """Download (or find in the cache) the tokenizer.json file."""
        tokenizer_file_name = self._tokenizer.vocab_files_names.get(
            'tokenizer_file', "tokenizer.json"
        )
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name
        )
        return tokenizer_path
