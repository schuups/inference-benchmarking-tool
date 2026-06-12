"""Dataset sources (SPECIFICATIONS.md §10.5).

Source-failure semantics per §10.1: any failure aborts the run with a clear
error — there is no silent fallback to synthetic data.

v1 status (IMPLEMENTATION_PLAN.md M1 source order):
- synthetic               — implemented (no network).
- longbench               — implemented via HuggingFace `datasets` (pre-staged
                            or cached locally; lazily imported).
- wildchat                — NOT YET IMPLEMENTED: aborts with a clear error.
- reasoning_trace_replay  — NOT YET IMPLEMENTED: aborts with a clear error.
"""

from __future__ import annotations

import random

from .registry import Source
from .tokenizers import Tokenizer


class DatasetSourceError(RuntimeError):
    """§10.1: source failures abort the run; never silently fall back."""


_FILLER_VOCABULARY = (
    "alpine summit ridge glacier valley meadow stream boulder forest trail "
    "granite moraine cirque couloir saddle traverse cornice scree tarn col "
    "pass crest ledge gully spur buttress arete chimney slab crag bivouac "
    "ascent descent cache marker beacon relay node packet tensor kernel "
    "buffer stride batch layer logits prefill decode router expert shard"
).split()


class SyntheticSource:
    """Filler text from a fixed vocabulary; deterministic given the rng."""

    def __init__(self, source: Source, tokenizer: Tokenizer):
        self._tokenizer = tokenizer

    def body(self, target_tokens: int, rng: random.Random) -> str:
        return _fill_to_target("", target_tokens, self._tokenizer, lambda: rng.choice(_FILLER_VOCABULARY))


class LongBenchSource:
    """Real code/text drawn from THUDM/LongBench tasks (§10.5)."""

    def __init__(self, source: Source, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        tasks = source.config.get("tasks")
        if not tasks:
            raise DatasetSourceError("longbench source needs config.tasks (e.g. [lcc, repobench-p])")
        self._items = _load_longbench_items(tasks)
        if not self._items:
            raise DatasetSourceError(f"longbench: no items loaded for tasks {tasks}")

    def body(self, target_tokens: int, rng: random.Random) -> str:
        pieces: list[str] = []
        tokens = 0
        while tokens < target_tokens:
            item = self._items[rng.randrange(len(self._items))]
            pieces.append(item)
            tokens += self._tokenizer.count(item)
        words = " ".join(pieces).split()
        text = " ".join(words)
        while words and self._tokenizer.count(text) > target_tokens:
            drop = max(1, (self._tokenizer.count(text) - target_tokens) // 2)
            del words[-drop:]
            text = " ".join(words)
        return text


def _load_longbench_items(tasks: list[str]) -> list[str]:
    try:
        from datasets import load_dataset  # lazy: only needed for HF-backed sources
    except ImportError as exc:
        raise DatasetSourceError(
            "longbench source requires the `datasets` package "
            "(uv pip install datasets) and pre-staged/cached data (§10.1)"
        ) from exc
    items: list[str] = []
    for task in tasks:
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test")
        except Exception as exc:
            raise DatasetSourceError(f"longbench: failed to load task '{task}': {exc}") from exc
        items.extend(row["context"] for row in ds if row.get("context"))
    return items


def _fill_to_target(prefix: str, target_tokens: int, tokenizer: Tokenizer, next_word) -> str:
    words: list[str] = prefix.split()
    text = prefix
    while tokenizer.count(text) < target_tokens:
        words.extend(next_word() for _ in range(8))
        text = " ".join(words)
    while words and tokenizer.count(text) > target_tokens:
        words.pop()
        text = " ".join(words)
    return text


def make_source(source: Source, tokenizer: Tokenizer):
    if source.kind == "synthetic":
        return SyntheticSource(source, tokenizer)
    if source.kind == "longbench":
        return LongBenchSource(source, tokenizer)
    raise DatasetSourceError(
        f"source kind '{source.kind}' is not implemented yet "
        "(IMPLEMENTATION_PLAN.md M1 source order: synthetic → longbench → "
        "wildchat → reasoning_trace_replay); aborting per §10.1 — no silent fallback"
    )
