"""Dataset sources (SPECIFICATIONS.md §10.5).

Source-failure semantics per §10.1: any failure aborts the run with a clear
error — there is no silent fallback to synthetic data.

v1 status (IMPLEMENTATION_PLAN.md M1 source order):
- synthetic               — implemented (no network).
- longbench               — implemented via HuggingFace `datasets` (pre-staged
                            or cached locally; lazily imported).
- wildchat                — implemented (conversation-driven: real turn
                            boundaries shape the session per §10.5; per-turn
                            lengths clamped to the scenario's bounds).
- reasoning_trace_replay  — NOT YET IMPLEMENTED: aborts with a clear error.

Two source shapes exist: body sources (synthetic, longbench) produce text of a
target length and the generator drives session structure; conversation sources
(wildchat) return whole user-turn sequences and drive the structure themselves.
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


class ConversationSource:
    """Marker base for sources whose corpus drives the session structure (§10.5)."""


# ISO codes (registry config) -> WildChat language labels; raw labels also accepted.
_WILDCHAT_LANGUAGES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "nl": "Dutch",
}


def _wildchat_row_matches(row: dict, language_labels: set[str], min_turns: int) -> bool:
    if row.get("language") not in language_labels:
        return False
    user_turns = [m for m in row.get("conversation", []) if m.get("role") == "user"]
    return len(user_turns) >= min_turns


class WildChatSource(ConversationSource):
    """Real user<->assistant conversations from allenai/WildChat-1M (§10.5).

    Conversation turn boundaries drive the session structure; the generator
    clamps per-turn lengths to the scenario's declared distribution bounds.
    """

    def __init__(self, source: Source, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        cfg = source.config
        languages = cfg.get("languages") or ["en"]
        labels = {_WILDCHAT_LANGUAGES.get(code, code) for code in languages}
        min_turns = int(cfg.get("min_turns", 1))
        cap = int(cfg.get("max_conversations", 20_000))
        self._conversations = _load_wildchat_conversations(labels, min_turns, cap)
        if not self._conversations:
            raise DatasetSourceError(
                f"wildchat: no conversations matched languages={sorted(labels)}, "
                f"min_turns={min_turns}"
            )

    def conversation(self, rng: random.Random) -> list[str]:
        return self._conversations[rng.randrange(len(self._conversations))]


def _load_wildchat_conversations(
    language_labels: set[str], min_turns: int, cap: int
) -> list[list[str]]:
    try:
        from datasets import load_dataset  # lazy: only needed for HF-backed sources
    except ImportError as exc:
        raise DatasetSourceError(
            "wildchat source requires the `datasets` package "
            "(uv pip install datasets) and pre-staged/cached data (§10.1)"
        ) from exc
    try:
        # Streaming stops shard downloads once `cap` matches are collected.
        # NOTE: iteration order is fixed per dataset revision — pin the revision
        # for strict cross-machine reproducibility (§10.8).
        rows = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception as exc:
        raise DatasetSourceError(f"wildchat: failed to load allenai/WildChat-1M: {exc}") from exc
    conversations: list[list[str]] = []
    for row in rows:
        if not _wildchat_row_matches(row, language_labels, min_turns):
            continue
        conversations.append(
            [m["content"] for m in row["conversation"] if m.get("role") == "user"]
        )
        if len(conversations) >= cap:
            break
    return conversations


def trim_to_tokens(text: str, max_tokens: int, tokenizer: Tokenizer) -> str:
    """Clamp real text to a token budget by dropping trailing words (§10.5)."""
    if tokenizer.count(text) <= max_tokens:
        return text
    words = text.split()
    trimmed = " ".join(words)
    while words and tokenizer.count(trimmed) > max_tokens:
        drop = max(1, (tokenizer.count(trimmed) - max_tokens) // 2)
        del words[-drop:]
        trimmed = " ".join(words)
    return trimmed


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
    if source.kind == "wildchat":
        return WildChatSource(source, tokenizer)
    raise DatasetSourceError(
        f"source kind '{source.kind}' is not implemented yet "
        "(IMPLEMENTATION_PLAN.md M1 source order: synthetic → longbench → "
        "wildchat → reasoning_trace_replay); aborting per §10.1 — no silent fallback"
    )
