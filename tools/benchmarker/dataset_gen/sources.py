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
- reasoning_trace_replay  — implemented for gsm8k; other trace datasets abort
                            with a clear error until their loaders are added.

Three source shapes exist: body sources (synthetic, longbench) produce text of
a target length and the generator drives session structure; conversation
sources (wildchat) return whole user-turn sequences and drive the structure
themselves; trace sources (reasoning_trace_replay) return recorded
(question, answer) pairs whose answer length **overrides** `output_length`.
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
    # THUDM/LongBench is a script-based dataset, unsupported by `datasets` >= 3;
    # read the repo's data.zip (<task>.jsonl files, "context" field) directly.
    import io
    import json
    import zipfile

    try:
        from huggingface_hub import hf_hub_download  # lazy: HF-backed sources only
    except ImportError as exc:
        raise DatasetSourceError(
            "longbench source requires the `huggingface_hub` package "
            "(uv pip install datasets) and pre-staged/cached data (§10.1)"
        ) from exc
    try:
        zip_path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")
    except Exception as exc:
        raise DatasetSourceError(f"longbench: failed to fetch THUDM/LongBench data.zip: {exc}") from exc
    items: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for task in tasks:
            member = next(
                (c for c in (f"data/{task}.jsonl", f"{task}.jsonl") if c in names), None
            )
            if member is None:
                raise DatasetSourceError(
                    f"longbench: task '{task}' not found in data.zip "
                    f"(available: {sorted(n for n in names if n.endswith('.jsonl'))[:10]} …)"
                )
            with zf.open(member) as f:
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    row = json.loads(line)
                    if row.get("context"):
                        items.append(row["context"])
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
    # Whole-shard downloads via hf_hub_download (cached, resumable) beat
    # unauthenticated streaming, which rate-limits to a crawl on this 3.4 GB
    # dataset. Shards are read in order until `cap` matches are collected.
    # NOTE: order is fixed per dataset revision — pin the revision for strict
    # cross-machine reproducibility (§10.8).
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetSourceError(
            "wildchat source requires `huggingface_hub` and `pyarrow` "
            "(uv pip install datasets) and pre-staged/cached data (§10.1)"
        ) from exc
    repo = "allenai/WildChat-1M"
    try:
        shards = sorted(
            f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")
        )
    except Exception as exc:
        raise DatasetSourceError(f"wildchat: failed to list {repo}: {exc}") from exc
    if not shards:
        raise DatasetSourceError(f"wildchat: no parquet shards found in {repo}")

    conversations: list[list[str]] = []
    for shard in shards:
        if len(conversations) >= cap:
            break
        try:
            path = hf_hub_download(repo_id=repo, filename=shard, repo_type="dataset")
        except Exception as exc:
            raise DatasetSourceError(f"wildchat: failed to fetch shard {shard}: {exc}") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["language", "conversation"], batch_size=512):
            for row in batch.to_pylist():
                if not _wildchat_row_matches(row, language_labels, min_turns):
                    continue
                conversations.append(
                    [m["content"] for m in row["conversation"] if m.get("role") == "user"]
                )
                if len(conversations) >= cap:
                    break
            if len(conversations) >= cap:
                break
    return conversations


class TraceSource:
    """Marker base for sources replaying recorded (prompt, output) pairs (§10.5)."""


# dataset name (registry config) -> (HF repo, config, split, question field, answer field)
_REASONING_TRACE_DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "test", "question", "answer"),
}


class ReasoningTraceSource(TraceSource):
    """Recorded reasoning traces; the answer's token count overrides output_length."""

    def __init__(self, source: Source, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        name = source.config.get("dataset")
        if not name:
            raise DatasetSourceError(
                "reasoning_trace_replay needs config.dataset (e.g. gsm8k)"
            )
        self._pairs = _load_reasoning_traces(name)
        if not self._pairs:
            raise DatasetSourceError(f"reasoning_trace_replay: dataset '{name}' yielded no pairs")

    def trace(self, rng: random.Random) -> tuple[str, str]:
        return self._pairs[rng.randrange(len(self._pairs))]


def _load_reasoning_traces(name: str) -> list[tuple[str, str]]:
    spec = _REASONING_TRACE_DATASETS.get(name)
    if spec is None:
        raise DatasetSourceError(
            f"reasoning_trace_replay dataset '{name}' not supported "
            f"(supported: {sorted(_REASONING_TRACE_DATASETS)}); extending the "
            "dataset table in sources.py is a small change"
        )
    repo, config, split, q_field, a_field = spec
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetSourceError(
            "reasoning_trace_replay requires the `datasets` package "
            "(uv pip install datasets) and pre-staged/cached data (§10.1)"
        ) from exc
    try:
        ds = load_dataset(repo, config, split=split)
    except Exception as exc:
        raise DatasetSourceError(f"reasoning_trace_replay: failed to load '{name}': {exc}") from exc
    return [(row[q_field], row[a_field]) for row in ds if row.get(q_field) and row.get(a_field)]


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
    if source.kind == "reasoning_trace_replay":
        return ReasoningTraceSource(source, tokenizer)
    raise DatasetSourceError(  # unreachable for registry-validated kinds; defensive
        f"source kind '{source.kind}' has no implementation; aborting per §10.1"
    )
