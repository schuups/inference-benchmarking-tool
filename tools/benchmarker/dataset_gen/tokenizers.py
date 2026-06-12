"""Tokenizer abstraction (SPECIFICATIONS.md §10.6).

Production uses the target model's HuggingFace tokenizer; tests and offline
smoke runs use the deterministic WordTokenizer (1 whitespace word == 1 token),
keeping the laptop test suite hermetic.
"""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    tokenizer_id: str

    def count(self, text: str) -> int: ...


class WordTokenizer:
    tokenizer_id = "word"

    def count(self, text: str) -> int:
        return len(text.split())


class HFTokenizer:
    def __init__(self, tokenizer_id: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for HF tokenizers — "
                "uv pip install transformers (or use --tokenizer word for smoke runs)"
            ) from exc
        self.tokenizer_id = tokenizer_id
        self._tok = AutoTokenizer.from_pretrained(tokenizer_id)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


def load_tokenizer(tokenizer_id: str) -> Tokenizer:
    if tokenizer_id == "word":
        return WordTokenizer()
    return HFTokenizer(tokenizer_id)
