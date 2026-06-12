"""Prompt-pool loading (the dataset generator's prompts.jsonl artifact)."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Turn:
    scenario: str
    session_idx: int
    turn_idx: int
    prompt_text: str
    text_tokens: int
    max_tokens: int
    think_time_ms: float | None


@dataclass(frozen=True)
class PoolSession:
    session_idx: int
    scenario: str
    mode: str  # open_loop | sequential
    turns: tuple[Turn, ...]


def load_pool(path: Path) -> dict[str, deque[PoolSession]]:
    """Pool grouped per class, sessions in pool order (consumed across sweep steps)."""
    by_session: dict[int, list[dict]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_session.setdefault(r["session_idx"], []).append(r)
    pool: dict[str, deque[PoolSession]] = {}
    for session_idx in sorted(by_session):
        rows = sorted(by_session[session_idx], key=lambda r: r["turn_idx"])
        turns = tuple(
            Turn(
                scenario=r["scenario"],
                session_idx=r["session_idx"],
                turn_idx=r["turn_idx"],
                prompt_text=r["prompt_text"],
                text_tokens=r["text_tokens"],
                max_tokens=r["max_tokens"],
                think_time_ms=r.get("think_time_ms"),
            )
            for r in rows
        )
        sess = PoolSession(
            session_idx=session_idx,
            scenario=rows[0]["scenario"],
            mode=rows[0]["session_mode"],
            turns=turns,
        )
        pool.setdefault(sess.scenario, deque()).append(sess)
    return pool
