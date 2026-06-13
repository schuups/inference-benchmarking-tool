"""Eval suites for the builtin grader (§12.5).

The builtin backend covers small, network-free suites — a cheap sanity gate
vehicle and the deterministic target for the M11 tests (graded against the mock
server's canned answers). Standard published suites (gsm8k, gpqa_diamond, …) are
run by `LmEvalBackend` via lm-eval-harness, which owns their datasets, prompts,
and metrics.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


def numeric_exact_match(response: str, gold: str) -> float:
    """1.0 if the last number in the response equals `gold` (gsm8k-style), else 0.0."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
    if not nums:
        return 0.0
    pred = nums[-1]
    try:
        return 1.0 if abs(float(pred) - float(gold)) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if pred == gold else 0.0


@dataclass(frozen=True)
class EvalItem:
    prompt: str
    gold: str


@dataclass(frozen=True)
class EvalSuite:
    name: str
    items: tuple[EvalItem, ...]
    metric: str = "exact_match"
    scorer: Callable[[str, str], float] = field(default=numeric_exact_match)


SMOKE_MATH = EvalSuite(
    name="smoke-math",
    items=(
        EvalItem("What is 2 + 2? Reply with the number only.", "4"),
        EvalItem("What is 10 - 3? Reply with the number only.", "7"),
        EvalItem("What is 5 * 6? Reply with the number only.", "30"),
        EvalItem("What is 12 / 4? Reply with the number only.", "3"),
    ),
)

BUILTIN_SUITES: dict[str, EvalSuite] = {SMOKE_MATH.name: SMOKE_MATH}
