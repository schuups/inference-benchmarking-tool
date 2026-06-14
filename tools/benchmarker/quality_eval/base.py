"""Quality-eval backend seam (SPECIFICATIONS.md §13.5, IMPLEMENTATION_PLAN.md M11).

`QualityEvalRunner` (runner.py) implements M7's `QualityEvaluator` protocol and
delegates the actual scoring of a suite against the deployed endpoint to an
`EvalBackend`:

- `BuiltinEvalBackend` (grader.py) — an in-process grader over small registered
  suites; the tested path (mock canned answers) and a cheap custom-gate vehicle.
- `LmEvalBackend` (lm_eval_backend.py) — the EleutherAI lm-evaluation-harness, the
  spec's standard-suite engine (gsm8k, gpqa_diamond, …); validated at E1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class QualityEvalError(RuntimeError):
    """Infrastructure failure in the eval phase (missing harness/suite, etc.) —
    distinct from a quality *gate failure*, which is a normal graded outcome."""


@dataclass
class EvalScore:
    metric: str  # e.g. "exact_match"
    score: float  # mean over the graded items, [0, 1]
    sample_size: int  # number of items actually graded
    sampling_params: dict  # decoding params actually used (§14.9)
    harness_version: str  # provenance (§14.9)


class EvalBackend(Protocol):
    async def evaluate(
        self,
        endpoint: str,
        model: str,
        suite: str,
        sample_size: int | None,
        eval_concurrency: int,
    ) -> EvalScore:
        """Grade `suite` against the OpenAI-compatible `endpoint`; return the score."""
