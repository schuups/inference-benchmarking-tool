"""Builtin in-process grader (§13.5) — the tested EvalBackend.

Sends each suite item to the deployed OpenAI-compatible endpoint via the load
generator's streaming client (natural decoding, `ignore_eos=False`), scores the
response against the gold answer, and returns the mean. Concurrency-capped by
`eval_concurrency`. Fine-grained per-suite sampling control is lm-eval's job
(LmEvalBackend); this backend records the decoding params it actually sends.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from tools.benchmarker.load_gen.client import execute_request

from .base import EvalScore, QualityEvalError
from .suites import BUILTIN_SUITES, EvalSuite

log = logging.getLogger("quality_eval.builtin")

HARNESS_VERSION = "builtin-1"


class BuiltinEvalBackend:
    def __init__(
        self,
        suites: dict[str, EvalSuite] | None = None,
        *,
        max_tokens: int = 256,
        request_timeout_s: float = 120.0,
    ):
        self._suites = suites if suites is not None else BUILTIN_SUITES
        self._max_tokens = max_tokens
        self._request_timeout_s = request_timeout_s

    async def evaluate(
        self,
        endpoint: str,
        model: str,
        suite: str,
        sample_size: int | None,
        eval_concurrency: int,
    ) -> EvalScore:
        suite_obj = self._suites.get(suite)
        if suite_obj is None:
            raise QualityEvalError(
                f"builtin backend has no suite {suite!r} "
                f"(have {sorted(self._suites)}); use LmEvalBackend for standard suites"
            )
        items = suite_obj.items[:sample_size] if sample_size else suite_obj.items
        if not items:
            raise QualityEvalError(f"suite {suite!r} has no items")
        sem = asyncio.Semaphore(max(1, eval_concurrency))
        sampling_params = {"max_tokens": self._max_tokens, "ignore_eos": False}

        async with aiohttp.ClientSession() as http:
            async def grade_one(item) -> float:
                async with sem:
                    outcome = await execute_request(
                        http, endpoint,
                        model=model,
                        messages=[{"role": "user", "content": item.prompt}],
                        max_tokens=self._max_tokens,
                        ignore_eos=False,  # natural decoding (§13.5)
                        request_timeout_s=self._request_timeout_s,
                    )
                    if not outcome.success:
                        log.warning("eval item failed (scored 0): %s", outcome.error)
                        return 0.0
                    return suite_obj.scorer(outcome.output_text, item.gold)

            scores = await asyncio.gather(*[grade_one(i) for i in items])

        mean = sum(scores) / len(scores)
        return EvalScore(
            metric=suite_obj.metric,
            score=mean,
            sample_size=len(items),
            sampling_params=sampling_params,
            harness_version=HARNESS_VERSION,
        )
