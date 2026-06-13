"""lm-eval-harness EvalBackend (§12.5) — the standard-suite engine.

Runs the EleutherAI lm-evaluation-harness against the deployed OpenAI-compatible
endpoint via its `local-chat-completions` model, with natural decoding and the
suite-defined sampling/metrics. This is the production path for published suites
(gsm8k, gpqa_diamond, …); the harness owns their datasets, prompts, and scoring.

PROVISIONAL — like the vLLM model-load regexes (readiness.py) and the NVSHMEM
parsers (grade.py), the lm-eval model-args and result-parsing here are seeded
from the harness's documented shapes and MUST be validated against the pinned
lm-eval version on the Benchmarker image at E1 (capture a results fixture then).
lm-eval is not a laptop dependency; it ships in the M5 benchmarker image.

Gated suites (open decision 8): GPQA-Diamond (`gpqa_diamond`) is HF-gated — before
it runs, accept the dataset license on HuggingFace and provide `HF_TOKEN` in the
Benchmarker environment (the engine/benchmarker jobs already export
`HUGGING_FACE_HUB_TOKEN` from `~/.hf_token`). Without access, swap in an ungated
hard suite in `compare.suites`. GSM8K is ungated and works out of the box.

Thinking-model answer parsing (reasoning delimiters, e.g. DeepSeek `<think>`) is
not handled here yet — tracked in TODOs.md; not blocking for non-thinking gates.
"""

from __future__ import annotations

import asyncio
import logging

from .base import EvalScore, QualityEvalError

log = logging.getLogger("quality_eval.lm_eval")


class LmEvalBackend:
    def __init__(self, *, num_fewshot: int | None = None):
        self._num_fewshot = num_fewshot

    async def evaluate(
        self,
        endpoint: str,
        model: str,
        suite: str,
        sample_size: int | None,
        eval_concurrency: int,
    ) -> EvalScore:
        try:
            import lm_eval  # noqa: PLC0415 — lazy: only the Benchmarker image ships it
        except ImportError as exc:
            raise QualityEvalError(
                "lm-eval-harness is not installed — it is required for standard quality "
                "suites (§12.5) and ships in the M5 Benchmarker image. Use a builtin suite "
                "for a network-free gate, or set skip_quality_gate/skip_quality_compare."
            ) from exc

        # local-chat-completions targets an OpenAI-compatible /v1/chat/completions.
        model_args = ",".join(
            [
                f"base_url={endpoint}/v1/chat/completions",
                f"model={model}",
                f"num_concurrent={max(1, eval_concurrency)}",
                "tokenized_requests=False",
            ]
        )

        def _run() -> dict:
            return lm_eval.simple_evaluate(
                model="local-chat-completions",
                model_args=model_args,
                tasks=[suite],
                limit=sample_size,
                num_fewshot=self._num_fewshot,
                apply_chat_template=True,
            )

        # simple_evaluate is synchronous and CPU/IO-bound on the harness side.
        results = await asyncio.to_thread(_run)
        metric, score = _pick_metric(results, suite)
        graded = _graded_count(results, suite, sample_size)
        return EvalScore(
            metric=metric,
            score=score,
            sample_size=graded,
            sampling_params={"apply_chat_template": True, "num_fewshot": self._num_fewshot},
            harness_version=getattr(lm_eval, "__version__", "lm-eval-unknown"),
        )


def _pick_metric(results: dict, suite: str) -> tuple[str, float]:
    """Pull the primary scalar metric for `suite` from lm-eval's results dict.

    lm-eval keys metrics like 'exact_match,strict-match'; '_stderr' variants and
    'alias' are skipped. Prefers exact_match/acc; falls back to the first scalar.
    """
    table = (results.get("results") or {}).get(suite)
    if not isinstance(table, dict):
        raise QualityEvalError(f"lm-eval returned no results for suite {suite!r}")
    scalars = {
        k: v
        for k, v in table.items()
        if isinstance(v, (int, float)) and not k.endswith("_stderr") and k != "alias"
    }
    if not scalars:
        raise QualityEvalError(f"lm-eval results for {suite!r} carry no scalar metric: {table}")
    for pref in ("exact_match", "acc_norm", "acc"):
        for key, val in scalars.items():
            if key.split(",")[0] == pref:
                return key, float(val)
    key = next(iter(scalars))
    return key, float(scalars[key])


def _graded_count(results: dict, suite: str, sample_size: int | None) -> int:
    n = (results.get("n-samples") or {}).get(suite, {})
    if isinstance(n, dict) and "effective" in n:
        return int(n["effective"])
    return int(sample_size) if sample_size else 0
