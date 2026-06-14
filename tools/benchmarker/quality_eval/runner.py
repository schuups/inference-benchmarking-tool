"""Quality-eval runner (§13.5, M11) — implements M7's QualityEvaluator seam.

Stage A (pre-sweep sanity gate): grade a small subset of `gate.suite` against a
blunt absolute `gate.floor`; pass/fail. Stage B (post-sweep comparison): grade
each `compare.suites` at each `compare.eval_concurrency` level; measurement, not
a gate. Both produce `quality_evals` rows (§14.9) without `run_id` — the
orchestrator (M7) stamps `run_id` and persists them.

There is no standing quality reference: Stage-B deltas are computed in-report
across the experiment's deployment configs (§13.5).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from tools.benchmarker.orchestrator import GateOutcome, Instance

from .base import EvalBackend, EvalScore, QualityEvalError

log = logging.getLogger("quality_eval")


class QualityEvalRunner:
    def __init__(self, backend: EvalBackend):
        self._backend = backend

    async def stage_a_gate(self, instances: list[Instance], model: str, gate) -> GateOutcome:
        endpoint, instance_id = self._target(instances)
        score = await self._backend.evaluate(
            endpoint, model, gate.suite, gate.sample_size, eval_concurrency=1
        )
        passed = score.score >= gate.floor
        log.info(
            "Stage-A gate: %s = %.4f vs floor %.4f → %s",
            gate.suite, score.score, gate.floor, "PASS" if passed else "FAIL",
        )
        row = self._row(
            instance_id, "gate", gate.suite, 1, score,
            floor=gate.floor, status="pass" if passed else "fail",
        )
        return GateOutcome(rows=[row], passed=passed)

    async def stage_b_compare(self, instances: list[Instance], model: str, compare) -> list[dict]:
        endpoint, instance_id = self._target(instances)
        rows: list[dict] = []
        for suite in compare.suites:
            for concurrency in compare.eval_concurrency:
                score = await self._backend.evaluate(
                    endpoint, model, suite, sample_size=None, eval_concurrency=concurrency
                )
                log.info("Stage-B: %s @ concurrency=%d = %.4f", suite, concurrency, score.score)
                rows.append(
                    self._row(instance_id, "compare", suite, concurrency, score, floor=None, status=None)
                )
        return rows

    @staticmethod
    def _target(instances: list[Instance]) -> tuple[str, str]:
        if not instances:
            raise QualityEvalError("no deployed instances to evaluate against")
        return instances[0].base_url, instances[0].instance_id

    @staticmethod
    def _row(instance_id, stage, suite, concurrency, score: EvalScore, *, floor, status) -> dict:
        return {
            "instance_id": instance_id,
            "stage": stage,
            "suite": suite,
            "eval_concurrency": concurrency,
            "sample_size": score.sample_size,
            "metric": score.metric,
            "score": score.score,
            "floor": floor,
            "status": status,
            "sampling_params": score.sampling_params,
            "harness_version": score.harness_version,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
