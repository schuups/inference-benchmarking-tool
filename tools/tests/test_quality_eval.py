"""M11 DoD: against the mock server's canned answers, verify the builtin grader,
Stage-A pass/fail logic, Stage-B score collection, and quality_evals row shape.
"""

import pytest

from tools.common.results_db import ResultsDB
from tools.benchmarker.orchestrator import Instance
from tools.benchmarker.quality_eval.base import QualityEvalError
from tools.benchmarker.quality_eval.grader import BuiltinEvalBackend
from tools.benchmarker.quality_eval.runner import QualityEvalRunner
from tools.benchmarker.quality_eval.suites import numeric_exact_match
from tools.common.config import QualityCompare, QualityGate
from tools.testing.mock_openai_server import MockConfig, run_server

BASE_PORT = 8950
# Mock canned answers matching the builtin smoke-math suite prompts.
CORRECT = [("2 + 2", "4"), ("10 - 3", "7"), ("5 * 6", "30"), ("12 / 4", "3")]


def _instances(port):
    return [Instance("i0", f"http://127.0.0.1:{port}", node="testnode")]


# ----------------------------------------------------------------- scorer unit


def test_numeric_exact_match():
    assert numeric_exact_match("The answer is 4", "4") == 1.0
    assert numeric_exact_match("4", "4") == 1.0
    assert numeric_exact_match("1,234", "1234") == 1.0  # commas stripped
    assert numeric_exact_match("the answer is five", "5") == 0.0  # no digits
    assert numeric_exact_match("it is 5", "4") == 0.0


# ----------------------------------------------------------------- backend


@pytest.mark.asyncio
async def test_builtin_backend_scores_correct_and_wrong():
    runner = await run_server(MockConfig(ttft_ms=2, tpot_ms=0.0, canned=CORRECT), BASE_PORT)
    try:
        score = await BuiltinEvalBackend().evaluate(
            f"http://127.0.0.1:{BASE_PORT}", "mock/model", "smoke-math", None, 1
        )
        assert score.score == 1.0
        assert score.sample_size == 4
        assert score.metric == "exact_match"
        assert score.harness_version == "builtin-1"
    finally:
        await runner.cleanup()

    # no canned answers → filler tokens → no numeric match → 0.0
    runner = await run_server(MockConfig(ttft_ms=2, tpot_ms=0.0), BASE_PORT + 1)
    try:
        score = await BuiltinEvalBackend().evaluate(
            f"http://127.0.0.1:{BASE_PORT + 1}", "mock/model", "smoke-math", None, 4
        )
        assert score.score == 0.0
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_builtin_backend_unknown_suite_raises():
    with pytest.raises(QualityEvalError, match="no suite"):
        await BuiltinEvalBackend().evaluate("http://127.0.0.1:1", "m", "gsm8k", None, 1)


# ----------------------------------------------------------------- runner


@pytest.mark.asyncio
async def test_stage_a_gate_pass(tmp_path):
    runner = await run_server(MockConfig(ttft_ms=2, tpot_ms=0.0, canned=CORRECT), BASE_PORT + 2)
    try:
        q = QualityEvalRunner(BuiltinEvalBackend())
        gate = QualityGate(suite="smoke-math", sample_size=4, floor=0.5)
        outcome = await q.stage_a_gate(_instances(BASE_PORT + 2), "mock/model", gate)
        assert outcome.passed is True
        assert len(outcome.rows) == 1
        row = outcome.rows[0]
        assert row["stage"] == "gate" and row["status"] == "pass"
        assert row["suite"] == "smoke-math" and row["floor"] == 0.5
        assert row["score"] == 1.0 and row["eval_concurrency"] == 1
        # rows carry no run_id (the orchestrator stamps it); shape matches §14.9
        assert "run_id" not in row
        db = ResultsDB(tmp_path / "r.db")
        db.insert_many("quality_evals", [{"run_id": "run1", **r} for r in outcome.rows])
        assert db.count("quality_evals") == 1
        db.close()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_stage_a_gate_fail():
    runner = await run_server(MockConfig(ttft_ms=2, tpot_ms=0.0), BASE_PORT + 3)  # no canned → 0.0
    try:
        q = QualityEvalRunner(BuiltinEvalBackend())
        gate = QualityGate(suite="smoke-math", sample_size=4, floor=0.5)
        outcome = await q.stage_a_gate(_instances(BASE_PORT + 3), "mock/model", gate)
        assert outcome.passed is False
        assert outcome.rows[0]["status"] == "fail"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_stage_b_collects_rows_per_suite_and_concurrency(tmp_path):
    runner = await run_server(MockConfig(ttft_ms=2, tpot_ms=0.0, canned=CORRECT), BASE_PORT + 4)
    try:
        q = QualityEvalRunner(BuiltinEvalBackend())
        compare = QualityCompare(suites=["smoke-math"], eval_concurrency=[1, 2])
        rows = await q.stage_b_compare(_instances(BASE_PORT + 4), "mock/model", compare)
        assert len(rows) == 2  # 1 suite × 2 concurrency levels
        assert {r["eval_concurrency"] for r in rows} == {1, 2}
        for r in rows:
            assert r["stage"] == "compare"
            assert r["floor"] is None and r["status"] is None  # measurement, not a gate
            assert r["score"] == 1.0
        # persists into §14.9 quality_evals without shape drift
        db = ResultsDB(tmp_path / "r.db")
        db.insert_many("quality_evals", [{"run_id": "run1", **r} for r in rows])
        assert db.count("quality_evals") == 2
        db.close()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_runner_no_instances_raises():
    q = QualityEvalRunner(BuiltinEvalBackend())
    with pytest.raises(QualityEvalError, match="no deployed instances"):
        await q.stage_a_gate([], "m", QualityGate(suite="smoke-math"))
