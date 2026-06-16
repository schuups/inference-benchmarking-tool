"""M7 DoD: Benchmarker orchestrator phase ordering, persistence, gate handling.

Drives run_experiment() in-process against the mock OpenAI server via an injected
MockLauncher (no cluster), with a stub QualityEvaluator standing in for M11.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.common.config import SCENARIOS_DIR, BenchmarkConfig
from tools.benchmarker.dataset_gen.tokenizers import WordTokenizer
from tools.benchmarker.orchestrator import (
    GateOutcome,
    Instance,
    RunAborted,
    run_experiment,
)
from tools.testing.mock_openai_server import MockConfig, run_server

# vLLM-style log lines so parse_model_load (§10.2) populates the instances row.
ENGINE_LOG = (
    "INFO 06-13 [model_runner.py] Model loading took 12.3 GiB and 5.50 seconds\n"
    "INFO 06-13 [llm_engine.py] init engine (profile, create kv cache, warmup model) took 8.20 seconds\n"
)
TABLES = (
    "experiments", "instances", "requests", "server_stats",
    "hardware_stats", "system_prechecks", "quality_evals",
)


def _cfg(**overrides) -> BenchmarkConfig:
    base = {
        "name": "orch-test",
        "deployments": [
            {"target": "clariden", "backend": "vllm", "backend_version": "v0.22.1",
             "model": "mock/model"},
        ],
        "dataset_config": {
            "scenario_mix": [{"scenario": "smoke-synthetic", "weight": 1.0}],
            "num_prompts": 60, "seed": 7,
        },
        "rate_levels": [2.0],
        "phases": {"warmup_s": 1, "measurement_s": 1, "drain_timeout_s": 3, "request_timeout_s": 5},
    }
    base.update(overrides)
    return BenchmarkConfig.model_validate(base)


def _passing_precheck(*, smoke=False, gate_exit_code=0, rows=None) -> dict:
    if rows is None:
        rows = [
            {"metric": "nccl_all_reduce_128_mib", "measured": 120.0, "expected": None,
             "tolerance_pct": None, "status": "pass", "ts": "2026-06-13T00:00:00+00:00"},
            {"metric": "sequential_read_1_mib_blocks", "measured": 8.0, "expected": None,
             "tolerance_pct": None, "status": "pass", "ts": "2026-06-13T00:00:00+00:00"},
        ]
    return {"cluster": "clariden", "scope": "4× GH200, 1 node",
            "smoke_test_mode": smoke, "gate_exit_code": gate_exit_code, "rows": rows}


def _counts(db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        conn.close()


class MockLauncher:
    """In-process engine stand-in: records pre-submit pool state, plants the
    pre-check results.json + sampler NDJSON, and serves the mock OpenAI API."""

    def __init__(self, port, *, precheck=None, emit_hw=False, engine_log="",
                 alive=True, start_server=True):
        self.port = port
        self._precheck = precheck
        self._emit_hw = emit_hw
        self._engine_log = engine_log
        self._alive = alive
        self._start_server = start_server
        self.pool_existed_at_submit = None
        self.teardown_called = False
        self._runner = None

    async def submit(self, run_dir):
        run_dir = Path(run_dir)
        # §1 ordering: the dataset pool must already exist when the engine is spawned.
        self.pool_existed_at_submit = (run_dir / "dataset" / "prompts.jsonl").exists()
        if self._precheck is not None:
            pdir = run_dir / "prechecks"
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / "results.json").write_text(json.dumps(self._precheck))
        if self._emit_hw:
            base = datetime.now(timezone.utc)
            lines = []
            for i in range(300):  # 0.2s × 300 = 60s — overlaps the imminent sweep windows
                ts = (base + timedelta(seconds=0.2 * i)).isoformat()
                lines.append(json.dumps({"ts": ts, "node": "testnode", "gpu_index": 0, "gpu_util_pct": 55.0}))
                lines.append(json.dumps({"ts": ts, "node": "testnode", "gpu_index": None, "cpu_util_pct": 30.0}))
            (run_dir / "hw-testnode.ndjson").write_text("\n".join(lines) + "\n")
        if self._start_server:
            self._runner = await run_server(MockConfig(ttft_ms=2, tpot_ms=0.0), self.port)
        return [Instance("i0", f"http://127.0.0.1:{self.port}", node="testnode")]

    def engine_log_text(self):
        return self._engine_log

    def is_alive(self):
        return self._alive

    async def teardown(self):
        self.teardown_called = True
        if self._runner:
            await self._runner.cleanup()


class StubQuality:
    """Stands in for M11. gate_passed drives the Stage-A gate outcome."""

    def __init__(self, gate_passed=True):
        self.gate_passed = gate_passed

    async def stage_a_gate(self, instances, model, gate) -> GateOutcome:
        return GateOutcome(
            rows=[{
                "instance_id": instances[0].instance_id, "stage": "gate", "suite": gate.suite,
                "eval_concurrency": 1, "sample_size": gate.sample_size, "metric": "exact_match",
                "score": 0.8 if self.gate_passed else 0.1, "floor": gate.floor,
                "status": "pass" if self.gate_passed else "fail",
                "sampling_params": {"temperature": 0.0}, "harness_version": "stub",
                "ts": "2026-06-13T00:00:00+00:00",
            }],
            passed=self.gate_passed,
        )

    async def stage_b_compare(self, instances, model, compare) -> list[dict]:
        return [
            {"instance_id": instances[0].instance_id, "stage": "compare", "suite": suite,
             "eval_concurrency": conc, "sample_size": 10, "metric": "exact_match", "score": 0.7,
             "floor": None, "status": None, "sampling_params": {"top_p": 1.0},
             "harness_version": "stub", "ts": "2026-06-13T00:00:00+00:00"}
            for suite in compare.suites for conc in compare.eval_concurrency
        ]


async def _run(launcher, run_dir, run_id, *, cfg=None, quality=None):
    cfg = cfg or _cfg()
    return await run_experiment(
        cfg, cfg.deployments[0], run_id, run_dir, WordTokenizer(), launcher, SCENARIOS_DIR,
        quality=quality,
    )


@pytest.mark.asyncio
async def test_full_run_persists_all_tables_dataset_before_engine(tmp_path):
    run_dir = tmp_path / "run"
    launcher = MockLauncher(8930, precheck=_passing_precheck(), emit_hw=True, engine_log=ENGINE_LOG)
    summary = await _run(
        launcher, run_dir, "20260613-000000_mock-model_vllm_clariden_a0",
        quality=StubQuality(),
    )

    assert launcher.pool_existed_at_submit is True  # §1: dataset generated before engine spawn
    assert launcher.teardown_called is True
    assert summary.persisted and not summary.smoke_test_mode
    assert summary.instances == 1 and summary.requests > 0 and not summary.quality_flagged

    db_path = run_dir / f"run_{summary.run_id}.db"
    assert db_path.exists()
    c = _counts(db_path)
    assert c["experiments"] == 1
    assert c["instances"] == 1
    assert c["requests"] > 0
    assert c["server_stats"] > 0
    assert c["system_prechecks"] == 2
    assert c["quality_evals"] > 0    # 1 gate + Stage-B compare rows
    assert c["hardware_stats"] > 0   # sampler NDJSON ingested into the sweep window

    conn = sqlite3.connect(db_path)
    try:
        weights, total = conn.execute(
            "SELECT model_load_weights_s, model_load_total_s FROM instances"
        ).fetchone()
        assert weights == pytest.approx(5.50)   # parsed from ENGINE_LOG (§10.2)
        assert total is not None and total >= 0
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_reuses_prestaged_dataset_pool(tmp_path, monkeypatch):
    # §11.4: identical dataset_config+seed across a sweep's cells -> identical pool, so
    # the Coordinator generates it once and stages pool+manifest into each run_dir.
    # The orchestrator must REUSE the staged pool and NOT regenerate.
    import shutil
    from tools.benchmarker.dataset_gen.generator import (
        MANIFEST_FILENAME, POOL_FILENAME, generate as real_generate,
    )

    cfg = _cfg()
    src = tmp_path / "src" / "dataset"
    real_generate(cfg, WordTokenizer(), src, SCENARIOS_DIR)  # produce a pool to stage

    run_dir = tmp_path / "run"
    (run_dir / "dataset").mkdir(parents=True)
    for name in (POOL_FILENAME, MANIFEST_FILENAME):
        shutil.copy(src / name, run_dir / "dataset" / name)

    def _boom(*a, **k):
        raise AssertionError("generate() called despite a pre-staged pool")

    monkeypatch.setattr("tools.benchmarker.orchestrator.generate", _boom)
    launcher = MockLauncher(8931, precheck=_passing_precheck(), emit_hw=True, engine_log=ENGINE_LOG)
    summary = await _run(
        launcher, run_dir, "20260613-000000_mock-model_vllm_clariden_b0", quality=StubQuality(),
    )
    assert launcher.pool_existed_at_submit is True
    assert summary.persisted and summary.requests > 0


@pytest.mark.asyncio
async def test_smoke_mode_no_persistence_and_two_warnings(tmp_path, caplog):
    run_dir = tmp_path / "run"
    launcher = MockLauncher(8931, precheck=_passing_precheck(smoke=True), engine_log=ENGINE_LOG)
    with caplog.at_level(logging.WARNING):
        summary = await _run(
            launcher, run_dir, "20260613-000000_mock-model_vllm_clariden_smk",
            quality=StubQuality(),
        )
    assert summary.smoke_test_mode and not summary.persisted
    assert summary.db_path is None
    assert not (run_dir / f"run_{summary.run_id}.db").exists()  # §8.2: nothing on disk
    smoke = [r for r in caplog.records if "SMOKE-TEST MODE" in r.getMessage()]
    assert len(smoke) >= 2  # unmissable at launch and at termination (§8.2)


@pytest.mark.asyncio
async def test_precheck_gate_abort_persists_prechecks_then_aborts(tmp_path):
    run_dir = tmp_path / "run"
    fail_row = [{"metric": "nccl_all_reduce_128_mib", "measured": 5.0, "expected": 122.0,
                 "tolerance_pct": -10.0, "status": "fail", "ts": "2026-06-13T00:00:00+00:00"}]
    launcher = MockLauncher(
        8932, precheck=_passing_precheck(gate_exit_code=4, rows=fail_row), engine_log=ENGINE_LOG
    )
    run_id = "20260613-000000_mock-model_vllm_clariden_gate"
    with pytest.raises(RunAborted, match="pre-check gate aborted"):
        await _run(launcher, run_dir, run_id, quality=StubQuality())
    assert launcher.teardown_called is True
    c = _counts(run_dir / f"run_{run_id}.db")
    assert c["experiments"] == 1 and c["system_prechecks"] == 1  # §8.4: measurements persisted
    assert c["requests"] == 0  # engine never started → no sweep


@pytest.mark.asyncio
async def test_quality_gate_failure_aborts(tmp_path):
    run_dir = tmp_path / "run"
    launcher = MockLauncher(8933, precheck=_passing_precheck(), engine_log=ENGINE_LOG)
    run_id = "20260613-000000_mock-model_vllm_clariden_qab"
    with pytest.raises(RunAborted, match="Stage-A quality gate FAILED"):
        await _run(launcher, run_dir, run_id, quality=StubQuality(gate_passed=False))
    assert launcher.teardown_called is True
    c = _counts(run_dir / f"run_{run_id}.db")
    assert c["quality_evals"] == 1 and c["requests"] == 0  # gate row persisted, no sweep


@pytest.mark.asyncio
async def test_quality_gate_continue_flags_run(tmp_path, caplog):
    run_dir = tmp_path / "run"
    launcher = MockLauncher(8934, precheck=_passing_precheck(), engine_log=ENGINE_LOG)
    cfg = _cfg(quality_eval={"gate": {"on_fail": "continue"}})
    with caplog.at_level(logging.WARNING):
        summary = await _run(
            launcher, run_dir, "20260613-000000_mock-model_vllm_clariden_qcn",
            cfg=cfg, quality=StubQuality(gate_passed=False),
        )
    assert summary.quality_flagged is True
    assert summary.requests > 0  # ran the sweep despite the failed gate
    assert any("QUALITY-FLAGGED" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_no_quality_evaluator_skips_stages(tmp_path):
    run_dir = tmp_path / "run"
    launcher = MockLauncher(8935, precheck=_passing_precheck(), engine_log=ENGINE_LOG)
    summary = await _run(
        launcher, run_dir, "20260613-000000_mock-model_vllm_clariden_noq", quality=None,
    )
    assert summary.persisted and summary.requests > 0 and not summary.quality_flagged
    assert _counts(run_dir / f"run_{summary.run_id}.db")["quality_evals"] == 0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_dead_engine_before_readiness_aborts(tmp_path):
    run_dir = tmp_path / "run"
    launcher = MockLauncher(
        8936, precheck=None, start_server=False, alive=False, engine_log="boom: CUDA OOM"
    )
    with pytest.raises(RunAborted, match="exited before readiness"):
        await _run(
            launcher, run_dir, "20260613-000000_mock-model_vllm_clariden_dead",
            quality=StubQuality(),
        )
    assert launcher.teardown_called is True
