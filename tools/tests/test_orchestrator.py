"""M7 DoD: phase ordering, §7.4 gate handling, §7.2 smoke mode, finalisation."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from tools.common.config import BenchmarkConfig
from tools.benchmarker.dataset_gen.generator import POOL_FILENAME
from tools.benchmarker.dataset_gen.tokenizers import WordTokenizer
from tools.benchmarker.orchestrator import RunAborted, run_experiment
from tools.testing.mock_openai_server import MockConfig, run_server

PORT = 8920


def _config(tmp_path) -> tuple[BenchmarkConfig, Path]:
    registry = tmp_path / "scenarios"
    registry.mkdir()
    (registry / "smoke.yaml").write_text(yaml.safe_dump({
        "name": "smoke", "summary": "test", "maturity": "exploratory",
        "source": {"kind": "synthetic"},
        "input_length": {"distribution": "fixed", "params": {"value": 30}},
        "output_length": {"distribution": "fixed", "params": {"value": 5}},
        "session": {
            "mode": "open_loop",
            "turns_per_session": {"distribution": "fixed", "params": {"value": 1}},
            "prefix_strategy": "append_delta",
        },
        "manifest": {"modelled": ["pipeline"], "not_modelled": ["everything"]},
    }))
    cfg = BenchmarkConfig.model_validate({
        "name": "orchestrator-test",
        "deployments": [{"target": "clariden", "backend": "vllm",
                         "backend_version": "x", "model": "mock/model"}],
        "dataset_config": {"scenario_mix": [{"scenario": "smoke", "weight": 1.0}],
                           "num_prompts": 150, "seed": 7},
        "rate_levels": [20.0, 40.0],
        "phases": {"warmup_s": 1, "measurement_s": 1, "drain_timeout_s": 3,
                   "request_timeout_s": 5, "server_ready_timeout_s": 10},
    })
    return cfg, registry


class MockLauncher:
    """In-process engine: asserts §1 ordering, can plant §7 results + hw samples."""

    def __init__(self, port: int, precheck_payload: dict | None = None, hw_samples: bool = False):
        self.port = port
        self.precheck_payload = precheck_payload
        self.hw_samples = hw_samples
        self.pool_existed_at_launch: bool | None = None
        self.torn_down = False
        self._runner = None
        self._hw_task = None

    async def launch(self, run_dir: Path):
        self.pool_existed_at_launch = (run_dir / POOL_FILENAME).exists()
        if self.precheck_payload is not None:
            prechecks = run_dir / "prechecks"
            prechecks.mkdir(exist_ok=True)
            (prechecks / "results.json").write_text(json.dumps(self.precheck_payload))
        self._runner = await run_server(MockConfig(ttft_ms=5, tpot_ms=1), self.port)
        if self.hw_samples:
            async def emit():
                with open(run_dir / "hw-mocknode.ndjson", "a") as f:
                    while True:
                        f.write(json.dumps({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "node": "mocknode", "gpu_index": None, "cpu_util_pct": 12.5,
                        }) + "\n")
                        f.flush()
                        await asyncio.sleep(0.2)
            self._hw_task = asyncio.create_task(emit())
        return [("i0", f"http://127.0.0.1:{self.port}")]

    async def teardown(self):
        self.torn_down = True
        if self._hw_task:
            self._hw_task.cancel()
        if self._runner:
            await self._runner.cleanup()

    def engine_log_text(self):
        return "Model loading took 1.0 GiB and 2.50 seconds\n"

    def is_alive(self):
        return not self.torn_down


@pytest.mark.asyncio
async def test_end_to_end_phases_and_persistence(tmp_path):
    cfg, registry = _config(tmp_path)
    launcher = MockLauncher(PORT, precheck_payload={
        "gate_exit_code": 0, "smoke_test_mode": False,
        "rows": [{"metric": "nccl_all_reduce_128_mib", "measured": 120.0,
                  "expected": None, "tolerance_pct": None, "status": "pass",
                  "ts": "2026-06-12T00:00:00Z"}],
    }, hw_samples=True)
    run_dir = tmp_path / "run"
    summary = await run_experiment(cfg, cfg.deployments[0], "test-run", run_dir,
                                   WordTokenizer(), launcher, registry)
    # §1 ordering: the engine was launched only after the pool existed
    assert launcher.pool_existed_at_launch is True
    assert launcher.torn_down is True
    assert summary.persisted and not summary.smoke_test_mode
    assert summary.requests > 0 and summary.sessions_truncated == 0
    # M11 hooks pending — loudly recorded, not silently dropped
    assert summary.quality_stages_pending == ["gate", "compare"]

    import sqlite3
    conn = sqlite3.connect(run_dir / "run_test-run.db")
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("experiments", "instances", "requests", "server_stats",
                        "system_prechecks", "hardware_stats")}
    assert counts["experiments"] == 1
    assert counts["instances"] == 1
    assert counts["requests"] == summary.requests
    assert counts["server_stats"] > 0
    assert counts["system_prechecks"] == 1
    assert counts["hardware_stats"] > 0  # NDJSON ingested within λ windows
    load = conn.execute("SELECT model_load_weights_s FROM instances").fetchone()[0]
    assert load == pytest.approx(2.5)  # §9.2 parsed from the engine log
    # both λ levels measured
    rates = {r[0] for r in conn.execute("SELECT DISTINCT rate_lambda FROM requests")}
    assert rates == {20.0, 40.0}
    conn.close()


@pytest.mark.asyncio
async def test_precheck_gate_abort_tears_down(tmp_path):
    cfg, registry = _config(tmp_path)
    launcher = MockLauncher(PORT + 1, precheck_payload={
        "gate_exit_code": 3, "smoke_test_mode": False, "rows": [],
    })
    with pytest.raises(RunAborted, match="§7.4"):
        await run_experiment(cfg, cfg.deployments[0], "r", tmp_path / "run",
                             WordTokenizer(), launcher, registry)
    assert launcher.torn_down is True  # §6: teardown on the failure path too


@pytest.mark.asyncio
async def test_smoke_mode_persists_nothing_and_warns(tmp_path, caplog):
    cfg, registry = _config(tmp_path)
    launcher = MockLauncher(PORT + 2, precheck_payload={
        "gate_exit_code": 0, "smoke_test_mode": True, "rows": [],
    })
    run_dir = tmp_path / "run"
    with caplog.at_level("WARNING", logger="benchmarker"):
        summary = await run_experiment(cfg, cfg.deployments[0], "smoke-run", run_dir,
                                       WordTokenizer(), launcher, registry)
    assert summary.smoke_test_mode and not summary.persisted
    assert not (run_dir / "run_smoke-run.db").exists()  # §7.2: nothing on disk
    smoke_warnings = [r for r in caplog.records if "SMOKE-TEST MODE" in r.message]
    assert len(smoke_warnings) >= 2  # unmissable at launch AND termination (§7.2)


@pytest.mark.asyncio
async def test_dead_engine_fast_fails_with_gate_grace(tmp_path, monkeypatch):
    """The branch attempt #2 shipped broken: engine dies, no results.json."""
    from tools.benchmarker import orchestrator as orch

    monkeypatch.setattr(orch, "DEAD_ENGINE_GRACE_S", 0.1)
    cfg, registry = _config(tmp_path)

    class DeadEngineLauncher(MockLauncher):
        async def launch(self, run_dir):
            self.pool_existed_at_launch = (run_dir / POOL_FILENAME).exists()
            return [("i0", f"http://127.0.0.1:{PORT + 9}")]  # nothing listens

        def is_alive(self):
            return False

    launcher = DeadEngineLauncher(PORT + 9)
    with pytest.raises(orch.RunAborted, match="engine job died"):
        await run_experiment(cfg, cfg.deployments[0], "r", tmp_path / "run",
                             WordTokenizer(), launcher, registry)
    assert launcher.torn_down is True


@pytest.mark.asyncio
async def test_quality_hooks_invoked_when_provided(tmp_path):
    cfg, registry = _config(tmp_path)
    calls = []

    async def gate(endpoints, db, run_id):
        calls.append("gate")

    async def compare(endpoints, db, run_id):
        calls.append("compare")

    launcher = MockLauncher(PORT + 3)
    summary = await run_experiment(cfg, cfg.deployments[0], "q-run", tmp_path / "run",
                                   WordTokenizer(), launcher, registry,
                                   quality_gate=gate, quality_compare=compare)
    assert calls == ["gate", "compare"]  # gate before the sweep, compare after
    assert summary.quality_stages_pending == []
