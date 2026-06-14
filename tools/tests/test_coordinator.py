"""M8 DoD: Coordinator merge idempotency, resumable state, teardown (both paths),
precheck policy, full orchestration vs a fake backend, and the >100MB staged
per-run DB round-trip.
"""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from tools.common.results_db import ResultsDB
from tools.coordinator.backend import FakeClusterBackend
from tools.coordinator.coordinator import Coordinator, CoordinatorError
from tools.coordinator.merge import merge_run_db
from tools.coordinator.policy import decide_on_precheck
from tools.coordinator.state import PHASES, RunState
from tools.coordinator.teardown import teardown_plan


# --------------------------------------------------------------- fixtures/helpers


def _make_run_db(path: Path, run_id: str, n_requests: int = 3) -> None:
    db = ResultsDB(path)
    db.insert("experiments", {"run_id": run_id, "model": "m", "backend": "vllm",
                              "created_at": "2026-06-13T00:00:00+00:00"})
    db.insert("instances", {"run_id": run_id, "instance_id": "i0", "endpoint": "http://n:8000"})
    db.insert_many("requests", [
        {"run_id": run_id, "rate_lambda": 1.0, "request_id": i, "session_idx": i,
         "instance_id": "i0", "scenario": "smoke-synthetic", "turn_idx": 0,
         "issued_at_ms": float(i), "final_turn": 1, "ttft_ms": 50.0, "tpot_ms": 5.0,
         "e2e_ms": 100.0, "input_tokens": 10, "output_tokens": 5, "success": 1, "error": None}
        for i in range(n_requests)
    ])
    db.close()


def _count(db_path: Path, table: str, run_id: str | None = None) -> int:
    conn = sqlite3.connect(db_path)
    try:
        if run_id is None:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id=?", (run_id,)).fetchone()[0]
    finally:
        conn.close()


def _state(tmp_path, run_id, platform="slurm", target="clariden"):
    return RunState(
        run_id=run_id, platform=platform, target=target,
        run_dir_local=str(tmp_path / "exp" / run_id),
        run_dir_remote=str(tmp_path / "remote" / run_id),
    )


PASS_PRECHECK = {"gate_exit_code": 0, "smoke_test_mode": False,
                 "rows": [{"metric": "nccl_all_reduce", "measured": 120.0, "status": "pass"}]}


# ------------------------------------------------------------------------- merge


def test_merge_idempotent_and_multi_run(tmp_path):
    a, b, central = tmp_path / "a.db", tmp_path / "b.db", tmp_path / "central.db"
    _make_run_db(a, "runA", n_requests=3)
    _make_run_db(b, "runB", n_requests=2)

    merge_run_db(a, central, "runA")
    merge_run_db(b, central, "runB")
    assert _count(central, "requests") == 5
    assert _count(central, "experiments") == 2

    merge_run_db(a, central, "runA")  # re-merge is idempotent (delete-then-insert)
    assert _count(central, "requests") == 5
    assert _count(central, "experiments") == 2
    assert _count(central, "requests", "runA") == 3


def test_merge_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        merge_run_db(tmp_path / "nope.db", tmp_path / "c.db", "r")


# ------------------------------------------------------------------------- state


def test_runstate_roundtrip_and_phase_logic(tmp_path):
    rd = tmp_path / "run"
    RunState(run_id="r", platform="slurm", target="clariden",
             run_dir_local=str(rd), run_dir_remote="/scratch/r").save()
    assert RunState.exists(rd)

    loaded = RunState.load(rd)
    assert loaded.run_id == "r" and loaded.phase == "created"
    assert loaded.is_before("staged") and not loaded.is_done("staged")

    loaded.advance("submitted")
    assert loaded.is_done("staged") and loaded.is_done("submitted")
    assert loaded.is_before("completed")
    assert RunState.load(rd).phase == "submitted"  # persisted


def test_runstate_rejects_bad_phase(tmp_path):
    with pytest.raises(ValueError):
        RunState(run_id="r", platform="slurm", target="c",
                 run_dir_local=str(tmp_path), run_dir_remote="x", phase="bogus")


# ------------------------------------------------------------------------ policy


def test_decide_on_precheck():
    fail = {"gate_exit_code": 4, "rows": [{"metric": "x", "measured": 5, "status": "fail"}]}
    assert decide_on_precheck(fail, "abort", "abort").proceed is False
    assert decide_on_precheck(fail, "abort", "continue").proceed is True
    warn = {"gate_exit_code": 3, "rows": [{"metric": "y", "measured": 1, "status": "warn"}]}
    assert decide_on_precheck(warn, "abort", "abort").proceed is False
    assert decide_on_precheck(warn, "continue", "abort").proceed is True
    ok = {"gate_exit_code": 0, "rows": [{"metric": "z", "measured": 1, "status": "pass"}]}
    assert decide_on_precheck(ok, "abort", "abort").proceed is True


# ---------------------------------------------------------------------- teardown


def test_teardown_plan_slurm_and_k8s_pvc_retention(tmp_path):
    slurm = _state(tmp_path, "r1")
    slurm.benchmarker_handle, slurm.engine_handles = "jb", ["je"]
    kinds = {(a.kind, a.target) for a in teardown_plan(slurm)}
    assert ("cancel", "jb") in kinds and ("cancel", "je") in kinds
    assert ("remove_dir", slurm.run_dir_remote) in kinds

    k8s = _state(tmp_path, "r2", platform="k8s", target="breithorn")
    k8s.benchmarker_handle, k8s.engine_handles = "pod/b", ["deployment/e", "service/e"]
    plan = teardown_plan(k8s)
    assert all(a.kind != "remove_dir" for a in plan)  # §7.6: PVC/scratch retained on K8s
    assert any(a.target == "deployment/e" for a in plan)


# ------------------------------------------------------------------ orchestration


@pytest.mark.asyncio
async def test_coordinator_happy_path(tmp_path):
    run_id = "20260613-000000_mock-model_vllm_clariden_aa00"
    state = _state(tmp_path, run_id)
    Path(state.run_dir_local).mkdir(parents=True)
    fixture = tmp_path / "fixture_run.db"
    _make_run_db(fixture, run_id, n_requests=4)
    backend = FakeClusterBackend(
        result_db=fixture, precheck=PASS_PRECHECK,
        status_script=["running", "completed"], engine_handles=["job-engine-1"],
    )
    central = tmp_path / "results.db"
    final = await Coordinator(state, backend, central, poll_interval_s=0, max_polls=20).run()

    assert final.phase == "torn_down"
    assert final.benchmarker_handle == "job-fake-1"
    assert final.engine_handles == ["job-engine-1"]
    assert final.db_sha256
    assert (Path(state.run_dir_local) / f"run_{run_id}.db").exists()  # collected locally
    assert _count(central, "requests", run_id) == 4  # merged
    assert _count(central, "experiments", run_id) == 1
    # teardown ran on success: both jobs cancelled + scratch removed (§7)
    assert ("cancel", "job-fake-1") in backend.calls
    assert ("cancel", "job-engine-1") in backend.calls
    assert ("remove_dir", state.run_dir_remote) in backend.calls
    assert not Path(state.run_dir_remote).exists()


@pytest.mark.asyncio
async def test_coordinator_resume_skips_completed_phases(tmp_path):
    run_id = "20260613-000000_mock-model_vllm_clariden_bb11"
    state = _state(tmp_path, run_id)
    Path(state.run_dir_local).mkdir(parents=True)
    # simulate a prior run that already staged + submitted: pre-plant remote artifacts
    remote = Path(state.run_dir_remote)
    (remote / "prechecks").mkdir(parents=True)
    (remote / "prechecks" / "results.json").write_text('{"gate_exit_code":0,"rows":[]}')
    _make_run_db(remote / f"run_{run_id}.db", run_id, n_requests=2)
    state.benchmarker_handle = "job-fake-1"
    state.advance("submitted")

    backend = FakeClusterBackend(status_script=["completed"], engine_handles=["job-engine-1"])
    central = tmp_path / "results.db"
    final = await Coordinator(state, backend, central, poll_interval_s=0, max_polls=5).run()

    assert final.phase == "torn_down"
    assert not any(c[0] == "stage" for c in backend.calls)  # skipped — already staged
    assert not any(c[0] == "submit" for c in backend.calls)  # skipped — already submitted
    assert _count(central, "requests", run_id) == 2


@pytest.mark.asyncio
async def test_coordinator_teardown_on_failure(tmp_path):
    run_id = "20260613-000000_mock-model_vllm_clariden_cc22"
    state = _state(tmp_path, run_id)
    Path(state.run_dir_local).mkdir(parents=True)
    fail_precheck = {"gate_exit_code": 4, "rows": [{"metric": "nccl", "measured": 5.0, "status": "fail"}]}
    backend = FakeClusterBackend(
        result_db=None, precheck=fail_precheck,
        status_script=["failed"], engine_handles=["job-engine-1"],
    )
    central = tmp_path / "results.db"
    with pytest.raises(CoordinatorError, match="failed"):
        await Coordinator(state, backend, central, poll_interval_s=0, on_fail="abort").run()

    assert state.phase == "torn_down"  # §7: teardown still runs on the failure path
    assert state.error
    assert ("cancel", "job-fake-1") in backend.calls
    assert ("cancel", "job-engine-1") in backend.calls


# ----------------------------------------------------------- staged DB transfer


@pytest.mark.asyncio
async def test_staged_download_roundtrip_large_db(tmp_path):
    src = tmp_path / "remote" / "run_x.db"
    src.parent.mkdir(parents=True)
    db = ResultsDB(src)
    db.insert("experiments", {"run_id": "x", "model": "m", "backend": "vllm",
                              "scenario_manifest": "x" * (105 * 1024 * 1024), "created_at": "t"})
    db.close()
    assert src.stat().st_size > 100 * 1024 * 1024  # >100MB per the M8 DoD

    local = tmp_path / "local" / "run_x.db"
    sha = await FakeClusterBackend().fetch_db(str(src), local)

    assert sha == hashlib.sha256(src.read_bytes()).hexdigest()  # bit-for-bit through staging
    conn = sqlite3.connect(local)  # and still a valid, queryable SQLite DB
    try:
        assert conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
    finally:
        conn.close()


def test_phases_are_linear():
    assert PHASES[0] == "created" and PHASES[-1] == "torn_down"
