"""M3 DoD: §14 schema conformance, write contention, smoke-mode, NDJSON ingestion."""

import json
import threading
from datetime import datetime, timedelta, timezone

from tools.common.results_db import ResultsDB
from tools.benchmarker.load_gen.scheduler import RequestRow

# Independent copy of the §14 column lists — catches silent drift in db.SCHEMA.
SPEC_COLUMNS = {
    "experiments": [
        "run_id", "model", "backend", "backend_config", "dataset_config",
        "scenario_mix", "scenario_manifest", "slos", "quality_eval",
        "rate_levels", "warmup_s", "measurement_s", "created_at",
    ],
    "instances": [
        "run_id", "instance_id", "endpoint", "node", "model_load_total_s",
        "model_load_weights_s", "model_load_engine_init_s",
        "model_load_cuda_graph_capture_s", "model_load_inductor_compile_s",
    ],
    "requests": [
        "run_id", "rate_lambda", "request_id", "session_idx", "instance_id",
        "scenario", "turn_idx", "issued_at_ms", "final_turn", "ttft_ms",
        "tpot_ms", "e2e_ms", "input_tokens", "output_tokens", "success", "error",
    ],
    "server_stats": [
        "run_id", "instance_id", "rate_lambda", "ts", "requests_running",
        "requests_waiting", "gpu_cache_pct", "spec_accept_rate",
    ],
    "hardware_stats": [
        "run_id", "instance_id", "rate_lambda", "ts", "gpu_index",
        "gpu_util_pct", "gpu_mem_used_gb", "gpu_mem_pct", "gpu_power_w",
        "gpu_temp_c", "gpu_sm_active_pct", "gpu_tensor_active_pct",
        "gpu_dram_bw_gbs", "nvlink_rx_gbs", "nvlink_tx_gbs", "pcie_rx_gbs",
        "pcie_tx_gbs", "cpu_util_pct", "cpu_iowait_pct", "ram_used_gb",
        "ram_pct", "ram_bw_gbs", "storage_read_gbs", "storage_read_iops",
        "net_rx_gbs", "net_tx_gbs",
    ],
    "system_prechecks": [
        "run_id", "instance_id", "metric", "measured", "expected",
        "tolerance_pct", "status", "ts",
    ],
    "quality_evals": [
        "run_id", "instance_id", "stage", "suite", "eval_concurrency",
        "sample_size", "metric", "score", "floor", "status",
        "sampling_params", "harness_version", "ts",
    ],
}


def test_schema_matches_spec(tmp_path):
    db = ResultsDB(tmp_path / "run.db")
    for table, columns in SPEC_COLUMNS.items():
        assert db.columns(table) == columns, table
    db.close()


def test_insert_round_trip_and_json_encoding(tmp_path):
    db = ResultsDB(tmp_path / "run.db")
    db.insert(
        "experiments",
        {
            "run_id": "r1",
            "model": "m",
            "backend": "vllm",
            "scenario_mix": [{"scenario": "a", "weight": 1.0}],
            "rate_levels": [0.1, 0.2],
            "warmup_s": 60,
            "measurement_s": 120,
            "created_at": "2026-06-12T00:00:00Z",
        },
    )
    assert db.count("experiments") == 1
    raw = db._conn.execute("SELECT scenario_mix, slos FROM experiments").fetchone()
    assert json.loads(raw[0]) == [{"scenario": "a", "weight": 1.0}]
    assert raw[1] is None  # absent optional column -> NULL
    db.close()


def test_unknown_column_rejected(tmp_path):
    db = ResultsDB(tmp_path / "run.db")
    try:
        db.insert("requests", {"run_id": "r1", "bogus": 1})
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "bogus" in str(exc)
    finally:
        db.close()


def test_request_rows_from_scheduler(tmp_path):
    db = ResultsDB(tmp_path / "run.db")
    rows = [
        RequestRow(
            rate_lambda=0.1, request_id=i, session_idx=i, scenario="a", turn_idx=0,
            final_turn=1, issued_at_ms=10.0 * i, ttft_ms=50.0, tpot_ms=5.0,
            e2e_ms=100.0, input_tokens=10, output_tokens=5, success=1, error=None,
            instance_id="i0",
        )
        for i in range(3)
    ]
    db.insert_request_rows("r1", rows)
    assert db.count("requests") == 3
    db.close()


def test_concurrent_writers(tmp_path):
    db = ResultsDB(tmp_path / "run.db")
    n_threads, n_rows = 8, 200

    def write(thread_idx: int):
        for i in range(n_rows):
            db.insert(
                "server_stats",
                {"run_id": "r1", "instance_id": f"t{thread_idx}", "rate_lambda": 0.1,
                 "ts": "2026-06-12T00:00:00Z", "requests_running": i},
            )

    threads = [threading.Thread(target=write, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert db.count("server_stats") == n_threads * n_rows
    db.close()


def test_smoke_mode_persists_nothing(tmp_path):
    path = tmp_path / "run.db"
    db = ResultsDB(path, persist=False)
    db.insert("experiments", {"run_id": "r1", "created_at": "x"})
    assert db.count("experiments") == 1  # write path fully exercised (§8.2)
    db.close()
    assert not path.exists()  # nothing landed on disk


def test_hardware_ndjson_ingestion_with_lambda_windows(tmp_path):
    base = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)
    samples = []
    for minute, util in [(0, 10.0), (5, 50.0), (15, 90.0)]:
        samples.append(
            {"ts": (base + timedelta(minutes=minute)).isoformat(), "node": "n1",
             "gpu_index": 0, "gpu_util_pct": util}
        )
    ndjson = tmp_path / "hw.ndjson"
    ndjson.write_text("".join(json.dumps(s) + "\n" for s in samples))

    db = ResultsDB(tmp_path / "run.db")
    windows = [
        (0.1, base - timedelta(minutes=1), base + timedelta(minutes=2)),   # covers minute 0
        (0.2, base + timedelta(minutes=4), base + timedelta(minutes=8)),   # covers minute 5
        # minute 15 falls outside every window (between-step idle) -> skipped
    ]
    ingested = db.ingest_hardware_ndjson("r1", ndjson, "i0", windows)
    assert ingested == 2
    got = db._conn.execute(
        "SELECT rate_lambda, gpu_util_pct FROM hardware_stats ORDER BY rate_lambda"
    ).fetchall()
    assert got == [(0.1, 10.0), (0.2, 50.0)]
    db.close()
