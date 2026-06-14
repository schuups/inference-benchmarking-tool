"""Results database schema + access (SPECIFICATIONS.md §14) — seven tables.

Canonical home for the shared §14 contract. The Benchmarker writes it (cluster
side); the Coordinator's central-DB merge and the Reports generator read it
(laptop side). It lives under `tools/common/` so every component depends on the
schema without a laptop-side module reaching into the cluster-side package.

Concurrency design (plan M3): WAL journal mode + a single-writer lock. All DB
writers live in the Benchmarker process (load gen + scrapers run on one asyncio
loop; the hardware sampler is a separate *process on the engine nodes* writing
NDJSON, ingested here at finalisation) — so a process-local lock suffices and a
threaded contention test asserts it.

Smoke-test mode (§8.2): `ResultsDB(path, persist=False)` runs the identical
write path against an in-memory database — the pipeline is exercised
end-to-end, nothing lands on disk.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

SCHEMA: dict[str, list[tuple[str, str]]] = {
    "experiments": [
        ("run_id", "TEXT PRIMARY KEY"),
        ("model", "TEXT"),
        ("backend", "TEXT"),
        ("backend_config", "TEXT"),
        ("dataset_config", "TEXT"),
        ("scenario_mix", "TEXT"),
        ("scenario_manifest", "TEXT"),
        ("slos", "TEXT"),
        ("quality_eval", "TEXT"),
        ("rate_levels", "TEXT"),
        ("warmup_s", "INTEGER"),
        ("measurement_s", "INTEGER"),
        ("created_at", "TEXT"),
    ],
    "instances": [
        ("run_id", "TEXT"),
        ("instance_id", "TEXT"),
        ("endpoint", "TEXT"),
        ("node", "TEXT"),
        ("model_load_total_s", "REAL"),
        ("model_load_weights_s", "REAL"),
        ("model_load_engine_init_s", "REAL"),
        ("model_load_cuda_graph_capture_s", "REAL"),
        ("model_load_inductor_compile_s", "REAL"),
    ],
    "requests": [
        ("run_id", "TEXT"),
        ("rate_lambda", "REAL"),
        ("request_id", "INTEGER"),
        ("session_idx", "INTEGER"),
        ("instance_id", "TEXT"),
        ("scenario", "TEXT"),
        ("turn_idx", "INTEGER"),
        ("issued_at_ms", "REAL"),
        ("final_turn", "INTEGER"),
        ("ttft_ms", "REAL"),
        ("tpot_ms", "REAL"),
        ("e2e_ms", "REAL"),
        ("input_tokens", "INTEGER"),
        ("output_tokens", "INTEGER"),
        ("success", "INTEGER"),
        ("error", "TEXT"),
    ],
    "server_stats": [
        ("run_id", "TEXT"),
        ("instance_id", "TEXT"),
        ("rate_lambda", "REAL"),
        ("ts", "TEXT"),
        ("requests_running", "INTEGER"),
        ("requests_waiting", "INTEGER"),
        ("gpu_cache_pct", "REAL"),
        ("spec_accept_rate", "REAL"),
    ],
    "hardware_stats": [
        ("run_id", "TEXT"),
        ("instance_id", "TEXT"),
        ("rate_lambda", "REAL"),
        ("ts", "TEXT"),
        ("gpu_index", "INTEGER"),
        ("gpu_util_pct", "REAL"),
        ("gpu_mem_used_gb", "REAL"),
        ("gpu_mem_pct", "REAL"),
        ("gpu_power_w", "REAL"),
        ("gpu_temp_c", "REAL"),
        ("gpu_sm_active_pct", "REAL"),
        ("gpu_tensor_active_pct", "REAL"),
        ("gpu_dram_bw_gbs", "REAL"),
        ("nvlink_rx_gbs", "REAL"),
        ("nvlink_tx_gbs", "REAL"),
        ("pcie_rx_gbs", "REAL"),
        ("pcie_tx_gbs", "REAL"),
        ("cpu_util_pct", "REAL"),
        ("cpu_iowait_pct", "REAL"),
        ("ram_used_gb", "REAL"),
        ("ram_pct", "REAL"),
        ("ram_bw_gbs", "REAL"),
        ("storage_read_gbs", "REAL"),
        ("storage_read_iops", "REAL"),
        ("net_rx_gbs", "REAL"),
        ("net_tx_gbs", "REAL"),
    ],
    "system_prechecks": [
        ("run_id", "TEXT"),
        ("instance_id", "TEXT"),
        ("metric", "TEXT"),
        ("measured", "REAL"),
        ("expected", "REAL"),
        ("tolerance_pct", "REAL"),
        ("status", "TEXT"),
        ("ts", "TEXT"),
    ],
    "quality_evals": [
        ("run_id", "TEXT"),
        ("instance_id", "TEXT"),
        ("stage", "TEXT"),
        ("suite", "TEXT"),
        ("eval_concurrency", "INTEGER"),
        ("sample_size", "INTEGER"),
        ("metric", "TEXT"),
        ("score", "REAL"),
        ("floor", "REAL"),
        ("status", "TEXT"),
        ("sampling_params", "TEXT"),
        ("harness_version", "TEXT"),
        ("ts", "TEXT"),
    ],
}

_JSON_COLUMNS = {
    "backend_config", "dataset_config", "scenario_mix", "scenario_manifest",
    "slos", "quality_eval", "rate_levels", "sampling_params",
}


def json_columns(table: str) -> tuple[str, ...]:
    """JSON-encoded columns of `table`, in schema order — the single source of
    truth for which columns to JSON-decode on read (used by the reports layer so
    it never maintains its own copy of this list)."""
    return tuple(name for name, _ in SCHEMA[table] if name in _JSON_COLUMNS)


class ResultsDB:
    def __init__(self, path: Path | str, persist: bool = True):
        self.persist = persist
        target = str(path) if persist else ":memory:"
        self._conn = sqlite3.connect(target, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            for table, columns in SCHEMA.items():
                cols = ", ".join(f"{name} {sqltype}" for name, sqltype in columns)
                self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_instances "
                "ON instances (run_id, instance_id)"
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------- writers

    def insert(self, table: str, row: dict) -> None:
        self.insert_many(table, [row])

    def insert_many(self, table: str, rows: list[dict]) -> None:
        if not rows:
            return
        columns = [name for name, _ in SCHEMA[table]]
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})"
        prepared = []
        for row in rows:
            unknown = set(row) - set(columns)
            if unknown:
                raise ValueError(f"{table}: unknown columns {sorted(unknown)}")
            prepared.append(tuple(_encode(col, row.get(col)) for col in columns))
        with self._lock:
            self._conn.executemany(sql, prepared)
            self._conn.commit()

    def insert_request_rows(self, run_id: str, rows: list) -> None:
        """Persist scheduler RequestRow dataclasses (§14.3)."""
        self.insert_many(
            "requests", [{"run_id": run_id, **asdict(r)}for r in rows]
        )

    def insert_server_stats(self, run_id: str, rows: list[dict]) -> None:
        self.insert_many("server_stats", [{"run_id": run_id, **r} for r in rows])

    def ingest_hardware_ndjson(
        self,
        run_id: str,
        ndjson_path: Path,
        instance_id: str,
        windows: list[tuple[float, datetime, datetime]],
    ) -> int:
        """Map sampler NDJSON rows onto sweep steps by timestamp (§14.5).

        `windows` = [(rate_lambda, start, end), …] covering warmup+measurement+
        drain of each step (§13.3); samples outside every window (engine
        bring-up, idle gaps) are skipped. Returns ingested row count.
        """
        hardware_columns = {name for name, _ in SCHEMA["hardware_stats"]}
        rows: list[dict] = []
        with open(ndjson_path) as f:
            for line in f:
                sample = json.loads(line)
                ts = datetime.fromisoformat(sample["ts"])
                rate = next((r for r, start, end in windows if start <= ts <= end), None)
                if rate is None:
                    continue
                row = {k: v for k, v in sample.items() if k in hardware_columns}
                row.update({"run_id": run_id, "instance_id": instance_id, "rate_lambda": rate})
                rows.append(row)
        self.insert_many("hardware_stats", rows)
        return len(rows)

    # ------------------------------------------------------------- readers

    def count(self, table: str) -> int:
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def columns(self, table: str) -> list[str]:
        with self._lock:
            return [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")]


def _encode(column: str, value):
    if value is None:
        return None
    if column in _JSON_COLUMNS and not isinstance(value, str):
        return json.dumps(value, sort_keys=True)
    return value
