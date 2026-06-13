"""Merge a per-run results DB into the centralized results DB (M8, open decision 4).

`experiments/results.db` aggregates every run; per-run `run_<id>.db` files remain
the §13.8 provenance artifacts. Merges are **idempotent by run_id** (delete-then-
insert across all seven §13 tables), so re-downloading and re-merging a run — or
resuming a Coordinator mid-collection — is safe and never double-counts.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from tools.benchmarker.db import SCHEMA, ResultsDB

log = logging.getLogger("coordinator.merge")


def merge_run_db(per_run_db: Path | str, central_db: Path | str, run_id: str) -> dict[str, int]:
    """Merge one run's rows into the central DB; return per-table inserted counts.

    Both DBs share the §13 schema (created by ResultsDB), so `INSERT … SELECT *`
    aligns column-for-column. Existing rows for `run_id` are deleted first, making
    the operation idempotent.
    """
    per_run_db, central_db = Path(per_run_db), Path(central_db)
    if not per_run_db.exists():
        raise FileNotFoundError(f"per-run DB not found: {per_run_db}")

    central_db.parent.mkdir(parents=True, exist_ok=True)  # e.g. experiments/ on first run
    ResultsDB(central_db).close()  # ensure the central DB exists with the §13 schema

    conn = sqlite3.connect(central_db)
    try:
        conn.execute("ATTACH DATABASE ? AS src", (str(per_run_db),))
        counts: dict[str, int] = {}
        for table in SCHEMA:
            conn.execute(f"DELETE FROM main.{table} WHERE run_id = ?", (run_id,))
            cur = conn.execute(
                f"INSERT INTO main.{table} SELECT * FROM src.{table} WHERE run_id = ?",
                (run_id,),
            )
            counts[table] = cur.rowcount
        conn.commit()
        conn.execute("DETACH DATABASE src")  # outside the transaction
    finally:
        conn.close()
    log.info("merged run %s into %s: %s", run_id, central_db, counts)
    return counts
