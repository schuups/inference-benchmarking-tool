"""Cluster transport seam for the Coordinator (M8).

The deterministic coordinator logic (coordinator.py) depends only on the
`ClusterBackend` protocol. The
Benchmarker is **always a SLURM allocation** (§2), so the only real backend is
SLURM / FirecREST — and per the operator's decision (open decision 5) those effects
are driven by the assistant *in-session* via the FirecREST MCP tools, so there is no
autonomous backend here. There is **no K8s Coordinator backend**: a K8s *engine* target
is deployed by the Benchmarker itself (`tools.benchmarker.launchers.K8sEngineLauncher`)
from inside its SLURM allocation, and orphaned K8s objects are reclaimed by the Cleaner
(§7.7). `FakeClusterBackend` exercises the whole orchestration in tests,
including a real local compress→transfer→extract→checksum so the staged-download
round-trip (M8 DoD) is verified bit-for-bit.

Status is normalised to: "pending" | "running" | "completed" | "failed".
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Protocol


log = logging.getLogger("coordinator.backend")

_DONE = {"completed", "failed"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ClusterBackend(Protocol):
    platform: str

    async def stage(self, local_dir: Path, run_dir_remote: str) -> None:
        """Place the planner-rendered run artifacts onto cluster scratch."""

    async def submit(self, run_dir_remote: str, script: str) -> str:
        """Submit the Benchmarker job; return its handle (job id / pod name)."""

    async def status(self, handle: str) -> str:
        """Normalised job status: pending | running | completed | failed."""

    async def read_remote(self, remote_path: str) -> str | None:
        """Read a small remote text file (e.g. prechecks/results.json); None if absent."""

    async def discover_engine_handles(self, run_id: str) -> list[str]:
        """Find the inference-deployment resources to tear down (§7.1 labels)."""

    async def fetch_db(self, remote_db: str, local_db: Path) -> str:
        """Staged transfer of the per-run DB; return its sha256."""

    async def cancel(self, handle: str) -> None:
        """Cancel/delete a job/resource by handle (idempotent)."""

    async def remove_dir(self, remote_dir: str) -> None:
        """Remove a remote scratch directory (idempotent)."""


# --------------------------------------------------------------------- testing


class FakeClusterBackend:
    """In-process backend: 'remote' paths are real local paths under a tmp dir,
    so the full orchestration (incl. a genuine staged-download round-trip) runs
    without a cluster. `submit` plants the precheck results + per-run DB that the
    Benchmarker (M7) would produce remotely."""

    platform = "fake"

    def __init__(
        self,
        *,
        result_db: Path | None = None,
        precheck: dict | None = None,
        status_script: list[str] | None = None,
        engine_handles: list[str] | None = None,
    ):
        self._result_db = Path(result_db) if result_db else None
        self._precheck = precheck
        self._status_script = list(status_script or ["completed"])
        self._engine_handles = list(engine_handles or [])
        self.calls: list[tuple] = []

    async def stage(self, local_dir: Path, run_dir_remote: str) -> None:
        self.calls.append(("stage", run_dir_remote))
        Path(run_dir_remote).mkdir(parents=True, exist_ok=True)

    async def submit(self, run_dir_remote: str, script: str) -> str:
        self.calls.append(("submit", run_dir_remote, script))
        rd = Path(run_dir_remote)
        rd.mkdir(parents=True, exist_ok=True)
        run_id = rd.name
        if self._precheck is not None:
            (rd / "prechecks").mkdir(exist_ok=True)
            (rd / "prechecks" / "results.json").write_text(json.dumps(self._precheck))
        if self._result_db is not None:  # simulate the Benchmarker's per-run DB
            shutil.copy(self._result_db, rd / f"run_{run_id}.db")
        return "job-fake-1"

    async def status(self, handle: str) -> str:
        self.calls.append(("status", handle))
        return self._status_script[0] if len(self._status_script) == 1 else self._status_script.pop(0)

    async def read_remote(self, remote_path: str) -> str | None:
        p = Path(remote_path)
        return p.read_text() if p.exists() else None

    async def discover_engine_handles(self, run_id: str) -> list[str]:
        return list(self._engine_handles)

    async def fetch_db(self, remote_db: str, local_db: Path) -> str:
        """Real staged round-trip: gzip the remote DB, copy, gunzip locally, checksum."""
        self.calls.append(("fetch_db", remote_db))
        src = Path(remote_db)
        local_db = Path(local_db)
        local_db.parent.mkdir(parents=True, exist_ok=True)
        remote_gz = src.with_suffix(src.suffix + ".gz")
        with open(src, "rb") as f_in, gzip.open(remote_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        local_gz = local_db.with_suffix(local_db.suffix + ".gz")
        shutil.copy(remote_gz, local_gz)  # the "transfer"
        with gzip.open(local_gz, "rb") as f_in, open(local_db, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        local_gz.unlink()
        return _sha256(local_db)

    async def cancel(self, handle: str) -> None:
        self.calls.append(("cancel", handle))

    async def remove_dir(self, remote_dir: str) -> None:
        self.calls.append(("remove_dir", remote_dir))
        d = Path(remote_dir)
        if d.exists():
            shutil.rmtree(d)

