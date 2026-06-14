"""Cluster transport seam for the Coordinator (M8).

The deterministic coordinator logic (coordinator.py) depends only on the
`ClusterBackend` protocol. Per the operator's decision (open decision 5), the
**SLURM / FirecREST** effects are driven by the assistant *in-session* via the
FirecREST MCP tools — there is intentionally no autonomous SLURM backend here, so
SLURM orchestration runs interactively. The **K8s** path uses `kubectl` and is
headless-capable (`KubectlClusterBackend`), though its staging/PVC wiring is an
E5 deliverable. `FakeClusterBackend` exercises the whole orchestration in tests,
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

from tools.common.proc import kubectl as _kubectl

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
        """Find the inference-deployment resources to tear down (§6.1 labels)."""

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


# ----------------------------------------------------------------- kubernetes


_K8S_PHASE = {"Pending": "pending", "Running": "running", "Succeeded": "completed", "Failed": "failed"}


class KubectlClusterBackend:
    """Headless K8s backend (breithorn, E5). Well-defined ops are implemented;
    staging (results PVC + config injection) is wired with E5 — see TODOs.md."""

    platform = "k8s"

    def __init__(self, namespace: str, run_id_slug: str, manifest_dir: Path):
        self._ns = namespace
        self._slug = run_id_slug
        self._dir = Path(manifest_dir)

    async def stage(self, local_dir: Path, run_dir_remote: str) -> None:
        raise NotImplementedError(
            "K8s staging (results PVC + benchmark_config injection) lands with E5 — see TODOs.md"
        )

    async def submit(self, run_dir_remote: str, script: str) -> str:
        code, out, err = await _kubectl("apply", "-f", str(self._dir / script))
        if code != 0:
            raise RuntimeError(f"kubectl apply {script} failed: {err.strip() or out.strip()}")
        return f"pod/ib-benchmarker-{self._slug}"

    async def status(self, handle: str) -> str:
        code, out, _ = await _kubectl(
            "get", handle, "-n", self._ns, "-o", "jsonpath={.status.phase}"
        )
        if code != 0:
            return "pending"  # not yet visible; don't treat a transient miss as failure
        return _K8S_PHASE.get(out.strip(), "running")

    async def read_remote(self, remote_path: str) -> str | None:
        # results.json lives on the results PVC; read it via the benchmarker pod.
        code, out, _ = await _kubectl(
            "exec", "-n", self._ns, f"ib-benchmarker-{self._slug}", "--", "cat", remote_path
        )
        return out if code == 0 else None

    async def discover_engine_handles(self, run_id: str) -> list[str]:
        return [f"deployment/ib-engine-{self._slug}", f"service/ib-engine-{self._slug}"]

    async def fetch_db(self, remote_db: str, local_db: Path) -> str:
        local_db = Path(local_db)
        local_db.parent.mkdir(parents=True, exist_ok=True)
        code, _, err = await _kubectl(
            "cp", f"{self._ns}/ib-benchmarker-{self._slug}:{remote_db}", str(local_db)
        )
        if code != 0:
            raise RuntimeError(f"kubectl cp of per-run DB failed: {err.strip()}")
        return _sha256(local_db)

    async def cancel(self, handle: str) -> None:
        code, _, err = await _kubectl("delete", handle, "-n", self._ns, "--ignore-not-found")
        if code != 0:
            log.warning("kubectl delete %s failed: %s", handle, err.strip())

    async def remove_dir(self, remote_dir: str) -> None:
        # K8s scratch is the results PVC; the model-cache PVC is retained (§6.6).
        # The results PVC is reclaimed with the run's objects at E5; nothing to rm here.
        return None
