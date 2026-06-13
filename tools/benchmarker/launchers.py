"""Engine launchers for the Benchmarker orchestrator (M7).

The orchestrator (orchestrator.py) is launcher-agnostic — it depends only on the
EngineLauncher protocol. These concrete launchers submit the Planner-rendered
deployment artifacts from inside the Benchmarker allocation (§1: the Benchmarker
spawns the inference deployment) and resolve the endpoints the load generator
targets:

- SlurmEngineLauncher: nested `sbatch engine.sbatch`; endpoint
  http://<assigned-node>:8000; teardown via scancel (§6.4).
- K8sEngineLauncher: `kubectl apply -f engine.yaml`; endpoint = the in-cluster
  Service DNS name; teardown deletes the Deployment + Service, leaving the
  model-cache PVC in place (§6.5/§6.6).

v1 resolves a single instance per deployment (one engine launch == one run_id,
§15). Multi-instance deployments (data-parallel replicas, routing studies) extend
submit() to return more than one Instance.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess

from pathlib import Path

from .orchestrator import Instance

log = logging.getLogger("benchmarker.launcher")

ENGINE_PORT = 8000
_TERMINAL_SLURM_STATES = {
    "FAILED", "CANCELLED", "TIMEOUT", "COMPLETED", "NODE_FAIL", "OUT_OF_MEMORY",
    "BOOT_FAIL", "DEADLINE", "PREEMPTED",
}


async def _run(*args: str, cwd: Path | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _first_host(nodelist: str) -> str:
    """First host of a SLURM nodelist: 'nid[007660-007663]' → 'nid007660'."""
    m = re.match(r"(.+?)\[(\d+)", nodelist)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return nodelist.split(",")[0].strip()


class SlurmEngineLauncher:
    def __init__(
        self, engine_sbatch: Path, run_dir: Path, run_id: str, *, node_poll_timeout_s: float = 600.0
    ):
        self._sbatch = Path(engine_sbatch)
        self._run_dir = Path(run_dir)
        self._run_id = run_id
        self._poll_timeout = node_poll_timeout_s
        self._job_id: str | None = None

    async def submit(self, run_dir: Path) -> list[Instance]:
        code, out, err = await _run("sbatch", "--parsable", str(self._sbatch), cwd=self._run_dir)
        if code != 0:
            raise RuntimeError(f"sbatch failed (exit {code}): {err.strip() or out.strip()}")
        self._job_id = out.strip().split(";")[0]  # --parsable → "<jobid>[;cluster]"
        log.info("submitted engine job %s", self._job_id)
        node = await self._await_node()
        return [Instance("i0", f"http://{node}:{ENGINE_PORT}", node=node)]

    async def _await_node(self) -> str:
        waited = 0.0
        while waited < self._poll_timeout:
            code, out, _ = await _run("squeue", "-j", self._job_id, "-h", "-o", "%N")
            line = out.strip().splitlines()[0].strip() if out.strip() else ""
            if line and line != "(null)":
                return _first_host(line)
            await asyncio.sleep(5.0)
            waited += 5.0
        raise RuntimeError(f"engine job {self._job_id} not assigned a node within {self._poll_timeout}s")

    def _engine_log_path(self) -> Path:
        # engine.sbatch: --output=%x-%j.out, --job-name=ib-engine-<run_id>, --chdir=run_dir
        matches = sorted(self._run_dir.glob(f"ib-engine-{self._run_id}-*.out"))
        return matches[-1] if matches else self._run_dir / f"ib-engine-{self._run_id}-{self._job_id}.out"

    def engine_log_text(self) -> str:
        path = self._engine_log_path()
        return path.read_text(errors="replace") if path.exists() else ""

    def is_alive(self) -> bool:
        if self._job_id is None:
            return True
        r = subprocess.run(
            ["squeue", "-j", self._job_id, "-h", "-o", "%T"], capture_output=True, text=True
        )
        if r.returncode != 0:
            return True  # transient squeue failure — don't abort the run on a CLI hiccup
        state = r.stdout.strip().upper()
        if not state:
            return False  # job left the queue before readiness → ended/crashed
        return state not in _TERMINAL_SLURM_STATES

    async def teardown(self) -> None:
        if not self._job_id:
            return
        code, _, err = await _run("scancel", self._job_id)
        if code == 0:
            log.info("scancelled engine job %s", self._job_id)
        else:
            log.warning("scancel %s failed: %s", self._job_id, err.strip())


class K8sEngineLauncher:
    def __init__(self, engine_manifest: Path, namespace: str, run_id_slug: str):
        self._manifest = Path(engine_manifest)
        self._ns = namespace
        self._slug = run_id_slug

    async def submit(self, run_dir: Path) -> list[Instance]:
        code, out, err = await _run("kubectl", "apply", "-f", str(self._manifest))
        if code != 0:
            raise RuntimeError(f"kubectl apply failed (exit {code}): {err.strip() or out.strip()}")
        # In-cluster Service DNS; the engine's startupProbe gates model-load wait.
        host = f"ib-engine-{self._slug}.{self._ns}.svc"
        return [Instance("i0", f"http://{host}:{ENGINE_PORT}", node=None)]

    def engine_log_text(self) -> str:
        r = subprocess.run(
            ["kubectl", "logs", "-n", self._ns, f"deployment/ib-engine-{self._slug}", "--tail=500"],
            capture_output=True, text=True,
        )
        return r.stdout if r.returncode == 0 else ""

    def is_alive(self) -> bool:
        r = subprocess.run(
            ["kubectl", "get", "deployment", f"ib-engine-{self._slug}", "-n", self._ns,
             "-o", "jsonpath={.status.replicas}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return True  # don't abort on a transient kubectl failure
        return (r.stdout.strip() or "0") != "0"

    async def teardown(self) -> None:
        # Deletes Deployment + Service only; the model-cache PVC is retained (§6.6).
        code, _, err = await _run("kubectl", "delete", "-f", str(self._manifest), "--ignore-not-found")
        if code == 0:
            log.info("deleted k8s engine objects for %s", self._slug)
        else:
            log.warning("kubectl delete failed: %s", err.strip())
