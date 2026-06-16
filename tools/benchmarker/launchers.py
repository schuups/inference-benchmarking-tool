"""Engine launchers for the Benchmarker orchestrator (M7).

The orchestrator (orchestrator.py) is launcher-agnostic — it depends only on the
EngineLauncher protocol. These concrete launchers submit the Planner-rendered
deployment artifacts from inside the Benchmarker allocation (§1: the Benchmarker
spawns the inference deployment) and resolve the endpoints the load generator
targets:

- SlurmEngineLauncher: nested `sbatch engine.sbatch`; endpoint
  http://<assigned-node>:8000; teardown via scancel (§7.4).
- K8sEngineLauncher: `kubectl apply -f engine.yaml`; endpoint = the engine's
  Ingress URL (https://ibt-engine-<slug>.<ingress_domain>) when the cluster
  declares an ingress_domain (§6.2 — reachable from the SLURM Benchmarker), else
  the in-cluster Service DNS (same-cluster only); teardown deletes the
  Deployment + Service + Ingress, leaving the model-cache PVC in place (§7.5/§7.6).
- ExternalEndpointLauncher: the engine is deployed/torn-down out-of-band (the
  laptop/Coordinator runs kubectl — the decision-5 analog for K8s, Option B); the
  SLURM Benchmarker only waits on a given endpoint URL and drives load. No kubectl,
  so it works from a SLURM allocation that cannot reach the K8s API.

v1 resolves a single instance per deployment (one engine launch == one run_id,
§16). Multi-instance deployments (data-parallel replicas, routing studies) extend
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


async def _resolve_host(nodelist: str) -> str:
    """Rank-0 host of a SLURM nodelist. Prefer `scontrol show hostnames` (it
    expands every nodelist syntax — ranges, comma lists, multi-prefix); fall
    back to the regex only if scontrol is unavailable or yields nothing."""
    code, out, _ = await _run("scontrol", "show", "hostnames", nodelist)
    if code == 0 and out.split():
        return out.split()[0]  # rank-0 node serves the endpoint (multi-node: Ray head)
    return _first_host(nodelist)


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
                return await _resolve_host(line)
            await asyncio.sleep(5.0)
            waited += 5.0
        raise RuntimeError(f"engine job {self._job_id} not assigned a node within {self._poll_timeout}s")

    def _engine_log_path(self) -> Path:
        # engine.sbatch: --output=%x-%j.out, --job-name=ibt-engine-<run_id>, --chdir=run_dir
        matches = sorted(self._run_dir.glob(f"ibt-engine-{self._run_id}-*.out"))
        return matches[-1] if matches else self._run_dir / f"ibt-engine-{self._run_id}-{self._job_id}.out"

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
    def __init__(
        self, engine_manifest: Path, namespace: str, slug: str,
        ingress_domain: str | None = None,
    ):
        # `slug` MUST be the bounded k8s_slug(run_id) the Planner rendered into the
        # manifest (object names cap at 63 chars) — not run_id_slug — so the names
        # this launcher builds (ibt-engine-<slug>, the ingress host, the kubectl
        # deployment ref) match the applied objects.
        self._manifest = Path(engine_manifest)
        self._ns = namespace
        self._slug = slug
        self._ingress_domain = ingress_domain

    async def submit(self, run_dir: Path) -> list[Instance]:
        code, out, err = await _run("kubectl", "apply", "-f", str(self._manifest))
        if code != 0:
            raise RuntimeError(f"kubectl apply failed (exit {code}): {err.strip() or out.strip()}")
        if self._ingress_domain:
            # SLURM-Benchmarker-reachable endpoint via the rendered Ingress (§6.2).
            # The cert-manager letsencrypt cert is issued ~1-2 min after apply, so the
            # endpoint 503s / fails TLS during that window — _await_ready tolerates it.
            url = f"https://ibt-engine-{self._slug}.{self._ingress_domain}"
        else:
            # In-cluster Service DNS — reachable only from a same-cluster benchmarker.
            url = f"http://ibt-engine-{self._slug}.{self._ns}.svc:{ENGINE_PORT}"
        return [Instance("i0", url, node=None)]

    def engine_log_text(self) -> str:
        r = subprocess.run(
            ["kubectl", "logs", "-n", self._ns, f"deployment/ibt-engine-{self._slug}", "--tail=500"],
            capture_output=True, text=True,
        )
        return r.stdout if r.returncode == 0 else ""

    def is_alive(self) -> bool:
        r = subprocess.run(
            ["kubectl", "get", "deployment", f"ibt-engine-{self._slug}", "-n", self._ns,
             "-o", "jsonpath={.status.replicas}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return True  # don't abort on a transient kubectl failure
        return (r.stdout.strip() or "0") != "0"

    async def teardown(self) -> None:
        # Deletes Deployment + Service + Ingress + the /tools ConfigMap (all manifest-scoped,
        # so `delete -f` reclaims them); the model-cache PVC lives outside the manifest and
        # is retained (§7.6).
        code, _, err = await _run("kubectl", "delete", "-f", str(self._manifest), "--ignore-not-found")
        if code == 0:
            log.info("deleted k8s engine objects for %s", self._slug)
        else:
            log.warning("kubectl delete failed: %s", err.strip())


class ExternalEndpointLauncher:
    """Option B: the engine is deployed and torn down out-of-band (the laptop /
    Coordinator runs `kubectl` — the decision-5 analog for K8s). The SLURM
    Benchmarker only waits on the given endpoint URL and generates load. No
    kubectl is invoked here, so this works from a SLURM allocation with no access
    to the K8s API. Liveness is owned by the external deployer; the orchestrator's
    own /health readiness poll is the signal, so is_alive() is always True."""

    def __init__(self, endpoint_url: str):
        self._url = endpoint_url.rstrip("/")

    async def submit(self, run_dir: Path) -> list[Instance]:
        log.info("using externally-managed engine endpoint %s", self._url)
        return [Instance("i0", self._url, node=None)]

    def engine_log_text(self) -> str:
        return ""  # engine logs live on the other cluster — not reachable from SLURM

    def is_alive(self) -> bool:
        return True  # external deployer owns liveness; readiness /health poll is the signal

    async def teardown(self) -> None:
        log.info("external endpoint %s — teardown owned by the deployer (no-op)", self._url)
