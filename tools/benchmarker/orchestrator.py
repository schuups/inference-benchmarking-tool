"""Benchmarker orchestrator (SPECIFICATIONS.md §1, IMPLEMENTATION_PLAN.md M7).

The cluster-side driver that owns the §1 phase sequencing inside the Benchmarker
allocation:

    dataset generation
      → submit the inference deployment(s) ONLY after the prompt pool exists
      → wait for readiness + inductor primer (§10.3, §12.1)
      → Stage-A quality gate (§13.5, via the M11 QualityEvaluator seam)
      → rate-level sweep (§12.2)
      → Stage-B quality comparison (§13.5, via the seam)
      → finalise the per-run DB (ingest sampler NDJSON into hardware_stats, §14.5)

Smoke-test-mode propagation (§8.2): a pre-check cache miss flips the run into
smoke mode — the pipeline runs end-to-end but nothing is persisted, with an
unmissable warning at launch and at termination.

Pre-check gate (§8.4): the engine container enforces the gate inline
(`run_system_prechecks.sh && exec <engine>`), so an aborting gate means the
engine never starts. This orchestrator detects that from prechecks/results.json
instead of waiting out the full readiness timeout. The interactive operator
*pause* on a warn belongs to the laptop Coordinator (M8) watching this run;
non-interactive cluster execution follows the config's on_warn/on_fail policy,
which grade.py already applied via its exit code.

Engine submission, endpoint discovery, and teardown are delegated to an injected
EngineLauncher (launchers.py), so the whole sequence is exercised in-process
against a mock server with no cluster (M7 DoD).
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import aiohttp

from tools.common.config import BenchmarkConfig, Deployment

from tools.common.results_db import ResultsDB
from .dataset_gen.generator import POOL_FILENAME, generate
from .dataset_gen.tokenizers import Tokenizer
from .load_gen.pool import load_pool
from .load_gen.readiness import parse_model_load, run_primer
from .load_gen.scheduler import StepConfig, run_step

log = logging.getLogger("benchmarker")


# ----------------------------------------------------------------- data types


@dataclass(frozen=True)
class Instance:
    """One deployed engine endpoint the load generator targets (§14.2)."""

    instance_id: str
    base_url: str  # e.g. http://nid007660:8000 — no trailing slash, no path
    node: str | None = None


@dataclass
class GateOutcome:
    """Stage-A result from a QualityEvaluator: quality_evals rows + pass/fail."""

    rows: list[dict]
    passed: bool


@dataclass
class RunSummary:
    run_id: str
    persisted: bool
    smoke_test_mode: bool
    instances: int
    requests: int
    rate_levels: int
    sessions_truncated: int
    quality_flagged: bool
    db_path: Path | None


class RunAborted(RuntimeError):
    """Terminal, operator-facing abort (pre-check gate, quality gate, dead engine)."""


class _PrecheckAbort(RuntimeError):
    """Internal: the in-container §8.4 gate aborted; engine will not start."""

    def __init__(self, gate_exit_code: int):
        super().__init__(f"pre-check gate exit {gate_exit_code}")
        self.gate_exit_code = gate_exit_code


class EngineLauncher(Protocol):
    """Platform seam (SLURM / K8s / mock) — see launchers.py."""

    async def submit(self, run_dir: Path) -> list[Instance]:
        """Spawn the engine deployment; return its instances once endpoints resolve."""

    def engine_log_text(self) -> str:
        """Engine stdout/stderr — parsed for the §10.2 model-load breakdown."""

    def is_alive(self) -> bool:
        """False once the engine job has exited (crash detection during readiness)."""

    async def teardown(self) -> None:
        """Cancel/delete the spawned engine (§7.4/§7.5); idempotent."""


class QualityEvaluator(Protocol):
    """M11 seam (§13.5). Wired in main.py when M11 lands; None until then."""

    async def stage_a_gate(self, instances: list[Instance], model: str, gate) -> GateOutcome:
        ...

    async def stage_b_compare(self, instances: list[Instance], model: str, compare) -> list[dict]:
        ...


# --------------------------------------------------------------------- helpers


def _tail(text: str, n: int = 40) -> str:
    return "\n".join(text.splitlines()[-n:])


def _read_precheck(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _step_config(
    cfg: BenchmarkConfig, deployment: Deployment, rate: float, endpoints: list[tuple[str, str]]
) -> StepConfig:
    return StepConfig(
        rate_lambda=rate,
        warmup_s=cfg.phases.warmup_s,
        measurement_s=cfg.phases.measurement_s,
        drain_timeout_s=cfg.phases.drain_timeout_s,
        request_timeout_s=cfg.phases.request_timeout_s,
        arrival=cfg.arrival_process,
        routing=cfg.routing_strategy,
        output_length_mode=cfg.dataset_config.output_length_mode,
        seed=cfg.dataset_config.seed,
        endpoints=endpoints,
        model=deployment.model,
    )


def _persist_experiment(
    db: ResultsDB, run_id: str, cfg: BenchmarkConfig, deployment: Deployment, manifest: dict
) -> None:
    qe = cfg.quality_eval
    # §14.1: quality_eval is NULL when both stages are disabled.
    quality_eval = None if (qe.skip_quality_gate and qe.skip_quality_compare) else qe.model_dump()
    db.insert(
        "experiments",
        {
            "run_id": run_id,
            "model": deployment.model,
            "backend": deployment.backend,
            "backend_config": deployment.backend_config.model_dump(),
            "dataset_config": cfg.dataset_config.model_dump(),
            "scenario_mix": [
                {"scenario": e.scenario, "weight": e.weight}
                for e in cfg.dataset_config.scenario_mix
            ],
            "scenario_manifest": manifest,
            "slos": [s.model_dump() for s in cfg.slos] if cfg.slos else None,
            "quality_eval": quality_eval,
            "rate_levels": cfg.rate_levels,
            "warmup_s": cfg.phases.warmup_s,
            "measurement_s": cfg.phases.measurement_s,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _persist_prechecks(db: ResultsDB, run_id: str, instance_id: str, precheck: dict) -> None:
    rows = precheck.get("rows") or []
    if not rows:
        return
    db.insert_many(
        "system_prechecks",
        [{"run_id": run_id, "instance_id": instance_id, **r} for r in rows],
    )
    for r in rows:
        level = {"fail": logging.ERROR, "warn": logging.WARNING}.get(r["status"], logging.INFO)
        log.log(
            level, "[prechecks] %-4s %s measured=%s expected=%s",
            r["status"], r["metric"], r["measured"], r["expected"],
        )


def _ingest_hardware(
    db: ResultsDB,
    run_id: str,
    run_dir: Path,
    instances: list[Instance],
    windows: list[tuple[float, datetime, datetime]],
) -> int:
    """Map each per-node sampler file (hw-<host>.ndjson) onto its instance (§14.5)."""
    by_node = {i.node: i for i in instances if i.node}
    default_id = instances[0].instance_id if instances else "i0"
    total = 0
    for ndjson in sorted(run_dir.glob("hw-*.ndjson")):
        host = ndjson.stem.removeprefix("hw-")
        inst = by_node.get(host)
        instance_id = inst.instance_id if inst else default_id
        total += db.ingest_hardware_ndjson(run_id, ndjson, instance_id, windows)
    return total


DEAD_ENGINE_GRACE_S = 10.0  # let Lustre surface results.json after a §8.4 abort


async def _await_ready(
    http: aiohttp.ClientSession,
    base_url: str,
    results_path: Path,
    timeout_s: float,
    launcher: EngineLauncher,
    poll_s: float = 2.0,
) -> float:
    """Wait for /health 200; abort early on an in-container gate abort or a dead job.

    Tolerates the K8s Ingress + cert-manager TLS provisioning window (§6.2): for
    ~2-3 min after `kubectl apply` the endpoint refuses connections, fails TLS
    verification (cert not yet issued), or 50x's from nginx — all expected and
    retried until `timeout_s` (server_ready_timeout_s, default 3600 ≫ the window).

    Returns seconds waited → the instance's model_load_total_s input (§10.2; this
    coarse total covers scheduling + pre-checks + load, with the precise
    breakdown carried by the parsed model_load_* components).
    """
    start = time.perf_counter()
    last_note = 0.0
    last_probe = "no response yet"
    while True:
        gate_code = _read_precheck(results_path).get("gate_exit_code", 0)
        if gate_code != 0:
            raise _PrecheckAbort(gate_code)
        try:
            async with http.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    return time.perf_counter() - start
                last_probe = f"HTTP {resp.status}"  # 404/502/503 while ingress+backend wire up
        except (aiohttp.ClientError, ssl.SSLError, asyncio.TimeoutError, OSError) as exc:
            # Connection refused / TLS-cert-not-yet-valid while the Ingress and its
            # cert-manager letsencrypt cert provision — expected; keep polling.
            last_probe = type(exc).__name__
        if not launcher.is_alive():
            # The §8.4 gate may have aborted the engine job *after* writing its
            # results.json (run_system_prechecks.sh && exec <engine>): the job
            # exits before /health ever comes up. Give the shared filesystem
            # (Lustre) a moment to surface the file, then disambiguate a clean
            # gate abort from a genuine engine crash.
            await asyncio.sleep(DEAD_ENGINE_GRACE_S)
            gate_code = _read_precheck(results_path).get("gate_exit_code", 0)
            if gate_code != 0:
                raise _PrecheckAbort(gate_code)
            raise RunAborted(
                "engine job exited before readiness — last engine log lines:\n"
                + _tail(launcher.engine_log_text())
            )
        elapsed = time.perf_counter() - start
        if elapsed > timeout_s:
            raise RunAborted(
                f"engine at {base_url} not ready within server_ready_timeout_s={timeout_s} "
                f"(§10.1/§12.1); last probe: {last_probe}"
            )
        if elapsed - last_note >= 30:  # heartbeat so a long ingress/cert wait isn't mistaken for a hang
            log.info(
                "waiting for engine at %s — %.0fs elapsed, last probe: %s "
                "(a K8s ingress + letsencrypt cert can take ~2-3 min)",
                base_url, elapsed, last_probe,
            )
            last_note = elapsed
        await asyncio.sleep(poll_s)


async def _stage_a(
    db: ResultsDB,
    run_id: str,
    quality: QualityEvaluator | None,
    qe,
    instances: list[Instance],
    model: str,
    smoke: bool,
) -> bool:
    """Stage-A sanity gate (§13.5). Returns quality_flagged; raises on fail+abort."""
    if qe.skip_quality_gate:
        log.info("Stage-A quality gate skipped (skip_quality_gate, §13.5)")
        return False
    if quality is None:
        log.info("Stage-A quality gate: no evaluator wired (M11 pending) — skipped")
        return False
    outcome = await quality.stage_a_gate(instances, model, qe.gate)
    if not smoke and outcome.rows:
        db.insert_many("quality_evals", [{"run_id": run_id, **r} for r in outcome.rows])
    if outcome.passed:
        log.info("Stage-A quality gate PASSED (§13.5)")
        return False
    if qe.gate.on_fail == "abort":
        raise RunAborted(
            "Stage-A quality gate FAILED (§13.5) — aborting before the sweep; "
            "results quality-flagged"
        )
    log.warning(
        "Stage-A quality gate FAILED but on_fail=continue — run is QUALITY-FLAGGED (§13.5, §15.1)"
    )
    return True


async def _stage_b(
    db: ResultsDB,
    run_id: str,
    quality: QualityEvaluator | None,
    qe,
    instances: list[Instance],
    model: str,
    smoke: bool,
) -> None:
    """Stage-B quality comparison (§13.5) — measurement, not a gate."""
    if qe.skip_quality_compare:
        log.info("Stage-B quality comparison skipped (skip_quality_compare, §13.5)")
        return
    if quality is None:
        log.info("Stage-B quality comparison: no evaluator wired (M11 pending) — skipped")
        return
    rows = await quality.stage_b_compare(instances, model, qe.compare)
    if not smoke and rows:
        db.insert_many("quality_evals", [{"run_id": run_id, **r} for r in rows])
    log.info("Stage-B quality comparison: %d measurement rows (§13.5)", len(rows))


# ------------------------------------------------------------------- the driver


async def run_experiment(
    cfg: BenchmarkConfig,
    deployment: Deployment,
    run_id: str,
    run_dir: Path,
    tokenizer: Tokenizer,
    launcher: EngineLauncher,
    registry_dir: Path,
    quality: QualityEvaluator | None = None,
) -> RunSummary:
    run_dir = Path(run_dir)
    qe = cfg.quality_eval

    # ---- phase 1: dataset generation. The pool MUST exist before the engine is
    # spawned (§1 — no GPU idling during prompt prep), asserted by M7's tests.
    dataset_dir = run_dir / "dataset"
    log.info("[%s] generating dataset pool → %s", run_id, dataset_dir)
    manifest = generate(cfg, tokenizer, dataset_dir, registry_dir)
    pool_path = dataset_dir / POOL_FILENAME
    if not pool_path.exists():
        raise RunAborted(f"dataset pool {pool_path} missing after generation")
    pool = load_pool(pool_path)
    weights = {m["scenario"]: m["weight"] for m in manifest["mix"]}  # session-start weights (§12.3)

    db: ResultsDB | None = None
    smoke = False
    quality_flagged = False
    results_path = run_dir / "prechecks" / "results.json"
    db_path = run_dir / f"run_{run_id}.db"
    summary: RunSummary | None = None
    try:
        # ---- phase 2: spawn the inference deployment (pool ready → ordering held)
        log.info("[%s] submitting engine deployment", run_id)
        instances = await launcher.submit(run_dir)
        if not instances:
            raise RunAborted("engine launcher returned no instances")

        async with aiohttp.ClientSession() as http:
            # ---- phase 3: readiness, with in-container §8.4 gate-abort detection
            try:
                waits = await asyncio.gather(
                    *[
                        _await_ready(
                            http, inst.base_url, results_path,
                            cfg.phases.server_ready_timeout_s, launcher,
                        )
                        for inst in instances
                    ]
                )
            except _PrecheckAbort as abort:
                # §8.4: persist what the gate measured, then stop — the engine
                # never started so there is no sweep to run.
                precheck = _read_precheck(results_path)
                smoke = bool(precheck.get("smoke_test_mode", False))
                db = ResultsDB(db_path, persist=not smoke)
                _persist_experiment(db, run_id, cfg, deployment, manifest)
                _persist_prechecks(db, run_id, instances[0].instance_id, precheck)
                offending = [
                    f"{r['metric']}={r['measured']}"
                    for r in precheck.get("rows", [])
                    if r["status"] in ("warn", "fail")
                ]
                raise RunAborted(
                    f"system pre-check gate aborted (exit {abort.gate_exit_code}, §8.4) — "
                    f"engine not started. Offending: {', '.join(offending) or 'see system_prechecks'}"
                ) from abort

            # ---- phase 4: pre-check outcome → smoke mode → open DB → rows
            precheck = _read_precheck(results_path)
            smoke = bool(precheck.get("smoke_test_mode", False))
            if smoke:
                log.warning(
                    "[%s] SMOKE-TEST MODE (§8.2): collective-tests cache was cold — "
                    "results will NOT be persisted",
                    run_id,
                )
            db = ResultsDB(db_path, persist=not smoke)
            _persist_experiment(db, run_id, cfg, deployment, manifest)
            model_load = parse_model_load(launcher.engine_log_text())
            for inst, waited in zip(instances, waits):
                db.insert(
                    "instances",
                    {
                        "run_id": run_id,
                        "instance_id": inst.instance_id,
                        "endpoint": inst.base_url,
                        "node": inst.node,
                        "model_load_total_s": waited,
                        **model_load,
                    },
                )
            _persist_prechecks(db, run_id, instances[0].instance_id, precheck)

            # ---- phase 5: inductor pre-compilation primer (§10.3)
            primers = await asyncio.gather(
                *[run_primer(http, inst.base_url, model=deployment.model) for inst in instances]
            )
            for inst, pr in zip(instances, primers):
                if pr.warning:
                    log.warning("[%s] instance %s primer: %s", run_id, inst.instance_id, pr.warning)

            # ---- phase 6: Stage-A quality gate (§13.5)
            quality_flagged = await _stage_a(db, run_id, quality, qe, instances, deployment.model, smoke)

            # ---- phase 7: rate-level sweep (§12.2)
            endpoints = [(i.instance_id, i.base_url) for i in instances]
            windows: list[tuple[float, datetime, datetime]] = []
            n_requests = n_truncated = 0
            for rate in cfg.rate_levels:
                log.info("[%s] sweep step λ=%s", run_id, rate)
                start = datetime.now(timezone.utc)
                step = await run_step(pool, weights, _step_config(cfg, deployment, rate, endpoints))
                end = datetime.now(timezone.utc)
                db.insert_request_rows(run_id, step.requests)
                db.insert_server_stats(run_id, step.server_stats)
                windows.append((rate, start, end))
                n_requests += len(step.requests)
                n_truncated += step.sessions_truncated
                if step.lag_warning:
                    log.warning(
                        "[%s] λ=%s client event-loop lag %.0fms — latencies may be client "
                        "artefacts (shard the load generator)",
                        run_id, rate, step.lag_max_ms,
                    )
                if step.sessions_truncated:
                    log.info(
                        "[%s] λ=%s %d sessions truncated at drain (§12.2)",
                        run_id, rate, step.sessions_truncated,
                    )

            # ---- phase 8: Stage-B quality comparison (§13.5)
            await _stage_b(db, run_id, quality, qe, instances, deployment.model, smoke)

            # ---- phase 9: finalise — sampler NDJSON → hardware_stats (§14.5)
            ingested = _ingest_hardware(db, run_id, run_dir, instances, windows)
            log.info(
                "[%s] finalised: %d requests, %d hardware_stats rows",
                run_id, n_requests, ingested,
            )

            summary = RunSummary(
                run_id=run_id,
                persisted=not smoke,
                smoke_test_mode=smoke,
                instances=len(instances),
                requests=n_requests,
                rate_levels=len(cfg.rate_levels),
                sessions_truncated=n_truncated,
                quality_flagged=quality_flagged,
                db_path=db_path if not smoke else None,
            )
    finally:
        if db is not None:
            db.close()
        await launcher.teardown()
        if smoke:
            log.warning(
                "[%s] SMOKE-TEST MODE (§8.2): run completed but NOTHING was persisted — "
                "re-run on a warm collective-tests cache",
                run_id,
            )
    assert summary is not None  # only reached on the success path
    return summary
