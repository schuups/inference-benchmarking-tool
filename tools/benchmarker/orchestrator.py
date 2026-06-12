"""Benchmarker orchestrator (SPECIFICATIONS.md §1; IMPLEMENTATION_PLAN.md M7).

Cluster-side sequencing of one run (= one deployment config):

    dataset generation                     (GPUs not yet allocated, §1)
      -> engine launch                     (only after the pool exists)
      -> §7 pre-check gate + smoke mode    (results.json from the engine job)
      -> readiness + model-load + primer   (§11.1, §9.2, §9.3)
      -> [Stage-A quality gate]            (§12.5 — M11 hook, stubbed)
      -> λ sweep                           (§11.2/§11.3, per-step windows recorded)
      -> [Stage-B quality compare]         (§12.5 — M11 hook, stubbed)
      -> finalisation                      (sampler NDJSON ingestion, §13 rows)
      -> teardown                          (success and failure paths, §6)

The engine is launched through an injectable EngineLauncher: the SLURM
implementation submits the planner-rendered engine.sbatch from within the
Benchmarker allocation (validated at E1); the mock launcher backs the laptop
integration tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Protocol

import aiohttp

from tools.common.config import BenchmarkConfig, Deployment
from tools.benchmarker.db import ResultsDB
from tools.benchmarker.dataset_gen.generator import POOL_FILENAME, generate
from tools.benchmarker.dataset_gen.tokenizers import Tokenizer
from tools.benchmarker.load_gen.pool import load_pool
from tools.benchmarker.load_gen.readiness import parse_model_load, run_primer, wait_ready
from tools.benchmarker.load_gen.scheduler import StepConfig, run_step

log = logging.getLogger("benchmarker")


class RunAborted(RuntimeError):
    """Run aborted by a gate (§7.4) or a phase failure; teardown still runs."""


class EngineLauncher(Protocol):
    async def launch(self, run_dir: Path) -> list[tuple[str, str]]:
        """Submit/start the inference deployment; returns (instance_id, url)s."""
        ...

    async def teardown(self) -> None: ...

    def engine_log_text(self) -> str | None:
        """Engine log content for §9.2 model-load parsing; None if unavailable."""
        ...

    def is_alive(self) -> bool:
        """False once the engine deployment is known dead (fast-fail waits)."""
        ...


@dataclass
class SlurmEngineLauncher:
    """Submits the planner-rendered engine.sbatch from within the Benchmarker
    allocation (§1). Cluster half of M7 — validated at E1."""

    engine_sbatch: Path
    port: int = 8000
    job_id: str | None = None

    async def launch(self, run_dir: Path) -> list[tuple[str, str]]:
        proc = subprocess.run(
            ["sbatch", "--parsable", str(self.engine_sbatch)],
            capture_output=True, text=True, check=True, cwd=run_dir,
        )
        self.job_id = proc.stdout.strip().split(";")[0]
        log.info("engine job %s submitted", self.job_id)
        for _ in range(720):  # up to ~1h queue wait
            await asyncio.sleep(5)
            show = subprocess.run(
                ["squeue", "-j", self.job_id, "-h", "-o", "%T %N"],
                capture_output=True, text=True,
            )
            state, _, nodelist = show.stdout.strip().partition(" ")
            if state == "RUNNING" and nodelist:
                nodes = subprocess.run(
                    ["scontrol", "show", "hostnames", nodelist],
                    capture_output=True, text=True,
                ).stdout.split()
                # rank-0 node serves the endpoint (multi-node: Ray head, §M6)
                return [("i0", f"http://{nodes[0]}:{self.port}")]
            if state in ("FAILED", "CANCELLED", "TIMEOUT", "COMPLETED"):
                raise RunAborted(f"engine job {self.job_id} reached {state} before serving")
        raise RunAborted(f"engine job {self.job_id} did not start in time")

    async def teardown(self) -> None:
        if self.job_id:
            subprocess.run(["scancel", self.job_id], check=False)

    def engine_log_text(self) -> str | None:
        matches = sorted(Path.cwd().glob(f"ib-engine-*-{self.job_id}.out"))
        return matches[-1].read_text() if matches else None

    def is_alive(self) -> bool:
        if not self.job_id:
            return False
        out = subprocess.run(
            ["squeue", "-j", self.job_id, "-h", "-o", "%T"],
            capture_output=True, text=True,
        ).stdout.strip()
        return out in ("RUNNING", "COMPLETING")


QualityHook = Callable[[list[tuple[str, str]], ResultsDB, str], Awaitable[None]]


@dataclass
class RunSummary:
    run_id: str
    persisted: bool
    smoke_test_mode: bool = False
    sessions_started: int = 0
    sessions_truncated: int = 0
    requests: int = 0
    lag_warning: bool = False
    primer_warnings: list[str] = field(default_factory=list)
    quality_stages_pending: list[str] = field(default_factory=list)
    precheck_gate_exit: int | None = None
    windows: list[tuple[float, str, str]] = field(default_factory=list)


async def run_experiment(
    cfg: BenchmarkConfig,
    deployment: Deployment,
    run_id: str,
    run_dir: Path,
    tokenizer: Tokenizer,
    launcher: EngineLauncher,
    registry_dir: Path,
    quality_gate: QualityHook | None = None,
    quality_compare: QualityHook | None = None,
) -> RunSummary:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = RunSummary(run_id=run_id, persisted=True)

    # phase 1 — dataset generation (§10; before any GPU is allocated, §1)
    log.info("phase: dataset generation")
    manifest = generate(cfg, tokenizer, run_dir, registry_dir)
    pool_path = run_dir / POOL_FILENAME
    assert pool_path.exists()

    # phase 2 — engine launch (only now, §1)
    log.info("phase: engine launch")
    endpoints = await launcher.launch(run_dir)

    db: ResultsDB | None = None
    try:
        # phase 3 — §7 pre-check gate outcome + smoke mode. results.json is
        # written by run_system_prechecks.sh BEFORE the engine binary starts,
        # but AFTER the job begins running — so wait for whichever comes
        # first: the results file (gate ran) or a healthy endpoint (gate was
        # skipped via §7.5). The smoke flag must be known before the DB opens.
        gate = await _await_precheck_results(
            run_dir, endpoints, cfg.phases.server_ready_timeout_s
        )
        if gate is not None:
            summary.precheck_gate_exit = gate.get("gate_exit_code")
            summary.smoke_test_mode = bool(gate.get("smoke_test_mode"))
            if summary.precheck_gate_exit not in (0, None):
                raise RunAborted(
                    f"§7.4 pre-check gate aborted the run (exit {summary.precheck_gate_exit}); "
                    f"see {run_dir / 'prechecks' / 'results.json'}"
                )
        if summary.smoke_test_mode:
            summary.persisted = False
            log.warning(
                "SMOKE-TEST MODE (§7.2): collective-tests cache was cold — "
                "results will NOT be persisted; re-run the experiment afterwards"
            )
        db = ResultsDB(run_dir / f"run_{run_id}.db", persist=summary.persisted)
        _insert_experiment_row(db, cfg, deployment, run_id, manifest)
        if gate is not None:
            db.insert_many(
                "system_prechecks",
                [{"run_id": run_id, "instance_id": endpoints[0][0], **row} for row in gate.get("rows", [])],
            )

        # phase 4 — readiness, model-load (§9.2), primer (§9.3)
        log.info("phase: readiness + primer")
        async with aiohttp.ClientSession() as http:
            for instance_id, url in endpoints:
                waited = await wait_ready(http, url, cfg.phases.server_ready_timeout_s)
                load = parse_model_load(launcher.engine_log_text() or "")
                db.insert(
                    "instances",
                    {"run_id": run_id, "instance_id": instance_id, "endpoint": url,
                     "node": None, "model_load_total_s": waited, **load},
                )
                primer = await run_primer(http, url, model=deployment.model)
                if primer.warning:
                    summary.primer_warnings.append(f"{instance_id}: {primer.warning}")
                    log.warning("primer (%s): %s", instance_id, primer.warning)

        # phase 5 — Stage-A quality gate (§12.5; M11)
        await _quality_stage(quality_gate, "gate", cfg.quality_eval.skip_quality_gate,
                             endpoints, db, run_id, summary)

        # phase 6 — λ sweep (§11.2/§11.3); per-step windows for §13.5 ingestion
        pool = load_pool(pool_path)
        weights = {e.scenario: e.weight for e in cfg.dataset_config.scenario_mix}
        for rate in cfg.rate_levels:
            log.info("phase: sweep step λ=%s", rate)
            window_start = datetime.now(timezone.utc)
            result = await run_step(pool, weights, _step_config(cfg, deployment, rate, endpoints))
            window_end = datetime.now(timezone.utc)
            summary.windows.append((rate, window_start.isoformat(), window_end.isoformat()))
            summary.sessions_started += result.sessions_started
            summary.sessions_truncated += result.sessions_truncated
            summary.requests += len(result.requests)
            summary.lag_warning = summary.lag_warning or result.lag_warning
            db.insert_request_rows(run_id, result.requests)
            db.insert_server_stats(run_id, result.server_stats)

        # phase 7 — Stage-B quality comparison (§12.5; M11)
        await _quality_stage(quality_compare, "compare", cfg.quality_eval.skip_quality_compare,
                             endpoints, db, run_id, summary)

        # phase 8 — finalisation: ingest engine-node sampler NDJSON (§13.5)
        log.info("phase: finalisation")
        windows = [
            (rate, datetime.fromisoformat(start), datetime.fromisoformat(end))
            for rate, start, end in summary.windows
        ]
        for ndjson in sorted(run_dir.glob("hw-*.ndjson")):
            ingested = db.ingest_hardware_ndjson(run_id, ndjson, endpoints[0][0], windows)
            log.info("ingested %d hardware samples from %s", ingested, ndjson.name)
        if summary.smoke_test_mode:
            log.warning("SMOKE-TEST MODE (§7.2): run completed but nothing was persisted")
        return summary
    finally:
        await launcher.teardown()
        if db is not None:
            db.close()


async def _quality_stage(hook, stage: str, skipped: bool, endpoints, db, run_id, summary) -> None:
    if skipped:
        log.info("quality %s skipped via config (§12.5)", stage)
        return
    if hook is None:
        # M11 pending: surfaced loudly, never silently dropped
        summary.quality_stages_pending.append(stage)
        log.warning("§12.5 quality %s NOT RUN — M11 quality-eval runner pending", stage)
        return
    log.info("phase: quality %s (§12.5)", stage)
    await hook(endpoints, db, run_id)


async def _await_precheck_results(
    run_dir: Path, endpoints: list[tuple[str, str]], timeout_s: float
) -> dict | None:
    path = run_dir / "prechecks" / "results.json"
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as http:
        while time.monotonic() < deadline:
            if path.exists():
                return json.loads(path.read_text())
            try:  # healthy endpoint + no file => §7.5 skip (or absent gate)
                async with http.get(f"{endpoints[0][1]}/health") as resp:
                    if resp.status == 200:
                        return None
            except aiohttp.ClientError:
                pass
            if not launcher.is_alive():
                # the gate may have aborted the engine job (§7.4) after writing
                # its results — give the shared filesystem a moment, then read
                await asyncio.sleep(10)
                if path.exists():
                    return json.loads(path.read_text())
                raise RunAborted(
                    "engine job died before §7 pre-checks produced results — "
                    "see the engine job log in the run directory"
                )
            await asyncio.sleep(5)
    raise RunAborted(
        f"neither pre-check results nor a healthy endpoint within {timeout_s}s "
        f"(engine bring-up failed before §7 completed?)"
    )


def _step_config(cfg: BenchmarkConfig, deployment: Deployment, rate: float, endpoints) -> StepConfig:
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
        scrape_interval_s=1.0,
    )


def _insert_experiment_row(db: ResultsDB, cfg: BenchmarkConfig, deployment: Deployment,
                           run_id: str, manifest: dict) -> None:
    db.insert(
        "experiments",
        {
            "run_id": run_id,
            "model": deployment.model,
            "backend": deployment.backend,
            "backend_config": deployment.backend_config.model_dump(exclude_none=True),
            "dataset_config": cfg.dataset_config.model_dump(exclude_none=True),
            "scenario_mix": [e.model_dump(exclude_none=True) for e in cfg.dataset_config.scenario_mix],
            "scenario_manifest": manifest,
            "slos": [s.model_dump(exclude_none=True) for s in cfg.slos] if cfg.slos else None,
            "quality_eval": cfg.quality_eval.model_dump(),
            "rate_levels": cfg.rate_levels,
            "warmup_s": cfg.phases.warmup_s,
            "measurement_s": cfg.phases.measurement_s,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
