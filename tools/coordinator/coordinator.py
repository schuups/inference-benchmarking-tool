"""Coordinator orchestration loop (IMPLEMENTATION_PLAN.md M8).

Drives one experiment end-to-end over a `ClusterBackend`:

    stage → submit → monitor → collect (staged DB download) → merge → teardown

The loop is **resumable** — it reads the recorded phase from `RunState` and skips
already-completed steps, so a Coordinator killed by laptop sleep / network loss
reattaches and continues. Teardown (§6) runs on **both** success and failure; on
failure the loop first makes a best-effort attempt to salvage and merge any
partial per-run DB the Benchmarker persisted (§7.4), then tears down.

This loop runs autonomously for the K8s (kubectl) backend and in tests (the fake
backend). For SLURM, the operator's decision routes FirecREST through the MCP
tools driven by the assistant in-session, which performs the same phase sequence
using these helpers — there is no autonomous SLURM backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .backend import ClusterBackend
from .merge import merge_run_db
from .policy import decide_on_precheck
from .state import RunState
from .teardown import execute_teardown

log = logging.getLogger("coordinator")


class CoordinatorError(RuntimeError):
    pass


class Coordinator:
    def __init__(
        self,
        state: RunState,
        backend: ClusterBackend,
        central_db: Path | str,
        *,
        benchmarker_script: str = "benchmarker.sbatch",
        on_warn: str = "abort",
        on_fail: str = "abort",
        poll_interval_s: float = 30.0,
        max_polls: int | None = None,
    ):
        self.state = state
        self.backend = backend
        self.central_db = Path(central_db)
        self.script = benchmarker_script
        self.on_warn = on_warn
        self.on_fail = on_fail
        self.poll_interval_s = poll_interval_s
        self.max_polls = max_polls
        self._precheck_reported = False
        self.teardown_results: list = []

    async def run(self) -> RunState:
        try:
            if self.state.is_before("staged"):
                await self._stage()
                self.state.advance("staged")
            if self.state.is_before("submitted"):
                await self._submit()
                self.state.advance("submitted")
            if self.state.is_before("completed"):
                await self._monitor()
                self.state.advance("completed")
            if self.state.is_before("collected"):
                await self._collect()
                self.state.advance("collected")
            if self.state.is_before("merged"):
                await self._merge()
                self.state.advance("merged")
        except Exception as exc:
            self.state.error = str(exc)
            self.state.save()
            await self._salvage()  # best-effort: rescue a partial per-run DB (§7.4)
            await self._teardown_once()  # §6: teardown on the failure path
            raise
        await self._teardown_once()  # §6: teardown on the success path
        return self.state

    # ----------------------------------------------------------- phases

    async def _stage(self) -> None:
        log.info("[%s] staging artifacts → %s", self.state.run_id, self.state.run_dir_remote)
        await self.backend.stage(Path(self.state.run_dir_local), self.state.run_dir_remote)

    async def _submit(self) -> None:
        handle = await self.backend.submit(self.state.run_dir_remote, self.script)
        self.state.benchmarker_handle = handle
        log.info("[%s] submitted Benchmarker job %s", self.state.run_id, handle)

    async def _monitor(self) -> None:
        results_path = f"{self.state.run_dir_remote}/prechecks/results.json"
        polls = 0
        while True:
            status = await self.backend.status(self.state.benchmarker_handle)
            if not self.state.engine_handles:  # cache discovered handles for teardown
                self.state.engine_handles = await self.backend.discover_engine_handles(self.state.run_id)
                if self.state.engine_handles:
                    self.state.save()
            await self._report_precheck(results_path)
            if status == "completed":
                return
            if status == "failed":
                raise CoordinatorError(
                    f"Benchmarker job {self.state.benchmarker_handle} failed "
                    f"(run {self.state.run_id}); inspect the benchmarker/engine logs"
                )
            polls += 1
            if self.max_polls is not None and polls >= self.max_polls:
                raise CoordinatorError(f"monitor exceeded max_polls={self.max_polls}")
            await asyncio.sleep(self.poll_interval_s)

    async def _collect(self) -> None:
        sha = await self.backend.fetch_db(self._remote_db(), self._local_db())
        self.state.db_sha256 = sha
        log.info("[%s] collected per-run DB (sha256=%s…)", self.state.run_id, sha[:12])

    async def _merge(self) -> None:
        counts = merge_run_db(self._local_db(), self.central_db, self.state.run_id)
        log.info("[%s] merged into %s: %s", self.state.run_id, self.central_db, counts)

    # ----------------------------------------------------------- helpers

    def _remote_db(self) -> str:
        return f"{self.state.run_dir_remote}/run_{self.state.run_id}.db"

    def _local_db(self) -> Path:
        return Path(self.state.run_dir_local) / f"run_{self.state.run_id}.db"

    async def _report_precheck(self, results_path: str) -> None:
        if self._precheck_reported:
            return
        txt = await self.backend.read_remote(results_path)
        if not txt:
            return
        try:
            results = json.loads(txt)
        except json.JSONDecodeError:
            return  # partially written; report on a later poll
        decision = decide_on_precheck(results, self.on_warn, self.on_fail)
        self._precheck_reported = True
        level = {"fail": logging.ERROR, "warn": logging.WARNING}.get(decision.level, logging.INFO)
        log.log(level, "[%s] %s", self.state.run_id, decision.message)

    async def _salvage(self) -> None:
        """Failure path: try to download + merge any partial per-run DB before teardown."""
        try:
            if self.state.is_before("collected"):
                await self._collect()
                self.state.advance("collected")
            if self.state.is_before("merged"):
                await self._merge()
                self.state.advance("merged")
        except Exception as exc:
            log.warning(
                "[%s] partial-results salvage failed (continuing to teardown): %s",
                self.state.run_id, exc,
            )

    async def _teardown_once(self) -> None:
        if self.state.is_done("torn_down"):
            return
        self.teardown_results = await execute_teardown(self.state, self.backend)
        failed = [(a, detail) for a, ok, detail in self.teardown_results if not ok]
        if failed:
            # Don't record teardown as complete on partial failure (§6 "leave no
            # orphans"): keep the phase so a later --resume retries the actions,
            # and surface the failures to the operator without clobbering any
            # pre-existing run error.
            msg = "; ".join(f"teardown {a.kind} {a.target}: {detail}" for a, detail in failed)
            self.state.error = f"{self.state.error}; {msg}" if self.state.error else msg
            log.warning(
                "[%s] teardown incomplete: %d/%d actions failed — leaving phase=%r for retry",
                self.state.run_id, len(failed), len(self.teardown_results), self.state.phase,
            )
            return
        self.state.advance("torn_down")
