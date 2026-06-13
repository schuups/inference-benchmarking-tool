"""Cleaner (SPECIFICATIONS.md §6.7, IMPLEMENTATION_PLAN.md M10).

Reclaims state that escaped the Coordinator's per-run teardown — Coordinator
killed mid-run, network loss during teardown, runs predating a teardown fix.
**Operator-run, never automatic**: Claude only ever *reminds* the operator to run
it (`reminder_due`); it never prunes on its own.

Two stages:
  1. identify() — ALWAYS read-only. Lists candidates discovered via the §6.1
     labels/patterns and applies the skip policy (model-cache PVCs §6.6, the most
     recent N JFrog tags, scratch dirs owned by an active job, an age threshold).
  2. prune() — requires explicit operator approval (the CLI's `--yes`). Deletes
     exactly the approved candidates via the backend.

`identify()` is a pure function (tested over all three resource classes). Discovery/
deletion is delegated to a `CleanerBackend`: K8s via `kubectl` runs headless
(`KubectlCleanerBackend`). SLURM scratch discovery/removal is assistant-driven via
the FirecREST MCP (decision 5), feeding `scratch_candidates()` into the same
`identify()` policy. The JFrog tag backend (`jf`/REST) is a follow-up (see TODOs) —
`identify()` already applies the keep-recent-N policy to JFrog candidates. Cleaner
actions are logged on the laptop and are NOT persisted to the per-run results DB.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

log = logging.getLogger("cleaner")

MANAGED_BY = "inference-benchmarking"
# §6.2 run-ID shape: <YYYYMMDD-HHMMSS>_<model-slug>_<backend>_<target>_<4hex>
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}_[a-z0-9.-]+_[a-z0-9]+_[a-z0-9-]+_[0-9a-f]{4}$")

DEFAULT_AGE_THRESHOLD_H = 24.0
DEFAULT_KEEP_RECENT_JFROG = 3
DEFAULT_REMINDER_INTERVAL_H = 168.0  # weekly


def parse_run_id(name: str) -> str | None:
    """Return the name if it matches the §6.2 run-ID pattern, else None."""
    return name if RUN_ID_RE.match(name) else None


@dataclass
class Candidate:
    kind: str  # "k8s" | "scratch" | "jfrog"
    ident: str  # "<kind>/<name>" | scratch dir path | image tag
    age_hours: float
    run_id: str | None = None
    detail: str = ""


@dataclass
class CleanReport:
    prunable: list[Candidate] = field(default_factory=list)
    skipped: list[tuple[Candidate, str]] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Cleaner candidate report ({len(self.prunable)} prunable, {len(self.skipped)} skipped):"]
        lines.append("  PRUNABLE (delete only after approval):")
        lines += [f"    - [{c.kind}] {c.ident}  (age {c.age_hours:.0f}h) {c.detail}".rstrip()
                  for c in self.prunable] or ["    (none)"]
        lines.append("  SKIPPED:")
        lines += [f"    - [{c.kind}] {c.ident} — {reason}" for c, reason in self.skipped] or ["    (none)"]
        return "\n".join(lines)


def _skip_reason(c: Candidate, age_threshold_h: float, jfrog_keep: set[str], active_run_ids) -> str | None:
    if c.kind == "k8s" and "model-cache" in c.ident:
        return "model-cache PVC retained (§6.6)"
    if c.kind == "scratch":
        if c.run_id is None:
            return "not a benchmark run dir (§6.2 pattern)"
        if c.run_id in active_run_ids:
            return "owned by an active job"
    if c.kind == "jfrog" and c.ident in jfrog_keep:
        return f"among the {len(jfrog_keep)} most recent tags"
    if c.age_hours < age_threshold_h:
        return f"younger than the {age_threshold_h:.0f}h age threshold"
    return None


def identify(
    candidates: list[Candidate],
    *,
    age_threshold_h: float = DEFAULT_AGE_THRESHOLD_H,
    keep_recent_jfrog: int = DEFAULT_KEEP_RECENT_JFROG,
    active_run_ids=frozenset(),
) -> CleanReport:
    """Read-only §6.7 stage 1: partition candidates into prunable vs skipped."""
    jfrog = sorted((c for c in candidates if c.kind == "jfrog"), key=lambda c: c.age_hours)
    jfrog_keep = {c.ident for c in jfrog[:keep_recent_jfrog]}  # keep the youngest N
    report = CleanReport()
    for c in candidates:
        reason = _skip_reason(c, age_threshold_h, jfrog_keep, active_run_ids)
        if reason:
            report.skipped.append((c, reason))
        else:
            report.prunable.append(c)
    return report


def reminder_due(last_cleanup_iso: str | None, interval_h: float = DEFAULT_REMINDER_INTERVAL_H,
                 now: datetime | None = None) -> bool:
    """§6.7: whether Claude should remind the operator to run the Cleaner."""
    if last_cleanup_iso is None:
        return True
    now = now or datetime.now(timezone.utc)
    elapsed_h = (now - datetime.fromisoformat(last_cleanup_iso)).total_seconds() / 3600
    return elapsed_h >= interval_h


# --------------------------------------------------------------------- backends


class CleanerBackend(Protocol):
    async def list_candidates(self) -> list[Candidate]: ...
    async def delete(self, candidate: Candidate) -> None: ...


async def prune(backend: CleanerBackend, candidates: list[Candidate]) -> list[tuple[Candidate, bool, str]]:
    """§6.7 stage 2: delete the operator-approved candidates, best-effort."""
    results = []
    for c in candidates:
        try:
            await backend.delete(c)
            results.append((c, True, "deleted"))
            log.info("pruned [%s] %s", c.kind, c.ident)
        except Exception as exc:  # best-effort; report and continue
            results.append((c, False, str(exc)))
            log.warning("prune FAILED [%s] %s — %s", c.kind, c.ident, exc)
    return results


def scratch_candidates(entries: list[dict], now: datetime | None = None) -> list[Candidate]:
    """Build scratch Candidates from a FirecREST-style listing (assistant/MCP path).

    `entries` = [{"name": <dir>, "mtime_epoch": <float>, "path": <full path>}, …].
    """
    now = now or datetime.now(timezone.utc)
    out = []
    for e in entries:
        age_h = (now.timestamp() - e["mtime_epoch"]) / 3600
        out.append(Candidate(
            kind="scratch", ident=e.get("path", e["name"]), age_hours=age_h,
            run_id=parse_run_id(e["name"]),
        ))
    return out


async def _run(*args: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _age_hours(iso_ts: str, now: datetime) -> float:
    try:
        return (now - datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))).total_seconds() / 3600
    except ValueError:
        return 0.0


class KubectlCleanerBackend:
    """Headless K8s discovery/deletion via kubectl over the §6.1 managed-by label."""

    KINDS = "deployment,service,ingress,secret,persistentvolumeclaim"

    def __init__(self, namespace: str):
        self._ns = namespace

    async def list_candidates(self) -> list[Candidate]:
        code, out, err = await _run(
            "kubectl", "get", self.KINDS, "-n", self._ns,
            "-l", f"app.kubernetes.io/managed-by={MANAGED_BY}", "-o", "json",
        )
        if code != 0:
            raise RuntimeError(f"kubectl get failed: {err.strip() or out.strip()}")
        now = datetime.now(timezone.utc)
        items = json.loads(out).get("items", [])
        cands = []
        for it in items:
            kind = it.get("kind", "?")
            meta = it.get("metadata", {})
            name = meta.get("name", "?")
            cands.append(Candidate(
                kind="k8s", ident=f"{kind.lower()}/{name}",
                age_hours=_age_hours(meta.get("creationTimestamp", ""), now),
                run_id=meta.get("labels", {}).get("inference-benchmarking/run-id"),
            ))
        return cands

    async def delete(self, candidate: Candidate) -> None:
        kind_name = candidate.ident.split("/", 1)
        code, _, err = await _run("kubectl", "delete", kind_name[0], kind_name[1], "-n", self._ns)
        if code != 0:
            raise RuntimeError(err.strip())


class FakeCleanerBackend:
    """In-process backend for tests: returns a fixed candidate list, records deletes."""

    def __init__(self, candidates: list[Candidate]):
        self._candidates = candidates
        self.deleted: list[str] = []

    async def list_candidates(self) -> list[Candidate]:
        return list(self._candidates)

    async def delete(self, candidate: Candidate) -> None:
        self.deleted.append(candidate.ident)


# ----------------------------------------------------------------------- CLI


async def _cli_identify(backend: CleanerBackend, age_threshold_h: float, keep_recent: int) -> CleanReport:
    return identify(await backend.list_candidates(), age_threshold_h=age_threshold_h,
                    keep_recent_jfrog=keep_recent)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleaner (§6.7) — identify (default) or prune")
    parser.add_argument("--namespace", default="ml", help="K8s namespace (breithorn)")
    parser.add_argument("--age-threshold-h", type=float, default=DEFAULT_AGE_THRESHOLD_H)
    parser.add_argument("--keep-recent-jfrog", type=int, default=DEFAULT_KEEP_RECENT_JFROG)
    parser.add_argument("--prune", action="store_true", help="delete the prunable candidates")
    parser.add_argument("--yes", action="store_true", help="confirm pruning (required with --prune)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # K8s is the headless class here; SLURM scratch + JFrog discovery is assistant-driven
    # via the FirecREST MCP / jf and feeds identify() the same way (decision 5).
    backend = KubectlCleanerBackend(args.namespace)
    report = asyncio.run(_cli_identify(backend, args.age_threshold_h, args.keep_recent_jfrog))
    print(report.render())

    if args.prune:
        if not args.yes:
            print("\nRefusing to prune without --yes (§6.7 requires explicit operator approval).")
            return 2
        results = asyncio.run(prune(backend, report.prunable))
        print(f"\nPruned {sum(ok for _, ok, _ in results)}/{len(results)} candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
