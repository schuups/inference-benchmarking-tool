"""§6.3–6.6 per-run teardown plan + executor (M8).

Teardown runs on **both** success and failure paths. The plan is computed
deterministically from `RunState` (testable); execution is best-effort — each
action is logged and one failure does not abort the rest, since the goal is to
leave no orphans. Model-cache PVCs are intentionally retained (§6.6); the plan
never targets them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .backend import ClusterBackend
from .state import RunState

log = logging.getLogger("coordinator.teardown")


@dataclass(frozen=True)
class TeardownAction:
    kind: str  # "cancel" | "remove_dir"
    target: str
    description: str


def teardown_plan(state: RunState) -> list[TeardownAction]:
    actions: list[TeardownAction] = []
    if state.benchmarker_handle:
        actions.append(
            TeardownAction("cancel", state.benchmarker_handle, "cancel Benchmarker job (§6.3)")
        )
    for handle in state.engine_handles:
        actions.append(
            TeardownAction("cancel", handle, "cancel inference deployment (§6.4/§6.5)")
        )
    # SLURM scratch run dir (§6.3); K8s remove_dir is a no-op (PVC retained, §6.6).
    if state.platform == "slurm" and state.run_dir_remote:
        actions.append(
            TeardownAction("remove_dir", state.run_dir_remote, "remove scratch run dir (§6.3)")
        )
    return actions


async def execute_teardown(
    state: RunState, backend: ClusterBackend
) -> list[tuple[TeardownAction, bool, str]]:
    """Apply the teardown plan best-effort; return (action, ok, detail) per action."""
    results: list[tuple[TeardownAction, bool, str]] = []
    for action in teardown_plan(state):
        try:
            if action.kind == "cancel":
                await backend.cancel(action.target)
            elif action.kind == "remove_dir":
                await backend.remove_dir(action.target)
            else:  # pragma: no cover - guarded by teardown_plan
                raise ValueError(f"unknown teardown action {action.kind!r}")
            results.append((action, True, "ok"))
            log.info("teardown ok: %s %s", action.kind, action.target)
        except Exception as exc:  # best-effort: log and continue (§6 leave no orphans)
            results.append((action, False, str(exc)))
            log.warning("teardown FAILED (continuing): %s %s — %s", action.kind, action.target, exc)
    return results
