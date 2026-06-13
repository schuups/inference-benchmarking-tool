"""§7.4 pre-check gate observation for the Coordinator's monitor loop (M8).

The engine container already enforces the gate inline (grade.py's exit code
gates `&& exec <engine>`), so the *decision* is pre-made by the rendered
`system_prechecks_on_warn` / `_on_fail` env. The Coordinator's role is to
**observe and surface** that outcome to the operator from `prechecks/results.json`
— prominently for warn/fail — and, in the interactive Claude-driven path, to host
the operator's abort/proceed choice. `decide_on_precheck` reports whether the run
is expected to proceed under the configured policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrecheckDecision:
    proceed: bool
    level: str  # "pass" | "warn" | "fail"
    message: str


def decide_on_precheck(results: dict, on_warn: str, on_fail: str) -> PrecheckDecision:
    """Interpret a parsed `prechecks/results.json` under the configured policy."""
    rows = results.get("rows", [])
    statuses = {r.get("status") for r in rows}
    code = results.get("gate_exit_code", 0)
    offending = [f"{r['metric']}={r.get('measured')}" for r in rows if r.get("status") in ("warn", "fail")]
    detail = f" ({', '.join(offending)})" if offending else ""
    if "fail" in statuses:
        return PrecheckDecision(
            on_fail == "continue", "fail",
            f"pre-check FAIL (gate_exit_code={code}); on_fail={on_fail}{detail}",
        )
    if "warn" in statuses:
        return PrecheckDecision(
            on_warn == "continue", "warn",
            f"pre-check WARN; on_warn={on_warn}{detail}",
        )
    return PrecheckDecision(True, "pass", "pre-checks pass")
