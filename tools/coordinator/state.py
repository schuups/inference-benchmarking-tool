"""Coordinator run state (IMPLEMENTATION_PLAN.md M8) — resumable orchestration.

A per-run JSON state file lets the Coordinator reattach to an in-flight
experiment after laptop sleep / network loss and skip already-completed phases.
It lives in the local run directory (`experiments/<exp>/<run_id>/`) next to the
planner artifacts and the downloaded per-run DB (§14.8).

Phases are linear; `phase` records the last completed step. Failure does not get
its own phase — the error is recorded in `error` and `phase` stays at the last
good step, so a resume retries from there.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STATE_FILENAME = "coordinator_state.json"

# Linear phase order; resume continues from the recorded (last-completed) phase.
PHASES = (
    "created",    # state initialised, nothing submitted
    "staged",     # artifacts staged to cluster scratch
    "submitted",  # benchmarker job submitted
    "completed",  # benchmarker job finished (sweep done)
    "collected",  # per-run DB downloaded to the local run dir
    "merged",     # per-run DB merged into the centralized results DB
    "torn_down",  # §7 teardown applied
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunState:
    run_id: str
    platform: str  # "slurm" | "k8s"
    target: str  # cluster name
    run_dir_local: str  # experiments/<exp>/<run_id>
    run_dir_remote: str  # <scratch_base>/<run_id>
    phase: str = "created"
    benchmarker_handle: str | None = None  # SLURM job id / K8s pod name
    engine_handles: list[str] = field(default_factory=list)  # discovered, for §7.4/§7.5 teardown
    db_sha256: str | None = None  # checksum of the collected per-run DB
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"unknown phase {self.phase!r} (expected one of {PHASES})")
        if self.created_at is None:
            self.created_at = _now()

    @staticmethod
    def state_file(run_dir_local: Path | str) -> Path:
        return Path(run_dir_local) / STATE_FILENAME

    @property
    def state_path(self) -> Path:
        return self.state_file(self.run_dir_local)

    def save(self) -> None:
        self.updated_at = _now()
        Path(self.run_dir_local).mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, run_dir_local: Path | str) -> "RunState":
        return cls(**json.loads(cls.state_file(run_dir_local).read_text()))

    @classmethod
    def exists(cls, run_dir_local: Path | str) -> bool:
        return cls.state_file(run_dir_local).exists()

    # --- phase ordering helpers (resume logic) ---

    def is_done(self, phase: str) -> bool:
        """True if `phase` has been completed (current phase is at or past it)."""
        return PHASES.index(self.phase) >= PHASES.index(phase)

    def is_before(self, phase: str) -> bool:
        return not self.is_done(phase)

    def advance(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        self.phase = phase
        self.save()
