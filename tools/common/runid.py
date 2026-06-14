"""Run-ID generation + parsing (SPECIFICATIONS.md §6.2).

Format: <timestamp>_<model-slug>_<backend>_<target>_<4-hex>
The random suffix prevents collisions when multiple Coordinators start within
the same second. `parse_run_id` is the single canonical reader of this structure
(used by the Coordinator to recover the deployment target and by the Cleaner to
validate scratch-dir names) so the field layout is defined in exactly one place.
"""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import NamedTuple

_SLUG_RE = re.compile(r"[^a-z0-9.-]+")

# model-slug / target are slugified (no underscores); backend is a bare token;
# the timestamp is digits+hyphen and the suffix is 4 hex chars.
RUN_ID_RE = re.compile(
    r"^(?P<ts>\d{8}-\d{6})"
    r"_(?P<model>[a-z0-9.-]+)"
    r"_(?P<backend>[a-z0-9.-]+)"
    r"_(?P<target>[a-z0-9-]+)"
    r"_(?P<suffix>[0-9a-f]{4})$"
)


class RunIdParts(NamedTuple):
    ts: str
    model: str
    backend: str
    target: str
    suffix: str


def model_slug(model_id: str) -> str:
    """HF id -> filesystem/label-safe slug (last path segment, lowercased)."""
    return _SLUG_RE.sub("-", model_id.rsplit("/", 1)[-1].lower()).strip("-")


def make_run_id(
    model_id: str, backend: str, target: str, now: datetime | None = None
) -> str:
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{ts}_{model_slug(model_id)}_{backend}_{target}_{suffix}"


def parse_run_id(run_id: str) -> RunIdParts | None:
    """Parse a run_id into its components, or None if it is not well-formed."""
    m = RUN_ID_RE.match(run_id)
    return RunIdParts(**m.groupdict()) if m else None


def run_id_slug(run_id: str) -> str:
    """run_id → DNS/label-safe slug (K8s object names, §6.1). Underscores → '-'."""
    return re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
