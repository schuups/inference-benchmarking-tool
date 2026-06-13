"""Run-ID generation (SPECIFICATIONS.md §6.2).

Format: <timestamp>_<model-slug>_<backend>_<deployment>_<4-hex>
The random suffix prevents collisions when multiple Coordinators start within
the same second.
"""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone

_SLUG_RE = re.compile(r"[^a-z0-9.-]+")


def model_slug(model_id: str) -> str:
    """HF id -> filesystem/label-safe slug (last path segment, lowercased)."""
    return _SLUG_RE.sub("-", model_id.rsplit("/", 1)[-1].lower()).strip("-")


def make_run_id(
    model_id: str, backend: str, deployment: str, now: datetime | None = None
) -> str:
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{ts}_{model_slug(model_id)}_{backend}_{deployment}_{suffix}"


def run_id_slug(run_id: str) -> str:
    """run_id → DNS/label-safe slug (K8s object names, §6.1). Underscores → '-'."""
    return re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
