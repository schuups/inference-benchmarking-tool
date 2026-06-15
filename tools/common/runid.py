"""Run-ID generation + parsing (SPECIFICATIONS.md §7.2).

Format: <timestamp>_<model-slug>_<backend>_<target>_<4-hex>
The random suffix prevents collisions when multiple Coordinators start within
the same second. `parse_run_id` is the single canonical reader of this structure
(used by the Coordinator to recover the deployment target and by the Cleaner to
validate scratch-dir names) so the field layout is defined in exactly one place.
"""

from __future__ import annotations

import hashlib
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


def k8s_slug(run_id: str, max_len: int = 51) -> str:
    """DNS-1035-safe slug (<=max_len) for K8s object names derived from run_id.

    K8s resource names and label values cap at 63 chars; the longest prefix we add
    is `ibt-results-` (12), so the slug is bounded to 51. run_id_slug routinely
    overflows that (timestamp + model + backend + target + suffix), so when it does
    we truncate and append a short stable hash to keep names unique. The FULL run_id
    is always preserved in the `inference-benchmarking/run-id` label for traceability,
    so the compact name costs nothing for discovery/teardown."""
    base = run_id_slug(run_id)
    if len(base) <= max_len:
        return base
    h = hashlib.sha1(run_id.encode()).hexdigest()[:6]
    return f"{base[: max_len - len(h) - 1].rstrip('-')}-{h}"


def model_cache_slug(model_id: str) -> str:
    """Full HF id (org included) -> model-cache PVC slug, matching the breithorn
    convention `model-cache-<org>-<name>`. Unlike model_slug (last segment, used
    for run-ids), this keeps the org so the rendered K8s claim binds the
    pre-populated cache PVC, e.g. swiss-ai/Apertus-70B-Instruct-2509 ->
    swiss-ai-apertus-70b-instruct-2509 (PVC model-cache-swiss-ai-apertus-70b-instruct-2509)."""
    return _SLUG_RE.sub("-", model_id.lower()).strip("-")


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
    """run_id → DNS/label-safe slug (K8s object names, §7.1). Underscores → '-'."""
    return re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
