"""M10 DoD: §6.7 identification lists orphaned resources and applies the skip
policy (model-cache PVCs §6.6, recent-N JFrog, active-job scratch, age threshold);
pruning removes exactly the approved list.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tools.cleaner import (
    Candidate,
    FakeCleanerBackend,
    identify,
    parse_run_id,
    prune,
    reminder_due,
    scratch_candidates,
)

OLD_RUN = "20260613-120000_kimi-k2.6_vllm_clariden_ab12"
ACTIVE_RUN = "20260613-130000_kimi-k2.6_vllm_clariden_cd34"
YOUNG_RUN = "20260613-140000_kimi-k2.6_vllm_clariden_ef56"


def _candidates():
    return [
        Candidate("scratch", f"/scratch/{OLD_RUN}", age_hours=48, run_id=OLD_RUN),
        Candidate("scratch", f"/scratch/{ACTIVE_RUN}", age_hours=48, run_id=ACTIVE_RUN),
        Candidate("scratch", "/scratch/collective-tests-cache", age_hours=999, run_id=None),
        Candidate("scratch", f"/scratch/{YOUNG_RUN}", age_hours=2, run_id=YOUNG_RUN),
        Candidate("k8s", "persistentvolumeclaim/model-cache-kimi", age_hours=999),
        Candidate("k8s", "deployment/ib-engine-xyz", age_hours=48, run_id=OLD_RUN),
        Candidate("jfrog", "inference-benchmarking-runA", age_hours=1),
        Candidate("jfrog", "inference-benchmarking-runB", age_hours=10),
        Candidate("jfrog", "inference-benchmarking-runC", age_hours=100),
    ]


def test_parse_run_id():
    assert parse_run_id(OLD_RUN) == OLD_RUN
    assert parse_run_id("collective-tests-cache") is None
    assert parse_run_id("20260613-120000_m_vllm_clariden_zzzz") is None  # zzzz not hex


def test_identify_applies_skip_policy():
    report = identify(
        _candidates(), age_threshold_h=24, keep_recent_jfrog=2, active_run_ids={ACTIVE_RUN}
    )
    prunable = {c.ident for c in report.prunable}
    assert prunable == {
        f"/scratch/{OLD_RUN}",
        "deployment/ib-engine-xyz",
        "inference-benchmarking-runC",  # runA/runB kept as the 2 most recent
    }
    reasons = {c.ident: reason for c, reason in report.skipped}
    assert "model-cache" in reasons["persistentvolumeclaim/model-cache-kimi"]  # §6.6
    assert "active job" in reasons[f"/scratch/{ACTIVE_RUN}"]
    assert "not a benchmark run dir" in reasons["/scratch/collective-tests-cache"]
    assert "age threshold" in reasons[f"/scratch/{YOUNG_RUN}"]
    assert "most recent" in reasons["inference-benchmarking-runA"]


@pytest.mark.asyncio
async def test_prune_removes_exactly_the_approved_list():
    backend = FakeCleanerBackend(_candidates())
    report = identify(await backend.list_candidates(), keep_recent_jfrog=2, active_run_ids={ACTIVE_RUN})
    results = await prune(backend, report.prunable)
    assert all(ok for _, ok, _ in results)
    assert set(backend.deleted) == {c.ident for c in report.prunable}
    # the model-cache PVC was never in the prune set
    assert "persistentvolumeclaim/model-cache-kimi" not in backend.deleted


def test_scratch_candidates_from_listing():
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
    entries = [
        {"name": OLD_RUN, "path": f"/scratch/{OLD_RUN}", "mtime_epoch": (now - timedelta(hours=30)).timestamp()},
        {"name": "hf-cache", "path": "/scratch/hf-cache", "mtime_epoch": now.timestamp()},
    ]
    cands = scratch_candidates(entries, now=now)
    by_ident = {c.ident: c for c in cands}
    assert by_ident[f"/scratch/{OLD_RUN}"].run_id == OLD_RUN
    assert by_ident[f"/scratch/{OLD_RUN}"].age_hours == pytest.approx(30, abs=0.1)
    assert by_ident["/scratch/hf-cache"].run_id is None  # not a run dir


def test_reminder_due():
    now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
    assert reminder_due(None, now=now) is True  # never run
    assert reminder_due((now - timedelta(hours=200)).isoformat(), interval_h=168, now=now) is True
    assert reminder_due((now - timedelta(hours=10)).isoformat(), interval_h=168, now=now) is False
