import re
from datetime import datetime, timezone

from tools.common.runid import make_run_id, model_slug

RUN_ID_RE = re.compile(
    r"^\d{8}-\d{6}_[a-z0-9.-]+_(vllm|sglang|dynamo)_[a-z0-9-]+_[0-9a-f]{4}$"
)


def test_format():
    rid = make_run_id("moonshotai/Kimi-K2.6", "vllm", "clariden")
    assert RUN_ID_RE.match(rid), rid


def test_model_slug_sanitization():
    assert model_slug("swiss-ai/Apertus-70B-Instruct-2509") == "apertus-70b-instruct-2509"
    assert model_slug("moonshotai/Kimi-K2.6") == "kimi-k2.6"
    assert model_slug("Weird  Name!!") == "weird-name"


def test_suffix_uniqueness():
    now = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)
    ids = {make_run_id("m/x", "vllm", "clariden", now=now) for _ in range(50)}
    assert len(ids) > 1  # same second, distinct random suffixes


def test_timestamp_respected():
    now = datetime(2026, 6, 12, 8, 30, 15, tzinfo=timezone.utc)
    assert make_run_id("m/x", "vllm", "clariden", now=now).startswith("20260612-083015_")
