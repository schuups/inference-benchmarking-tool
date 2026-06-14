"""Per-instance server_stats scraping (§14.4) from the backend's /metrics."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import aiohttp

_GAUGES = {
    "requests_running": re.compile(r"vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)"),
    "requests_waiting": re.compile(r"vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)"),
    "gpu_cache_pct": re.compile(r"vllm:gpu_cache_usage_perc(?:\{[^}]*\})?\s+([0-9.eE+-]+)"),
    "spec_accept_rate": re.compile(r"vllm:spec_decode_draft_acceptance_rate(?:\{[^}]*\})?\s+([0-9.eE+-]+)"),
}


def parse_metrics(body: str) -> dict:
    out: dict = {}
    for key, pattern in _GAUGES.items():
        m = pattern.search(body)
        out[key] = float(m.group(1)) if m else None
    if out.get("gpu_cache_pct") is not None:
        out["gpu_cache_pct"] *= 100.0  # vLLM exports a 0-1 fraction
    return out


async def scrape_server_stats(
    http: aiohttp.ClientSession,
    endpoints: list[tuple[str, str]],
    rate_lambda: float,
    interval_s: float,
    sink: list[dict],
) -> None:
    """Appends one row per instance per tick to `sink` until cancelled."""
    while True:
        ts = datetime.now(timezone.utc).isoformat()
        for instance_id, url in endpoints:
            row = {
                "instance_id": instance_id,
                "rate_lambda": rate_lambda,
                "ts": ts,
                "requests_running": None,
                "requests_waiting": None,
                "gpu_cache_pct": None,
                "spec_accept_rate": None,
            }
            try:
                async with http.get(f"{url}/metrics") as resp:
                    if resp.status == 200:
                        row.update(parse_metrics(await resp.text()))
            except aiohttp.ClientError:
                pass  # sampling gap, not a request failure; row stays NULL-valued
            sink.append(row)
        await asyncio.sleep(interval_s)
