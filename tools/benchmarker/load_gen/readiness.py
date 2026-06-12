"""Readiness wait (§11.1), model-load breakdown parsing (§9.2), primer (§9.3).

The model-load regexes target vLLM's structured log lines. They are seeded from
upstream-vLLM message shapes and MUST be re-validated against a captured log of
the pinned vllm-cxi image at E1 (parser-fixture discipline per plan M2 DoD);
components a log does not expose stay None -> stored as NULL (§9.2).
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

import aiohttp

from .client import execute_request

MODEL_LOAD_PATTERNS = {
    "model_load_weights_s": re.compile(r"Model loading took [\d.]+ GiB and ([\d.]+) seconds"),
    "model_load_engine_init_s": re.compile(r"init engine \([^)]*\) took ([\d.]+) seconds"),
    "model_load_cuda_graph_capture_s": re.compile(r"Graph capturing finished in ([\d.]+) secs"),
    "model_load_inductor_compile_s": re.compile(r"torch\.compile takes ([\d.]+) s"),
}


def parse_model_load(log_text: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for column, pattern in MODEL_LOAD_PATTERNS.items():
        m = pattern.search(log_text)
        out[column] = float(m.group(1)) if m else None
    return out


async def wait_ready(
    http: aiohttp.ClientSession, url: str, timeout_s: float, poll_interval_s: float = 2.0
) -> float:
    """Polls /health until 200; returns seconds waited (-> model_load_total_s input)."""
    start = time.perf_counter()
    while True:
        try:
            async with http.get(f"{url}/health") as resp:
                if resp.status == 200:
                    return time.perf_counter() - start
        except aiohttp.ClientError:
            pass
        if time.perf_counter() - start > timeout_s:
            raise TimeoutError(
                f"instance at {url} not ready within server_ready_timeout_s={timeout_s} (§11.1)"
            )
        await asyncio.sleep(poll_interval_s)


@dataclass
class PrimerResult:
    primer_s: float
    probe1_ttft_ms: float | None
    probe2_ttft_ms: float | None
    warning: str | None


async def run_primer(
    http: aiohttp.ClientSession,
    url: str,
    *,
    model: str,
    prompt_tokens: int = 20_000,
    timeout_s: float = 300.0,
    probe_tokens: int = 2_000,
) -> PrimerResult:
    """§9.3: one large priming request, then two probes. The primer is judged
    self-calibratingly: if probe 1's TTFT is far above probe 2's, the first
    measurement-like request still paid a compile cost -> warn the operator."""
    start = time.perf_counter()
    primer = await execute_request(
        http, url,
        model=model,
        messages=[{"role": "user", "content": "prime " * prompt_tokens}],
        max_tokens=1, ignore_eos=True, request_timeout_s=timeout_s,
    )
    primer_s = time.perf_counter() - start
    if not primer.success:
        return PrimerResult(primer_s, None, None, f"primer request failed: {primer.error}")

    probes = []
    for _ in range(2):
        probes.append(
            await execute_request(
                http, url,
                model=model,
                messages=[{"role": "user", "content": "probe " * probe_tokens}],
                max_tokens=1, ignore_eos=True, request_timeout_s=timeout_s,
            )
        )
    t1, t2 = probes[0].ttft_ms, probes[1].ttft_ms
    warning = None
    if t1 is None or t2 is None:
        warning = "primer probes failed; steady-state TTFT unverified"
    elif t1 > 3 * t2 + 100:
        warning = (
            f"primer missed its target: first probe TTFT {t1:.0f}ms vs steady-state "
            f"{t2:.0f}ms — the first sweep request may still pay a compile cost (§9.3)"
        )
    return PrimerResult(primer_s, t1, t2, warning)
