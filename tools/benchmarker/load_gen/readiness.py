"""Readiness wait (§12.1), model-load breakdown parsing (§10.2), primer (§10.3).

The model-load regexes target vLLM's structured log lines. They are seeded from
upstream-vLLM message shapes and MUST be re-validated against a captured log of
the pinned vLLM image at E1 (parser-fixture discipline per plan M2 DoD);
components a log does not expose stay None -> stored as NULL (§10.2).
"""

from __future__ import annotations

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
    """§10.3: one large priming request, then two probes. The primer is judged
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
            f"{t2:.0f}ms — the first sweep request may still pay a compile cost (§10.3)"
        )
    return PrimerResult(primer_s, t1, t2, warning)
