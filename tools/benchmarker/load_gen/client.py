"""Streaming request execution + §13.1 error taxonomy."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import aiohttp


@dataclass
class RequestOutcome:
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float | None
    output_tokens: int
    output_text: str
    success: int
    error: str | None  # "<class>:<detail>" per §13.1


async def execute_request(
    http: aiohttp.ClientSession,
    url: str,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    ignore_eos: bool,
    request_timeout_s: float,
) -> RequestOutcome:
    """One streaming chat-completion; never raises — §13.1 classes in `error`."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "ignore_eos": ignore_eos,
    }
    start = time.perf_counter()
    first: float | None = None
    last: float | None = None
    chunks: list[str] = []
    saw_done = False
    try:
        async with http.post(f"{url}/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:120]
                return _failure(start, f"http_{resp.status}:{detail}", chunks, first, last)
            iterator = resp.content.__aiter__()
            while True:
                try:
                    remaining = (
                        request_timeout_s - (time.perf_counter() - start)
                        if first is None
                        else None  # §12.2: the client-side hard cutoff is TTFT-only
                    )
                    if remaining is not None and remaining <= 0:
                        raise asyncio.TimeoutError
                    raw = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError) as exc:
                    return _failure(start, f"server:malformed-sse:{exc}", chunks, first, last)
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                chunks.append(delta)
    except asyncio.TimeoutError:
        return _failure(start, f"timeout:ttft>{request_timeout_s}s", chunks, first, last)
    except asyncio.CancelledError:
        # drain deadline cancelled an in-flight request (§12.2); record, don't drop
        return _failure(start, "timeout:drain-cancelled", chunks, first, last)
    except aiohttp.ClientConnectionError as exc:
        return _failure(start, f"connection:{exc}", chunks, first, last)
    except aiohttp.ClientError as exc:
        return _failure(start, f"connection:{exc}", chunks, first, last)
    except Exception as exc:  # §13.1 'unknown': keep the raw message for triage
        return _failure(start, f"unknown:{exc}", chunks, first, last)

    if not saw_done:
        return _failure(start, "server:truncated-stream", chunks, first, last)
    end = time.perf_counter()
    n = len(chunks)
    return RequestOutcome(
        ttft_ms=(first - start) * 1000 if first else None,
        tpot_ms=((last - first) / (n - 1)) * 1000 if n > 1 else None,
        e2e_ms=(end - start) * 1000,
        output_tokens=n,
        output_text="".join(chunks),
        success=1,
        error=None,
    )


def _failure(start: float, error: str, chunks: list[str], first: float | None, last: float | None) -> RequestOutcome:
    now = time.perf_counter()
    n = len(chunks)
    return RequestOutcome(
        ttft_ms=(first - start) * 1000 if first else None,
        tpot_ms=((last - first) / (n - 1)) * 1000 if first and last and n > 1 else None,
        e2e_ms=(now - start) * 1000,
        output_tokens=n,
        output_text="".join(chunks),
        success=0,
        error=error,
    )
