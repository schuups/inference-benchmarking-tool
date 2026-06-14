"""Mock OpenAI-compatible inference server (IMPLEMENTATION_PLAN.md M2 test harness).

Ground truth for load-generator and quality-eval tests:
- streaming /v1/chat/completions with configurable TTFT / TPOT delays,
- deterministic canned-answer mode (fixed response per prompt substring — the
  M11 eval-logic tests grade against these),
- vLLM-style Prometheus /metrics (requests running/waiting, KV cache pct),
- /health, and fault injection (error rate, HTTP status, mid-stream aborts).

Standalone: python -m tools.testing.mock_openai_server --port 8001 --ttft-ms 100 --tpot-ms 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field

from aiohttp import web

FILLER_TOKEN = "tok "


@dataclass
class MockConfig:
    ttft_ms: float = 50.0
    tpot_ms: float = 5.0
    model: str = "mock/model"
    # canned-answer mode: first (substring, answer) match wins; None -> filler tokens
    canned: list[tuple[str, str]] = field(default_factory=list)
    error_rate: float = 0.0          # fraction of requests answered with error_status
    error_status: int = 500
    abort_mid_stream_after: int | None = None  # tokens emitted before truncating
    slow_first_n: int = 0            # cold-start simulation: first N requests use slow_ttft_ms
    slow_ttft_ms: float = 0.0
    seed: int = 0


class MockServer:
    def __init__(self, config: MockConfig | None = None):
        self.config = config or MockConfig()
        self._rng = random.Random(self.config.seed)
        self.requests_running = 0
        self.requests_received = 0
        self.app = web.Application()
        self.app.add_routes(
            [
                web.post("/v1/chat/completions", self._chat),
                web.get("/health", self._health),
                web.get("/metrics", self._metrics),
            ]
        )

    async def _health(self, request: web.Request) -> web.Response:
        return web.Response(text="OK")

    async def _metrics(self, request: web.Request) -> web.Response:
        body = (
            f"vllm:num_requests_running {self.requests_running}\n"
            f"vllm:num_requests_waiting 0\n"
            f"vllm:gpu_cache_usage_perc {min(0.95, self.requests_running * 0.05):.4f}\n"
        )
        return web.Response(text=body, content_type="text/plain")

    def _answer_for(self, prompt: str, max_tokens: int) -> list[str]:
        for needle, answer in self.config.canned:
            if needle in prompt:
                return answer.split()  # whitespace tokens, deterministic
        return [FILLER_TOKEN.strip()] * max_tokens

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        self.requests_received += 1
        if self.config.error_rate and self._rng.random() < self.config.error_rate:
            return web.Response(status=self.config.error_status, text="injected failure")
        payload = await request.json()
        prompt = " ".join(m.get("content", "") for m in payload.get("messages", []))
        max_tokens = int(payload.get("max_tokens", 16))
        ignore_eos = bool(payload.get("ignore_eos", False))
        tokens = self._answer_for(prompt, max_tokens)
        if ignore_eos:
            tokens = (tokens * (max_tokens // max(1, len(tokens)) + 1))[:max_tokens]
        else:
            tokens = tokens[:max_tokens]
        if self.config.abort_mid_stream_after is not None:
            tokens = tokens[: self.config.abort_mid_stream_after]

        ttft_ms = (
            self.config.slow_ttft_ms
            if self.requests_received <= self.config.slow_first_n
            else self.config.ttft_ms
        )
        response = web.StreamResponse()
        response.content_type = "text/event-stream"
        await response.prepare(request)
        self.requests_running += 1
        try:
            await asyncio.sleep(ttft_ms / 1000)
            for i, token in enumerate(tokens):
                if i > 0:
                    await asyncio.sleep(self.config.tpot_ms / 1000)
                chunk = {
                    "object": "chat.completion.chunk",
                    "model": self.config.model,
                    "choices": [{"index": 0, "delta": {"content": token + " "}}],
                }
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if self.config.abort_mid_stream_after is None:
                await response.write(b"data: [DONE]\n\n")
            # else: truncated stream, no DONE — clients must classify as 'server' (§12.1)
        finally:
            self.requests_running -= 1
        await response.write_eof()
        return response


async def run_server(config: MockConfig, port: int) -> web.AppRunner:
    server = MockServer(config)
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock OpenAI-compatible server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--ttft-ms", type=float, default=50.0)
    parser.add_argument("--tpot-ms", type=float, default=5.0)
    parser.add_argument("--error-rate", type=float, default=0.0)
    args = parser.parse_args()
    config = MockConfig(ttft_ms=args.ttft_ms, tpot_ms=args.tpot_ms, error_rate=args.error_rate)

    async def _serve() -> None:
        await run_server(config, args.port)
        print(f"mock server on 127.0.0.1:{args.port} (ttft={config.ttft_ms}ms tpot={config.tpot_ms}ms)")
        await asyncio.Event().wait()

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
