"""Mock-server ground truth: latency shape, canned answers, metrics, faults."""

import json
import time

import aiohttp
import pytest
import pytest_asyncio

from tools.testing.mock_openai_server import MockConfig, run_server

PORT = 8779


@pytest_asyncio.fixture()
async def server():
    config = MockConfig(
        ttft_ms=80.0,
        tpot_ms=5.0,
        canned=[("capital of France", "Paris is the capital of France .")],
    )
    runner = await run_server(config, PORT)
    yield config
    await runner.cleanup()


async def _stream(payload: dict) -> tuple[float, list[float], str, bool]:
    """Returns (ttft_s, inter_token_gaps_s, text, saw_done)."""
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    text, stamps, done = [], [], False
    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            assert resp.status == 200
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    done = True
                    break
                stamps.append(time.perf_counter())
                text.append(json.loads(data)["choices"][0]["delta"]["content"])
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return stamps[0] - start, gaps, "".join(text), done


@pytest.mark.asyncio
async def test_latency_ground_truth(server):
    ttft, gaps, _, done = await _stream(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 30, "ignore_eos": True}
    )
    assert done
    assert ttft == pytest.approx(0.080, abs=0.040)  # configured TTFT
    assert sum(gaps) / len(gaps) == pytest.approx(0.005, abs=0.004)  # configured TPOT


@pytest.mark.asyncio
async def test_canned_answer_mode(server):
    _, _, text, _ = await _stream(
        {"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 50}
    )
    assert text.split() == ["Paris", "is", "the", "capital", "of", "France", "."]


@pytest.mark.asyncio
async def test_ignore_eos_forces_exact_token_count(server):
    _, gaps, text, _ = await _stream(
        {"messages": [{"role": "user", "content": "capital of France please"}], "max_tokens": 24, "ignore_eos": True}
    )
    assert len(text.split()) == 24  # §11.6 forced mode: exactly max_tokens tokens


@pytest.mark.asyncio
async def test_metrics_endpoint(server):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{PORT}/metrics") as resp:
            body = await resp.text()
    assert "vllm:num_requests_running" in body
    assert "vllm:gpu_cache_usage_perc" in body


@pytest.mark.asyncio
async def test_fault_injection():
    runner = await run_server(MockConfig(error_rate=1.0, error_status=429), PORT + 1)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{PORT + 1}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "x"}]},
            ) as resp:
                assert resp.status == 429
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_mid_stream_abort_truncates_without_done():
    runner = await run_server(MockConfig(ttft_ms=5, tpot_ms=1, abort_mid_stream_after=3), PORT + 2)
    try:
        url = f"http://127.0.0.1:{PORT + 2}/v1/chat/completions"
        tokens, done = [], False
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 20}) as resp:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if line == "data: [DONE]":
                        done = True
                    elif line.startswith("data: "):
                        tokens.append(line)
        assert len(tokens) == 3 and not done  # truncated stream → §13.1 class 'server'
    finally:
        await runner.cleanup()
