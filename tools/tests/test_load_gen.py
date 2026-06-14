"""M2 DoD: scheduler semantics, error taxonomy, scraper, readiness, primer.

Timing assertions use generous tolerances — they verify *ordering and shape*,
not microsecond precision.
"""

import json
from collections import deque

import aiohttp
import pytest
import pytest_asyncio

from tools.common.config import ArrivalProcess
from tools.benchmarker.load_gen.pool import PoolSession, Turn, load_pool
from tools.benchmarker.load_gen.readiness import (
    parse_model_load,
    run_primer,
)
from tools.benchmarker.load_gen.scheduler import (
    PoolExhaustedError,
    StepConfig,
    run_step,
)
from tools.benchmarker.load_gen.scraper import parse_metrics
from tools.testing.mock_openai_server import MockConfig, run_server

BASE_PORT = 8810


def _session(idx: int, scenario: str, mode: str, n_turns: int, think_ms: float = 30.0) -> PoolSession:
    turns = tuple(
        Turn(
            scenario=scenario,
            session_idx=idx,
            turn_idx=t,
            prompt_text=f"[session-{idx:06d}] body" if t == 0 else f"follow-up {t}",
            text_tokens=10,
            max_tokens=5,
            think_time_ms=None if t == 0 else think_ms,
        )
        for t in range(n_turns)
    )
    return PoolSession(session_idx=idx, scenario=scenario, mode=mode, turns=turns)


def _pool(n_sessions: int = 40, mode: str = "sequential", n_turns: int = 3) -> dict:
    return {"test-class": deque(_session(i, "test-class", mode, n_turns) for i in range(n_sessions))}


def _cfg(port: int, **overrides) -> StepConfig:
    defaults = dict(
        rate_lambda=8.0,
        warmup_s=0.3,
        measurement_s=0.7,
        drain_timeout_s=5.0,
        request_timeout_s=5.0,
        arrival=ArrivalProcess(kind="poisson"),
        routing="random",
        output_length_mode="forced",
        seed=1234,
        endpoints=[("i0", f"http://127.0.0.1:{port}")],
        model="mock/model",
        scrape_interval_s=0.2,
    )
    defaults.update(overrides)
    return StepConfig(**defaults)


@pytest_asyncio.fixture()
async def mock(unused_tcp_port_factory=None):
    runner = await run_server(MockConfig(ttft_ms=10, tpot_ms=2), BASE_PORT)
    yield BASE_PORT
    await runner.cleanup()


# ----------------------------------------------------------------- scheduler


@pytest.mark.asyncio
async def test_step_end_to_end(mock):
    result = await run_step(_pool(), {"test-class": 1.0}, _cfg(mock))
    assert result.sessions_started > 0
    assert result.requests and all(r.success == 1 for r in result.requests)
    assert result.sessions_truncated == 0
    # all sessions completed: every started session has a final_turn row
    finals = sum(r.final_turn for r in result.requests)
    assert finals == result.sessions_started
    # TTFT matches the mock's ground truth
    ttfts = [r.ttft_ms for r in result.requests]
    assert 5 < sum(ttfts) / len(ttfts) < 60
    # issued_at within step bounds; rows sorted
    assert all(0 <= r.issued_at_ms for r in result.requests)
    assert [r.issued_at_ms for r in result.requests] == sorted(r.issued_at_ms for r in result.requests)
    # server_stats captured during the step
    assert result.server_stats and any(s["requests_running"] is not None for s in result.server_stats)


@pytest.mark.asyncio
async def test_sequential_turns_never_overlap(mock):
    result = await run_step(
        _pool(mode="sequential", n_turns=4), {"test-class": 1.0}, _cfg(mock, rate_lambda=4.0)
    )
    by_session = {}
    for r in result.requests:
        by_session.setdefault(r.session_idx, []).append(r)
    for rows in by_session.values():
        rows.sort(key=lambda r: r.turn_idx)
        for prev, nxt in zip(rows, rows[1:]):
            # next send >= previous completion (+ think time, minus tolerance)
            assert nxt.issued_at_ms >= prev.issued_at_ms + prev.e2e_ms - 5


@pytest.mark.asyncio
async def test_open_loop_does_not_wait_for_completion():
    # slow decode: each response takes ~400ms; open_loop think is 50ms
    runner = await run_server(MockConfig(ttft_ms=10, tpot_ms=20), BASE_PORT + 1)
    try:
        pool = {
            "test-class": deque(
                [
                    PoolSession(
                        session_idx=0,
                        scenario="test-class",
                        mode="open_loop",
                        turns=tuple(
                            Turn("test-class", 0, t, f"t{t}", 5, 20, None if t == 0 else 50.0)
                            for t in range(3)
                        ),
                    )
                ]
            )
        }
        cfg = _cfg(BASE_PORT + 1, rate_lambda=2.0, warmup_s=0.05, measurement_s=0.3)
        result = await run_step(pool, {"test-class": 1.0}, cfg)
        rows = sorted(result.requests, key=lambda r: r.turn_idx)
        assert len(rows) == 3
        # turn 1 was sent ~50ms after turn 0's send — far before turn 0's ~400ms e2e
        gap = rows[1].issued_at_ms - rows[0].issued_at_ms
        assert gap < rows[0].e2e_ms - 50
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_session_affinity_routing():
    runners = [await run_server(MockConfig(ttft_ms=5, tpot_ms=1), BASE_PORT + 2 + i) for i in range(2)]
    try:
        endpoints = [(f"i{i}", f"http://127.0.0.1:{BASE_PORT + 2 + i}") for i in range(2)]
        cfg = _cfg(BASE_PORT + 2, routing="session_affinity", endpoints=endpoints)
        result = await run_step(_pool(n_turns=3), {"test-class": 1.0}, cfg)
        for r in result.requests:
            assert r.instance_id == f"i{r.session_idx % 2}"  # §12.4
    finally:
        for r in runners:
            await r.cleanup()


@pytest.mark.asyncio
async def test_mix_draw_matches_weights(mock):
    pool = {
        "a": deque(_session(i, "a", "sequential", 1) for i in range(400)),
        "b": deque(_session(1000 + i, "b", "sequential", 1) for i in range(400)),
    }
    cfg = _cfg(mock, rate_lambda=120.0, warmup_s=0.5, measurement_s=1.5)
    result = await run_step(pool, {"a": 0.8, "b": 0.2}, cfg)
    counts = {"a": 0, "b": 0}
    for r in result.requests:
        if r.turn_idx == 0:
            counts[r.scenario] += 1
    share_a = counts["a"] / (counts["a"] + counts["b"])
    assert share_a == pytest.approx(0.8, abs=0.08)


@pytest.mark.asyncio
async def test_pool_exhaustion_aborts(mock):
    pool = {"test-class": deque(_session(i, "test-class", "sequential", 1) for i in range(2))}
    with pytest.raises(PoolExhaustedError, match="num_prompts"):
        await run_step(pool, {"test-class": 1.0}, _cfg(mock, rate_lambda=50.0))


@pytest.mark.asyncio
async def test_drain_truncates_slow_sessions(mock):
    # think time far beyond the drain deadline: follow-ups can never fire
    pool = {"test-class": deque(_session(i, "test-class", "sequential", 2, think_ms=60_000) for i in range(30))}
    cfg = _cfg(mock, rate_lambda=6.0, warmup_s=0.1, measurement_s=0.4, drain_timeout_s=0.3)
    result = await run_step(pool, {"test-class": 1.0}, cfg)
    assert result.sessions_truncated == result.sessions_started > 0
    assert all(r.final_turn == 0 for r in result.requests)  # nobody reached the last turn
    assert all(r.turn_idx == 0 for r in result.requests)  # only first turns were issued


@pytest.mark.asyncio
async def test_error_taxonomy_http_and_connection(mock):
    # http_429 from fault injection
    runner = await run_server(MockConfig(error_rate=1.0, error_status=429), BASE_PORT + 5)
    try:
        result = await run_step(
            _pool(n_turns=1), {"test-class": 1.0},
            _cfg(BASE_PORT + 5, rate_lambda=10.0, warmup_s=0.1, measurement_s=0.3),
        )
        assert result.requests and all(r.success == 0 for r in result.requests)
        assert all(r.error.startswith("http_429") for r in result.requests)
    finally:
        await runner.cleanup()
    # connection refusal: nothing listens on the port
    result = await run_step(
        _pool(n_turns=1), {"test-class": 1.0},
        _cfg(BASE_PORT + 6, rate_lambda=10.0, warmup_s=0.1, measurement_s=0.3),
    )
    assert result.requests and all(r.error.startswith("connection") for r in result.requests)


@pytest.mark.asyncio
async def test_error_taxonomy_ttft_timeout_and_truncated_stream():
    runner = await run_server(MockConfig(ttft_ms=500, tpot_ms=1), BASE_PORT + 7)
    try:
        cfg = _cfg(BASE_PORT + 7, rate_lambda=6.0, warmup_s=0.1, measurement_s=0.3, request_timeout_s=0.1)
        result = await run_step(_pool(n_turns=1), {"test-class": 1.0}, cfg)
        assert result.requests and all(r.error.startswith("timeout:ttft") for r in result.requests)
    finally:
        await runner.cleanup()
    runner = await run_server(MockConfig(ttft_ms=5, tpot_ms=1, abort_mid_stream_after=2), BASE_PORT + 8)
    try:
        cfg = _cfg(BASE_PORT + 8, rate_lambda=6.0, warmup_s=0.1, measurement_s=0.3)
        result = await run_step(_pool(n_turns=1), {"test-class": 1.0}, cfg)
        assert result.requests and all(r.error == "server:truncated-stream" for r in result.requests)
    finally:
        await runner.cleanup()


# ---------------------------------------------------------- pool round-trip


def test_pool_loader_round_trip(tmp_path):
    records = [
        {"scenario": "c", "session_idx": 0, "session_mode": "sequential", "turn_idx": 1,
         "prompt_text": "f", "text_tokens": 1, "max_tokens": 2, "think_time_ms": 9.0},
        {"scenario": "c", "session_idx": 0, "session_mode": "sequential", "turn_idx": 0,
         "prompt_text": "[session-000000] x", "text_tokens": 3, "max_tokens": 2, "think_time_ms": None},
    ]
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    pool = load_pool(path)
    sess = pool["c"][0]
    assert sess.mode == "sequential"
    assert [t.turn_idx for t in sess.turns] == [0, 1]  # sorted regardless of file order


# ------------------------------------------------- scraper / readiness / primer


def test_parse_metrics():
    body = (
        "vllm:num_requests_running 3\nvllm:num_requests_waiting 7\n"
        "vllm:gpu_cache_usage_perc 0.42\n"
    )
    parsed = parse_metrics(body)
    assert parsed["requests_running"] == 3
    assert parsed["requests_waiting"] == 7
    assert parsed["gpu_cache_pct"] == pytest.approx(42.0)
    assert parsed["spec_accept_rate"] is None


def test_parse_model_load_fixture():
    log = (
        "INFO 06-12 [model_runner.py] Model loading took 123.45 GiB and 25.33 seconds\n"
        "INFO 06-12 [llm_engine.py] init engine (profile, create kv cache, warmup model) took 61.20 seconds\n"
        "INFO 06-12 [model_runner.py] Graph capturing finished in 23 secs\n"
    )
    parsed = parse_model_load(log)
    assert parsed["model_load_weights_s"] == pytest.approx(25.33)
    assert parsed["model_load_engine_init_s"] == pytest.approx(61.20)
    assert parsed["model_load_cuda_graph_capture_s"] == pytest.approx(23)
    assert parsed["model_load_inductor_compile_s"] is None  # NULL per §10.2


@pytest.mark.asyncio
async def test_primer_steady_state_no_warning(mock):
    async with aiohttp.ClientSession() as http:
        result = await run_primer(
            http, f"http://127.0.0.1:{mock}", model="mock/model", prompt_tokens=100, probe_tokens=10
        )
    assert result.warning is None
    assert result.probe1_ttft_ms and result.probe2_ttft_ms


@pytest.mark.asyncio
async def test_primer_missed_target_warns():
    # primer (request 1) and probe 1 (request 2) still pay the cold cost; probe 2 is steady-state
    config = MockConfig(ttft_ms=10, tpot_ms=1, slow_first_n=2, slow_ttft_ms=400)
    runner = await run_server(config, BASE_PORT + 10)
    try:
        async with aiohttp.ClientSession() as http:
            result = await run_primer(
                http, f"http://127.0.0.1:{BASE_PORT + 10}", model="mock/model",
                prompt_tokens=50, probe_tokens=10,
            )
        assert result.warning and "primer missed" in result.warning
    finally:
        await runner.cleanup()
