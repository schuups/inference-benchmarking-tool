"""One sweep step: session-start schedule -> sessions -> request rows (§11).

Semantics implemented here:
- λ counts session starts (§11.3); each arrival draws its class from the mix
  weights (axis `mix`, §10.8) and consumes the next pool session of that class.
- `sequential` turns anchor on the previous turn's response + think time;
  `open_loop` turns anchor on the previous turn's send + think time (§10.7).
  Open-loop follow-ups carry whatever exchanges have completed by send time —
  the natural client behaviour when not waiting for responses.
- append_delta (§10.7): turn K+1's messages = prior transcript + new user turn.
- No new sessions after measurement end; in-flight sessions drain up to
  drain_timeout_s, then are cancelled — issued requests are recorded, the
  session is left without its final_turn row (truncated, §11.2/§12.2).
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp

from tools.common.config import ArrivalProcess

from .arrival import session_start_times
from .client import execute_request
from .pool import PoolSession
from .scraper import scrape_server_stats


@dataclass
class StepConfig:
    rate_lambda: float
    warmup_s: float
    measurement_s: float
    drain_timeout_s: float
    request_timeout_s: float
    arrival: ArrivalProcess
    routing: str  # random | session_affinity
    output_length_mode: str  # forced | natural
    seed: int
    endpoints: list[tuple[str, str]]  # (instance_id, base_url)
    model: str
    scrape_interval_s: float = 1.0
    lag_probe_interval_s: float = 0.05


@dataclass
class RequestRow:
    rate_lambda: float
    request_id: int
    session_idx: int
    scenario: str
    turn_idx: int
    final_turn: int
    issued_at_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float | None
    input_tokens: int
    output_tokens: int
    success: int
    error: str | None
    instance_id: str


@dataclass
class StepResult:
    requests: list[RequestRow] = field(default_factory=list)
    server_stats: list[dict] = field(default_factory=list)
    sessions_started: int = 0
    sessions_truncated: int = 0
    lag_max_ms: float = 0.0
    lag_warning: bool = False

    LAG_WARN_MS = 100.0


def _step_rng(seed: int, axis: str, rate: float) -> random.Random:
    digest = hashlib.blake2b(f"{seed}:{axis}:{rate}".encode(), digest_size=8)
    return random.Random(int.from_bytes(digest.digest(), "big"))


class _SessionRunner:
    def __init__(self, cfg: StepConfig, http: aiohttp.ClientSession, result: StepResult, t0: float):
        self._cfg = cfg
        self._http = http
        self._result = result
        self._t0 = t0
        self._route_rng = _step_rng(cfg.seed, "routing", cfg.rate_lambda)
        self._next_request_id = 0

    def _route(self, session_idx: int) -> tuple[str, str]:
        if self._cfg.routing == "session_affinity":
            return self._cfg.endpoints[session_idx % len(self._cfg.endpoints)]  # §11.4
        return self._cfg.endpoints[self._route_rng.randrange(len(self._cfg.endpoints))]

    async def run(self, sess: PoolSession) -> None:
        transcript: list[dict] = []  # completed exchanges, in turn order
        transcript_tokens = 0
        turn_tasks: list[asyncio.Task] = []

        async def run_turn(turn, send_at: float | None):
            nonlocal transcript, transcript_tokens
            if turn.think_time_ms and turn.turn_idx > 0:
                anchor_wait = (
                    send_at - time.perf_counter() if send_at is not None else turn.think_time_ms / 1000
                )
                if anchor_wait > 0:
                    await asyncio.sleep(anchor_wait)
            instance_id, url = self._route(sess.session_idx)
            messages = transcript + [{"role": "user", "content": turn.prompt_text}]
            input_tokens = transcript_tokens + turn.text_tokens
            issued_at = time.perf_counter()
            request_id = self._next_request_id
            self._next_request_id += 1
            outcome = await execute_request(
                self._http,
                url,
                model=self._cfg.model,
                messages=messages,
                max_tokens=turn.max_tokens,
                ignore_eos=self._cfg.output_length_mode == "forced",
                request_timeout_s=self._cfg.request_timeout_s,
            )
            self._result.requests.append(
                RequestRow(
                    rate_lambda=self._cfg.rate_lambda,
                    request_id=request_id,
                    session_idx=sess.session_idx,
                    scenario=sess.scenario,
                    turn_idx=turn.turn_idx,
                    final_turn=int(turn.turn_idx == len(sess.turns) - 1 and outcome.success == 1),
                    issued_at_ms=(issued_at - self._t0) * 1000,
                    ttft_ms=outcome.ttft_ms,
                    tpot_ms=outcome.tpot_ms,
                    e2e_ms=outcome.e2e_ms,
                    input_tokens=input_tokens,
                    output_tokens=outcome.output_tokens,
                    success=outcome.success,
                    error=outcome.error,
                    instance_id=instance_id,
                )
            )
            if outcome.success:
                transcript = messages + [{"role": "assistant", "content": outcome.output_text}]
                transcript_tokens = input_tokens + outcome.output_tokens

        if sess.mode == "sequential":
            for turn in sess.turns:
                await run_turn(turn, send_at=None)
        else:  # open_loop: sends anchored on previous send + think (§10.7)
            send_at = time.perf_counter()
            for turn in sess.turns:
                if turn.turn_idx > 0:
                    send_at += (turn.think_time_ms or 0.0) / 1000
                turn_tasks.append(
                    asyncio.create_task(
                        run_turn(turn, send_at=send_at if turn.turn_idx > 0 else None)
                    )
                )
            await asyncio.gather(*turn_tasks)


async def _lag_guard(result: StepResult, interval_s: float) -> None:
    """Event-loop lag monitor — client-side saturation detector (M2 DoD)."""
    while True:
        before = time.perf_counter()
        await asyncio.sleep(interval_s)
        lag_ms = (time.perf_counter() - before - interval_s) * 1000
        result.lag_max_ms = max(result.lag_max_ms, lag_ms)
        if lag_ms > StepResult.LAG_WARN_MS:
            result.lag_warning = True


class PoolExhaustedError(RuntimeError):
    """§10.4: num_prompts must outlast the request budget; never recycle prompts."""


async def run_step(
    pool: dict[str, deque[PoolSession]],
    weights: dict[str, float],
    cfg: StepConfig,
) -> StepResult:
    result = StepResult()
    schedule = session_start_times(
        cfg.arrival, cfg.rate_lambda, cfg.warmup_s + cfg.measurement_s,
        _step_rng(cfg.seed, "arrival", cfg.rate_lambda),
    )
    mix_rng = _step_rng(cfg.seed, "mix", cfg.rate_lambda)
    classes = sorted(weights)
    cumulative: list[tuple[float, str]] = []
    acc = 0.0
    for slug in classes:
        acc += weights[slug]
        cumulative.append((acc, slug))

    def draw_class() -> str:
        x = mix_rng.random() * acc
        return next(slug for bound, slug in cumulative if x <= bound)

    connector = aiohttp.TCPConnector(limit=0)  # no client-side connection cap
    async with aiohttp.ClientSession(connector=connector) as http:
        t0 = time.perf_counter()
        runner = _SessionRunner(cfg, http, result, t0)
        guard = asyncio.create_task(_lag_guard(result, cfg.lag_probe_interval_s))
        scraper = asyncio.create_task(
            scrape_server_stats(http, cfg.endpoints, cfg.rate_lambda, cfg.scrape_interval_s, result.server_stats)
        )
        session_tasks: list[asyncio.Task] = []
        try:
            for start_offset in schedule:
                delay = t0 + start_offset - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                slug = draw_class()
                queue = pool.get(slug)
                if not queue:
                    raise PoolExhaustedError(
                        f"prompt pool exhausted for class '{slug}' at λ={cfg.rate_lambda} "
                        "— increase dataset_config.num_prompts (§10.4)"
                    )
                session_tasks.append(asyncio.create_task(runner.run(queue.popleft())))
                result.sessions_started += 1
            # drain (§11.2): no new sessions; in-flight complete up to the deadline
            if session_tasks:
                done, pending = await asyncio.wait(session_tasks, timeout=cfg.drain_timeout_s)
                result.sessions_truncated = len(pending)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            guard.cancel()
            scraper.cancel()
            await asyncio.gather(guard, scraper, return_exceptions=True)
    result.requests.sort(key=lambda r: r.issued_at_ms)
    return result
