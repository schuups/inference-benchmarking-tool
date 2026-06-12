"""Arrival processes (SPECIFICATIONS.md §11.3).

λ counts **session starts** (sessions/s; §11.3 *What λ counts*). Both processes
produce a deterministic schedule of session-start times for one sweep step,
given a seeded RNG — reproducibility piggybacks on stdlib `random.Random`.

burst_mmpp is a two-state on/off Markov-Modulated Poisson Process at mean rate
λ. Parameterization from the benchmark YAML (§11.3): `burst_factor` B is the
peak-to-mean ratio (λ_burst = B·λ); `mean_burst_s` / `mean_idle_s` are the
exponential mean sojourn times of the two states. The idle rate follows from
keeping the long-run mean at λ:

    f       = mean_burst_s / (mean_burst_s + mean_idle_s)   (fraction of time in burst)
    λ_idle  = λ · (1 − f·B) / (1 − f)

which requires f·B < 1 — otherwise the burst state alone already exceeds the
target mean and no non-negative idle rate exists.
"""

from __future__ import annotations

import random

from tools.common.config import ArrivalProcess


def session_start_times(
    process: ArrivalProcess, rate: float, duration_s: float, rng: random.Random
) -> list[float]:
    """Session-start offsets (seconds, ascending) within [0, duration_s)."""
    if process.kind == "poisson":
        return _poisson(rate, duration_s, rng)
    return _burst_mmpp(
        rate,
        duration_s,
        rng,
        burst_factor=process.burst_factor,
        mean_burst_s=process.mean_burst_s,
        mean_idle_s=process.mean_idle_s,
    )


def _poisson(rate: float, duration_s: float, rng: random.Random) -> list[float]:
    times: list[float] = []
    t = rng.expovariate(rate)
    while t < duration_s:
        times.append(t)
        t += rng.expovariate(rate)
    return times


def mmpp_state_rates(rate: float, burst_factor: float, mean_burst_s: float, mean_idle_s: float) -> tuple[float, float]:
    """(λ_burst, λ_idle) keeping the long-run mean at `rate`; raises if infeasible."""
    f = mean_burst_s / (mean_burst_s + mean_idle_s)
    if f * burst_factor >= 1.0:
        raise ValueError(
            f"infeasible burst_mmpp: burst fraction {f:.3f} × burst_factor "
            f"{burst_factor} ≥ 1 — the burst state alone exceeds the target mean "
            "rate; increase mean_idle_s or lower burst_factor"
        )
    lambda_burst = burst_factor * rate
    lambda_idle = rate * (1 - f * burst_factor) / (1 - f)
    return lambda_burst, lambda_idle


def _burst_mmpp(
    rate: float,
    duration_s: float,
    rng: random.Random,
    *,
    burst_factor: float,
    mean_burst_s: float,
    mean_idle_s: float,
) -> list[float]:
    lambda_burst, lambda_idle = mmpp_state_rates(rate, burst_factor, mean_burst_s, mean_idle_s)
    times: list[float] = []
    t = 0.0
    in_burst = rng.random() < mean_burst_s / (mean_burst_s + mean_idle_s)
    while t < duration_s:
        sojourn = rng.expovariate(1 / (mean_burst_s if in_burst else mean_idle_s))
        state_end = min(t + sojourn, duration_s)
        state_rate = lambda_burst if in_burst else lambda_idle
        if state_rate > 0:
            arrival = t + rng.expovariate(state_rate)
            while arrival < state_end:
                times.append(arrival)
                arrival += rng.expovariate(state_rate)
        t = state_end
        in_burst = not in_burst
    return times
