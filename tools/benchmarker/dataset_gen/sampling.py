"""Deterministic sampling (SPECIFICATIONS.md §10.6, §10.8).

Per-class, per-axis sub-seeds via blake2b(f"{seed}:{scenario}:{axis}") and one
run-level `mix` axis. All randomness goes through stdlib `random.Random`, which
is reproducible across platforms — the §10.8 byte-for-byte contract depends on it.
"""

from __future__ import annotations

import hashlib
import math
import random

from .registry import Distribution

THINKING_MEAN_FACTOR = 2.5  # §10.6 thinking widening
THINKING_SIGMA_FACTOR = 1.5


def sub_seed(seed: int, scenario: str, axis: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{scenario}:{axis}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def class_rng(seed: int, scenario: str, axis: str) -> random.Random:
    return random.Random(sub_seed(seed, scenario, axis))


def widen_for_thinking(dist: Distribution) -> Distribution:
    """§10.6: mean ×2.5, sigma/stdev ×1.5; fixed value ×2.5; min/max kept as clamps."""
    p = dict(dist.params)
    if dist.distribution == "fixed":
        p["value"] = p["value"] * THINKING_MEAN_FACTOR
    else:
        p["mean"] = p["mean"] * THINKING_MEAN_FACTOR
        for key in ("sigma", "stdev"):
            if key in p:
                p[key] = p[key] * THINKING_SIGMA_FACTOR
    return Distribution(distribution=dist.distribution, params=p)


def sample(dist: Distribution, rng: random.Random) -> float:
    p = dist.params
    if dist.distribution == "fixed":
        value = p["value"]
    elif dist.distribution == "lognormal":
        # params.mean is the desired linear-space mean: mu = ln(mean) - sigma^2/2.
        sigma = p["sigma"]
        mu = math.log(p["mean"]) - sigma**2 / 2
        value = rng.lognormvariate(mu, sigma)
    else:  # normal
        value = rng.gauss(p["mean"], p.get("stdev", p.get("sigma")))
    lo, hi = p.get("min"), p.get("max")
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return value


def sample_int(dist: Distribution, rng: random.Random, minimum: int = 1) -> int:
    return max(minimum, round(sample(dist, rng)))


def expected_mean(dist: Distribution, probes: int = 10_000) -> float:
    """Deterministic empirical mean (fixed probe seed) — used for the §10.4
    num_prompts split and the manifest's expected_request_share."""
    if dist.distribution == "fixed":
        return float(dist.params["value"])
    rng = random.Random(0xC0FFEE)
    return sum(sample(dist, rng) for _ in range(probes)) / probes
