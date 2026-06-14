"""M2 DoD: arrival-process statistical properties under fixed seeds (§12.3)."""

import random
import statistics

import pytest

from tools.common.config import ArrivalProcess
from tools.benchmarker.load_gen.arrival import mmpp_state_rates, session_start_times

POISSON = ArrivalProcess(kind="poisson")
MMPP = ArrivalProcess(kind="burst_mmpp", burst_factor=5.0, mean_burst_s=20.0, mean_idle_s=180.0)


def _gaps(times):
    return [b - a for a, b in zip(times, times[1:])]


def test_poisson_mean_rate():
    times = session_start_times(POISSON, rate=2.0, duration_s=5000, rng=random.Random(42))
    assert len(times) / 5000 == pytest.approx(2.0, rel=0.05)


def test_poisson_cv_of_interarrivals_is_one():
    times = session_start_times(POISSON, rate=2.0, duration_s=5000, rng=random.Random(42))
    gaps = _gaps(times)
    cv = statistics.stdev(gaps) / statistics.mean(gaps)
    assert cv == pytest.approx(1.0, abs=0.05)


def test_poisson_deterministic_given_seed():
    a = session_start_times(POISSON, 1.0, 100, random.Random(7))
    b = session_start_times(POISSON, 1.0, 100, random.Random(7))
    c = session_start_times(POISSON, 1.0, 100, random.Random(8))
    assert a == b != c


def test_times_sorted_and_bounded():
    for proc in (POISSON, MMPP):
        times = session_start_times(proc, 1.0, 500, random.Random(3))
        assert times == sorted(times)
        assert all(0 <= t < 500 for t in times)


def test_mmpp_mean_rate_converges():
    times = session_start_times(MMPP, rate=2.0, duration_s=50_000, rng=random.Random(42))
    assert len(times) / 50_000 == pytest.approx(2.0, rel=0.10)


def _dispersion_index(times, duration_s, window_s=10):
    counts = [0] * int(duration_s // window_s)
    for t in times:
        counts[int(t // window_s)] += 1
    return statistics.variance(counts) / statistics.mean(counts)


def test_mmpp_burstier_than_poisson():
    mmpp_times = session_start_times(MMPP, 2.0, 20_000, random.Random(42))
    poisson_times = session_start_times(POISSON, 2.0, 20_000, random.Random(42))
    cv = statistics.stdev(_gaps(mmpp_times)) / statistics.mean(_gaps(mmpp_times))
    assert cv > 1.3  # inter-arrival CV above Poisson's 1 (§12.3)
    # the decisive burstiness signal: windowed index of dispersion (Poisson ≈ 1)
    assert _dispersion_index(poisson_times, 20_000) == pytest.approx(1.0, abs=0.3)
    assert _dispersion_index(mmpp_times, 20_000) > 5


def test_mmpp_state_rates_math():
    lb, li = mmpp_state_rates(rate=2.0, burst_factor=5.0, mean_burst_s=20.0, mean_idle_s=180.0)
    assert lb == pytest.approx(10.0)
    f = 20.0 / 200.0
    assert f * lb + (1 - f) * li == pytest.approx(2.0)  # long-run mean preserved


def test_mmpp_infeasible_parameters_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        mmpp_state_rates(rate=2.0, burst_factor=5.0, mean_burst_s=100.0, mean_idle_s=100.0)
