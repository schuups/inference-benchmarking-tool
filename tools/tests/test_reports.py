"""M9 DoD: λ*, users estimate, and session boundary rules on a fixture DB
with analytically known answers."""

import pytest

from tools.reports import panels
from tools.testing.fixtures import (
    AGENTIC_WALL_MS,
    EXPECTED_POPULATION,
    RUN_ID,
    USER_RATES,
    WINDOW_LO_MS,
    build_fixture_db as _build_fixture,
)


@pytest.fixture()
def conn(tmp_path):
    db = _build_fixture(tmp_path / "run.db")
    yield db._conn
    db.close()


def test_lambda_star_is_highest_contiguous_passing_rate(conn):
    table, lambda_star = panels.slo_attainment(conn)
    assert lambda_star == 0.2
    failing = table[(table.rate_lambda == 0.4) & (table.scenario == "chat-short-turns")
                    & (table.metric == "ttft_ms")]
    assert not failing.passed.any()


def test_users_estimate_matches_analytical_values(conn):
    est = panels.users_estimate(conn, USER_RATES)
    assert est is not None
    by_class = est.set_index("scenario")
    for cls, expected in EXPECTED_POPULATION.items():
        assert by_class.loc[cls, "lambda_star"] == 0.2
        assert by_class.loc[cls, "supportable_user_population"] == pytest.approx(expected, rel=0.01)
    # Little's law: agentic 0.8/s × 30s wall = 24 concurrent active sessions
    assert by_class.loc["agentic-coding", "concurrent_active_sessions"] == pytest.approx(
        0.8 * AGENTIC_WALL_MS / 1000.0, rel=0.01
    )


def test_truncated_sessions_excluded_from_session_metrics(tmp_path):
    db = _build_fixture(tmp_path / "run.db")
    # one truncated agentic session at λ*: first turn only, no final_turn row
    db.insert(
        "requests",
        {
            "run_id": RUN_ID, "rate_lambda": 0.2, "request_id": 99_999,
            "session_idx": 99_999, "instance_id": "i0", "scenario": "agentic-coding",
            "turn_idx": 0, "final_turn": 0, "issued_at_ms": WINDOW_LO_MS + 1.0,
            "ttft_ms": 300.0, "tpot_ms": 30.0, "e2e_ms": 5000.0,
            "input_tokens": 15000, "output_tokens": 800, "success": 1, "error": None,
        },
    )
    sess = panels.sessions(db._conn)
    truncated = sess[sess.session_idx == 99_999]
    assert not truncated.complete.iloc[0]
    est = panels.users_estimate(db._conn, USER_RATES)
    assert est.set_index("scenario").loc["agentic-coding", "truncated_sessions"] >= 1
    db.close()


def test_per_class_percentiles(conn):
    df = panels.latency_percentiles(conn, "ttft_ms", per_class=True)
    chat = df[(df.scenario == "chat-short-turns") & (df.rate_lambda == 0.4)]
    assert chat.p95.iloc[0] == pytest.approx(900.0)
    agentic = df[(df.scenario == "agentic-coding") & (df.rate_lambda == 0.4)]
    assert agentic.p95.iloc[0] == pytest.approx(300.0)


def test_no_lambda_star_when_all_rates_fail(tmp_path):
    db = _build_fixture(tmp_path / "run.db")
    db._conn.execute("UPDATE requests SET ttft_ms = 5000.0 WHERE scenario = 'chat-short-turns'")
    db._conn.commit()
    table, lambda_star = panels.slo_attainment(db._conn)
    assert lambda_star is None
    assert panels.users_estimate(db._conn, USER_RATES) is None  # undefined per §14.1
    db.close()
