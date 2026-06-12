"""Report-panel data functions (SPECIFICATIONS.md §12.2, §12.4, §14.1).

Pure functions over a per-run SQLite DB (§13) returning pandas DataFrames /
scalars — the notebook cells stay thin, and the load-bearing math (λ*, the
supportable-users estimate, session boundary rules) is unit-testable without
executing a notebook.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import pandas as pd

PERCENTILES = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}


def connect(db_path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def experiment(conn: sqlite3.Connection) -> dict:
    row = pd.read_sql_query("SELECT * FROM experiments", conn).iloc[0].to_dict()
    for key in ("scenario_mix", "scenario_manifest", "slos", "quality_eval", "rate_levels"):
        if row.get(key):
            row[key] = json.loads(row[key])
    return row


def measured_requests(conn: sqlite3.Connection) -> pd.DataFrame:
    """Requests issued within the measurement window (§11.2/§12.2 boundary)."""
    exp = experiment(conn)
    lo = exp["warmup_s"] * 1000.0
    hi = (exp["warmup_s"] + exp["measurement_s"]) * 1000.0
    df = pd.read_sql_query("SELECT * FROM requests", conn)
    return df[(df.issued_at_ms >= lo) & (df.issued_at_ms < hi)]


def latency_percentiles(conn, metric: str = "ttft_ms", per_class: bool = False) -> pd.DataFrame:
    """p50/p90/p95/p99 of `metric` vs λ over measured successful requests."""
    df = measured_requests(conn)
    df = df[df.success == 1]
    keys = ["rate_lambda", "scenario"] if per_class else ["rate_lambda"]
    rows = []
    for group_key, group in df.groupby(keys):
        entry = dict(zip(keys, group_key if isinstance(group_key, tuple) else (group_key,)))
        for name, q in PERCENTILES.items():
            entry[name] = group[metric].quantile(q)
        entry["n"] = len(group)
        rows.append(entry)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def error_rates(conn, per_class: bool = False) -> pd.DataFrame:
    df = measured_requests(conn)
    keys = ["rate_lambda", "scenario"] if per_class else ["rate_lambda"]
    out = df.groupby(keys).agg(
        n=("success", "size"), failed=("success", lambda s: int((s == 0).sum()))
    ).reset_index()
    out["error_rate_pct"] = 100.0 * out.failed / out.n
    return out


def sessions(conn) -> pd.DataFrame:
    """§12.2 session-level metrics over *complete* sessions (final_turn present,
    all turns succeeded); truncated sessions are counted, not aggregated."""
    df = pd.read_sql_query("SELECT * FROM requests", conn)
    grouped = df.groupby(["rate_lambda", "scenario", "session_idx"])
    rows = []
    for (rate, scenario, session_idx), turns in grouped:
        complete = bool(turns.final_turn.max()) and bool((turns.success == 1).all())
        e2e = (turns.issued_at_ms + turns.e2e_ms.fillna(0)).max() - turns.issued_at_ms.min()
        rows.append(
            {
                "rate_lambda": rate, "scenario": scenario, "session_idx": session_idx,
                "session_turns": len(turns), "session_e2e_ms": e2e,
                "session_input_tokens": turns.input_tokens.sum(),
                "session_output_tokens": turns.output_tokens.sum(),
                "session_success": int((turns.success == 1).all()),
                "complete": complete,
                "started_ms": turns.issued_at_ms.min(),
            }
        )
    return pd.DataFrame(rows)


def session_metric_percentile(sess: pd.DataFrame, rate: float, scenario: str, percentile: str) -> float | None:
    pool = sess[(sess.rate_lambda == rate) & (sess.scenario == scenario) & sess.complete]
    if pool.empty:
        return None
    return float(pool.session_e2e_ms.quantile(PERCENTILES[percentile]))


# ------------------------------------------------------------------ §12.4 λ*


@dataclass
class SLOOutcome:
    rate_lambda: float
    scenario: str
    metric: str
    percentile: str | None
    threshold: float
    measured: float | None
    passed: bool


def slo_attainment(conn) -> tuple[pd.DataFrame, float | None]:
    """Per-objective pass/fail per λ + λ* (highest λ meeting ALL objectives)."""
    exp = experiment(conn)
    slos = exp.get("slos") or []
    classes = [m["scenario"] for m in exp["scenario_mix"]]
    sess = sessions(conn)
    outcomes: list[SLOOutcome] = []
    for rate in sorted(exp["rate_levels"]):
        for slo in slos:
            targets = classes if slo["scenario"] == "all" else [slo["scenario"]]
            for cls in targets:
                measured = _slo_measure(conn, sess, rate, cls, slo)
                passed = measured is not None and measured <= slo["threshold"]
                outcomes.append(
                    SLOOutcome(rate, cls, slo["metric"], slo.get("percentile"),
                               slo["threshold"], measured, passed)
                )
    table = pd.DataFrame([o.__dict__ for o in outcomes])
    lambda_star = None
    if not table.empty:
        for rate in sorted(exp["rate_levels"]):
            step = table[table.rate_lambda == rate]
            if not step.empty and step.passed.all():
                lambda_star = rate
            else:
                break  # λ* is the highest *contiguous* passing rate
    return table, lambda_star


def _slo_measure(conn, sess, rate, cls, slo) -> float | None:
    metric = slo["metric"]
    if metric == "error_rate_pct":
        df = error_rates(conn, per_class=True)
        row = df[(df.rate_lambda == rate) & (df.scenario == cls)]
        return float(row.error_rate_pct.iloc[0]) if not row.empty else None
    if metric == "session_e2e_ms":
        return session_metric_percentile(sess, rate, cls, slo["percentile"])
    df = measured_requests(conn)
    pool = df[(df.rate_lambda == rate) & (df.scenario == cls) & (df.success == 1)]
    col = pool[metric].dropna()
    if col.empty:
        return None
    return float(col.quantile(PERCENTILES[slo["percentile"]]))


# ------------------------------------------------------ §14.1 users estimate


def users_estimate(conn, sessions_per_user_per_hour: dict[str, float]) -> pd.DataFrame | None:
    """Notebook-only λ*→users translation (§14.1).

    Per class at λ*: measured session-start throughput and mean wall-time of
    complete sessions →
      population        = throughput × 3600 / sessions_per_user_per_hour
      concurrent_active = throughput × mean session wall-time   (Little's law)
    """
    exp = experiment(conn)
    _, lambda_star = slo_attainment(conn)
    if lambda_star is None:
        return None
    sess = sessions(conn)
    window_s = exp["measurement_s"]
    lo = exp["warmup_s"] * 1000.0
    hi = (exp["warmup_s"] + exp["measurement_s"]) * 1000.0
    rows = []
    for mix_entry in exp["scenario_mix"]:
        cls = mix_entry["scenario"]
        pool = sess[(sess.rate_lambda == lambda_star) & (sess.scenario == cls)]
        started = pool[(pool.started_ms >= lo) & (pool.started_ms < hi)]
        throughput = len(started) / window_s  # sessions/s at λ*
        complete = pool[pool.complete]
        mean_wall_s = complete.session_e2e_ms.mean() / 1000.0 if not complete.empty else None
        rate = sessions_per_user_per_hour.get(cls)
        rows.append(
            {
                "scenario": cls,
                "lambda_star": lambda_star,
                "session_throughput_per_s": throughput,
                "mean_session_wall_s": mean_wall_s,
                "sessions_per_user_per_hour": rate,
                "supportable_user_population": (
                    throughput * 3600.0 / rate if rate else None
                ),
                "concurrent_active_sessions": (
                    throughput * mean_wall_s if mean_wall_s is not None else None
                ),
                "truncated_sessions": int((~pool.complete).sum()),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------- other panels


def prechecks_table(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM system_prechecks", conn)


def model_load_table(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM instances", conn)


def quality_table(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM quality_evals", conn)


def hardware_summary(conn) -> pd.DataFrame:
    """Mean per-signal per λ — the §14.1 headroom overlay source."""
    df = pd.read_sql_query("SELECT * FROM hardware_stats", conn)
    if df.empty:
        return df
    signals = [c for c in df.columns if c not in
               ("run_id", "instance_id", "rate_lambda", "ts", "gpu_index")]
    return df.groupby(["rate_lambda", df.gpu_index.notna()])[signals].mean(numeric_only=True).reset_index()


def raw_rate_table(conn) -> pd.DataFrame:
    ttft = latency_percentiles(conn, "ttft_ms")
    err = error_rates(conn)
    return ttft.merge(err[["rate_lambda", "error_rate_pct"]], on="rate_lambda")
