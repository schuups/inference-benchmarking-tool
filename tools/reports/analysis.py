"""Report analytics (SPECIFICATIONS.md §14.1) — the testable core of M9.

Pure pandas/sqlite functions the template notebook presents. All latency/SLO math
operates on the **measurement phase** only (§11.2): requests whose `issued_at_ms`
(ms from sweep-step start) falls in `[warmup_s, warmup_s + measurement_s)`.

Key derived quantities (asserted in tests against a known fixture):
- λ* (§12.4): the highest swept λ at which every declared SLO holds simultaneously.
- supportable users (§14.1): per-class session throughput ÷ per-user session rate,
  plus Little's-law concurrent sessions.
- capacity-vs-quality (§12.5/§14.1): users-at-λ* paired with quality scores across
  the experiment's deployment configs (run_ids).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tools.common.results_db import SCHEMA, json_columns

_PERCENTILES = (50, 90, 95, 99)
# JSON-encoded columns of the experiments row to decode on load — derived from
# the schema's single source of truth (results_db.json_columns), never re-listed.
_JSON_COLS = json_columns("experiments")


@dataclass
class ReportData:
    run_id: str
    model: str
    backend: str
    backend_config: dict
    scenario_mix: list
    manifest: dict
    slos: list
    quality_eval: dict | None
    rate_levels: list
    warmup_s: int
    measurement_s: int
    requests: pd.DataFrame
    instances: pd.DataFrame
    server_stats: pd.DataFrame
    hardware_stats: pd.DataFrame
    system_prechecks: pd.DataFrame
    quality_evals: pd.DataFrame


# --------------------------------------------------------------------- loading


def list_runs(db_path: Path | str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT run_id FROM experiments ORDER BY created_at")]
    finally:
        conn.close()


def load_run(db_path: Path | str, run_id: str | None = None) -> ReportData:
    conn = sqlite3.connect(db_path)
    try:
        if run_id is None:
            runs = [r[0] for r in conn.execute("SELECT run_id FROM experiments")]
            if len(runs) != 1:
                raise ValueError(f"{db_path} has {len(runs)} runs; pass run_id ({runs})")
            run_id = runs[0]
        cur = conn.execute("SELECT * FROM experiments WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"run_id {run_id!r} not found in {db_path}")
        exp = dict(zip([c[0] for c in cur.description], row))
        for col in _JSON_COLS:
            exp[col] = json.loads(exp[col]) if exp.get(col) else None
        tables = {
            name: pd.read_sql_query(f"SELECT * FROM {name} WHERE run_id=?", conn, params=(run_id,))
            for name in SCHEMA
            if name != "experiments"
        }
    finally:
        conn.close()
    return ReportData(
        run_id=run_id, model=exp["model"], backend=exp["backend"],
        backend_config=exp["backend_config"] or {}, scenario_mix=exp["scenario_mix"] or [],
        manifest=exp["scenario_manifest"] or {}, slos=exp["slos"] or [],
        quality_eval=exp["quality_eval"], rate_levels=exp["rate_levels"] or [],
        warmup_s=exp["warmup_s"], measurement_s=exp["measurement_s"],
        requests=tables["requests"], instances=tables["instances"],
        server_stats=tables["server_stats"], hardware_stats=tables["hardware_stats"],
        system_prechecks=tables["system_prechecks"], quality_evals=tables["quality_evals"],
    )


# ----------------------------------------------------------- measurement phase


def measurement_requests(report: ReportData) -> pd.DataFrame:
    """Requests in the measurement window [warmup_s, warmup_s+measurement_s) (§11.2)."""
    lo = report.warmup_s * 1000.0
    hi = (report.warmup_s + report.measurement_s) * 1000.0
    req = report.requests
    return req[(req["issued_at_ms"] >= lo) & (req["issued_at_ms"] < hi)].copy()


# ------------------------------------------------------------------- latencies


def latency_vs_lambda(
    req: pd.DataFrame, metric: str, percentiles=_PERCENTILES, by_scenario: bool = False
) -> pd.DataFrame:
    """Percentiles of `metric` over successful requests, per λ (and per class)."""
    ok = req[req["success"] == 1].dropna(subset=[metric])
    keys = ["rate_lambda"] + (["scenario"] if by_scenario else [])
    rows = []
    for key, grp in ok.groupby(keys):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for p in percentiles:
            rec[f"p{p}"] = float(grp[metric].quantile(p / 100))
        rec["n"] = int(len(grp))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True) if rows else pd.DataFrame(
        columns=[*keys, *(f"p{p}" for p in percentiles), "n"]
    )


def failure_rate_vs_lambda(req: pd.DataFrame, by_scenario: bool = False) -> pd.DataFrame:
    keys = ["rate_lambda"] + (["scenario"] if by_scenario else [])
    rows = []
    for key, grp in req.groupby(keys):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        total = len(grp)
        rec["error_rate_pct"] = 100.0 * int((grp["success"] == 0).sum()) / total if total else 0.0
        rec["n"] = int(total)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True) if rows else pd.DataFrame(
        columns=[*keys, "error_rate_pct", "n"]
    )


# --------------------------------------------------------------- session metrics


def session_metrics(req: pd.DataFrame) -> pd.DataFrame:
    """Per-session derived metrics (§12.2) over **completed** sessions only — those
    whose `final_turn==1` row is present. Grouped per class so one class's sessions
    never dilute another's (§14.1)."""
    if req.empty:
        return pd.DataFrame(
            columns=["rate_lambda", "scenario", "session_idx", "session_e2e_ms",
                     "session_turns", "session_success"]
        )
    df = req.copy()
    df["end_ms"] = df["issued_at_ms"] + df["e2e_ms"].fillna(0.0)
    keys = ["rate_lambda", "scenario", "session_idx"]
    agg = df.groupby(keys).agg(
        start_ms=("issued_at_ms", "min"),
        end_ms=("end_ms", "max"),
        session_turns=("turn_idx", "count"),
        session_success=("success", "min"),
        completed=("final_turn", "max"),
    ).reset_index()
    agg = agg[agg["completed"] == 1]  # §12.2 boundary: keep only completed sessions
    agg["session_e2e_ms"] = agg["end_ms"] - agg["start_ms"]
    return agg[[*keys, "session_e2e_ms", "session_turns", "session_success"]]


# --------------------------------------------------------------------- SLOs / λ*


def _percentile(vals: pd.Series, percentile: str) -> float:
    return float(vals.quantile(int(percentile[1:]) / 100))


def _evaluate_slo(mreq: pd.DataFrame, slo: dict) -> tuple[float | None, bool]:
    scen = slo["scenario"]
    sel = mreq if scen == "all" else mreq[mreq["scenario"] == scen]
    metric = slo["metric"]
    thr = slo["threshold"]
    if metric == "error_rate_pct":
        total = len(sel)
        measured = 100.0 * int((sel["success"] == 0).sum()) / total if total else 0.0
        return measured, (total > 0 and measured <= thr)
    if metric == "session_e2e_ms":
        vals = session_metrics(sel)["session_e2e_ms"].dropna()
    else:
        vals = sel[sel["success"] == 1][metric].dropna()
    if vals.empty:
        return None, False  # no data to confirm the objective holds → conservatively fail
    measured = _percentile(vals, slo["percentile"])
    return measured, measured <= thr


def evaluate_slos(report: ReportData) -> pd.DataFrame:
    """One row per (λ, SLO): measured value + pass/fail over the measurement phase."""
    mreq = measurement_requests(report)
    rows = []
    for lam in report.rate_levels:
        lam_req = mreq[mreq["rate_lambda"] == lam]
        for slo in report.slos:
            measured, passed = _evaluate_slo(lam_req, slo)
            rows.append({
                "rate_lambda": lam, "scenario": slo["scenario"], "metric": slo["metric"],
                "percentile": slo.get("percentile"), "threshold": slo["threshold"],
                "measured": measured, "passed": passed,
            })
    return pd.DataFrame(rows)


def lambda_star(report: ReportData) -> float | None:
    """Highest swept λ at which all declared SLOs hold simultaneously (§12.4)."""
    if not report.slos:
        return None
    att = evaluate_slos(report)
    if att.empty:
        return None
    all_pass = att.groupby("rate_lambda")["passed"].all()
    ok = [lam for lam in report.rate_levels if bool(all_pass.get(lam, False))]
    return max(ok) if ok else None


# ----------------------------------------------------------- supportable users


def supportable_users(
    report: ReportData, lam: float | None, sessions_per_user_per_hour: dict[str, float]
) -> pd.DataFrame:
    """Per class at λ: session throughput, supportable user population, and Little's-law
    concurrent sessions (§14.1). Empty when λ* is undefined."""
    if lam is None:
        return pd.DataFrame(
            columns=["scenario", "sessions_started", "session_throughput_per_s",
                     "mean_session_wall_s", "sessions_per_user_per_hour",
                     "supportable_users", "concurrent_sessions"]
        )
    mreq = measurement_requests(report)
    mreq = mreq[mreq["rate_lambda"] == lam]
    starts = mreq[mreq["turn_idx"] == 0]  # session starts = first turns in the window
    sm = session_metrics(mreq)
    rows = []
    classes = sorted(starts["scenario"].unique()) if not starts.empty else []
    for scen in classes:
        n_started = int((starts["scenario"] == scen).sum())
        throughput = n_started / report.measurement_s
        scen_sessions = sm[sm["scenario"] == scen]["session_e2e_ms"].dropna()
        mean_wall_s = float(scen_sessions.mean()) / 1000.0 if not scen_sessions.empty else 0.0
        per_hour = sessions_per_user_per_hour.get(scen)
        per_user_per_s = (per_hour / 3600.0) if per_hour else None
        rows.append({
            "scenario": scen,
            "sessions_started": n_started,
            "session_throughput_per_s": throughput,
            "mean_session_wall_s": mean_wall_s,
            "sessions_per_user_per_hour": per_hour,
            "supportable_users": (throughput / per_user_per_s) if per_user_per_s else None,
            "concurrent_sessions": throughput * mean_wall_s,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- quality


def quality_summary(report: ReportData) -> dict:
    """Stage-A gate outcome (+ quality-flagged) and Stage-B scores per concurrency (§12.5)."""
    qe = report.quality_evals
    gate = qe[qe["stage"] == "gate"] if not qe.empty else qe
    compare = qe[qe["stage"] == "compare"] if not qe.empty else qe
    flagged = (not gate.empty) and bool((gate["status"] == "fail").any())
    return {
        "gate": gate,
        "compare": compare,
        "quality_flagged": flagged,
    }


def capacity_vs_quality(
    db_path: Path | str,
    run_ids: list[str],
    sessions_per_user_per_hour: dict[str, float],
    suite: str | None = None,
) -> pd.DataFrame:
    """Per deployment config (run_id): users-at-λ* paired with a Stage-B quality score,
    and the inter-config deltas (§12.5/§14.1 'N× more users at −M pts')."""
    rows = []
    for rid in run_ids:
        report = load_run(db_path, rid)
        lam = lambda_star(report)
        users = supportable_users(report, lam, sessions_per_user_per_hour)
        total_users = float(users["supportable_users"].dropna().sum()) if not users.empty else 0.0
        compare = report.quality_evals
        compare = compare[compare["stage"] == "compare"] if not compare.empty else compare
        if suite is not None and not compare.empty:
            compare = compare[compare["suite"] == suite]
        score = float(compare["score"].mean()) if not compare.empty else None
        rows.append({
            "run_id": rid,
            "config": json.dumps(report.backend_config, sort_keys=True),
            "lambda_star": lam, "supportable_users": total_users, "quality_score": score,
        })
    df = pd.DataFrame(rows)
    if len(df) > 1:
        base_users = df["supportable_users"].iloc[0]
        base_q = df["quality_score"].iloc[0]
        df["users_x_vs_first"] = df["supportable_users"] / base_users if base_users else None
        df["quality_delta_vs_first"] = (
            df["quality_score"] - base_q if base_q is not None else None
        )
    return df


# ------------------------------------------------------------------- hardware


def hardware_vs_lambda(hw: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Mean of a §12.3 telemetry signal per λ (untapped-headroom overlays, §14.1)."""
    if hw.empty or signal not in hw.columns:
        return pd.DataFrame(columns=["rate_lambda", signal])
    sub = hw.dropna(subset=[signal])
    if sub.empty:
        return pd.DataFrame(columns=["rate_lambda", signal])
    return (
        sub.groupby("rate_lambda")[signal].mean().reset_index().sort_values("rate_lambda")
    )
