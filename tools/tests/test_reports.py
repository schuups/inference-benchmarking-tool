"""M9 DoD: report analytics asserted against a fixture with known λ*, supportable
users, and capacity-vs-quality deltas; plus a headless notebook-execution smoke test.
"""

import pytest

from tools.reports import analysis
from tools.reports.fixtures import build_fixture_db


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("reports") / "results.db"
    expected = build_fixture_db(path)
    return path, expected


# --------------------------------------------------------------- measurement phase


def test_measurement_window_excludes_warmup(fixture_db):
    path, exp = fixture_db
    report = analysis.load_run(path, exp["run_a"])
    assert report.warmup_s == 10 and report.measurement_s == 10
    mreq = analysis.measurement_requests(report)
    assert len(mreq) == 60  # 20 per λ × 3; the 6 warmup rows are excluded
    assert (mreq["ttft_ms"] != 9999.0).all()  # the bad warmup ttft never leaks in


def test_latency_percentiles_per_lambda(fixture_db):
    path, exp = fixture_db
    report = analysis.load_run(path, exp["run_a"])
    lat = analysis.latency_vs_lambda(analysis.measurement_requests(report), "ttft_ms")
    p95 = {row["rate_lambda"]: row["p95"] for _, row in lat.iterrows()}
    assert p95 == {1.0: 300.0, 2.0: 450.0, 3.0: 600.0}


# ------------------------------------------------------------------- λ* / SLOs


def test_lambda_star(fixture_db):
    path, exp = fixture_db
    assert analysis.lambda_star(analysis.load_run(path, exp["run_a"])) == exp["lambda_star_a"]
    assert analysis.lambda_star(analysis.load_run(path, exp["run_b"])) == exp["lambda_star_b"]


def test_slo_attainment_table(fixture_db):
    path, exp = fixture_db
    report = analysis.load_run(path, exp["run_a"])
    att = analysis.evaluate_slos(report)
    by_lambda = att.groupby("rate_lambda")["passed"].all()
    assert bool(by_lambda[1.0]) and bool(by_lambda[2.0]) and not bool(by_lambda[3.0])


# ----------------------------------------------------------- supportable users


def test_supportable_users_at_lambda_star(fixture_db):
    path, exp = fixture_db
    report = analysis.load_run(path, exp["run_a"])
    lam = analysis.lambda_star(report)
    users = analysis.supportable_users(report, lam, exp["sessions_per_user_per_hour"])
    row = users[users["scenario"] == exp["scenario"]].iloc[0]
    assert row["session_throughput_per_s"] == pytest.approx(2.0)
    assert row["supportable_users"] == pytest.approx(exp["users_a"])
    assert row["concurrent_sessions"] == pytest.approx(exp["concurrent_a"])  # Little's law


def test_supportable_users_undefined_when_no_lambda_star(fixture_db):
    path, exp = fixture_db
    report = analysis.load_run(path, exp["run_a"])
    assert analysis.supportable_users(report, None, exp["sessions_per_user_per_hour"]).empty


# ------------------------------------------------------------------- quality


def test_quality_summary_not_flagged(fixture_db):
    path, exp = fixture_db
    summary = analysis.quality_summary(analysis.load_run(path, exp["run_a"]))
    assert summary["quality_flagged"] is False
    assert not summary["compare"].empty


def test_capacity_vs_quality_delta(fixture_db):
    path, exp = fixture_db
    cvq = analysis.capacity_vs_quality(
        path, [exp["run_a"], exp["run_b"]], exp["sessions_per_user_per_hour"], suite="gsm8k"
    )
    a = cvq[cvq["run_id"] == exp["run_a"]].iloc[0]
    b = cvq[cvq["run_id"] == exp["run_b"]].iloc[0]
    assert a["supportable_users"] == pytest.approx(2.0)
    assert b["supportable_users"] == pytest.approx(3.0)
    assert a["quality_score"] == pytest.approx(0.80)
    assert b["quality_score"] == pytest.approx(0.76)
    assert b["users_x_vs_first"] == pytest.approx(1.5)  # 1.5× users…
    assert b["quality_delta_vs_first"] == pytest.approx(-0.04)  # …at −0.04 pts


# ------------------------------------------------------------------- hardware


def test_hardware_vs_lambda(fixture_db):
    path, exp = fixture_db
    report = analysis.load_run(path, exp["run_a"])
    hw = analysis.hardware_vs_lambda(report.hardware_stats, "gpu_sm_active_pct")
    sm = {row["rate_lambda"]: row["gpu_sm_active_pct"] for _, row in hw.iterrows()}
    assert sm == {1.0: 35.0, 2.0: 60.0, 3.0: 85.0}  # headroom overlay data present


def test_hardware_compare_figure_empty_panel(fixture_db):
    # §15.1 side-by-side telemetry: a run with no telemetry (the cross-cluster K8s gap) must draw
    # an annotated empty panel, not be dropped — absence is a disclosed gap, not a measured zero.
    import dataclasses

    import pandas as pd

    from tools.reports import plots

    path, exp = fixture_db
    report = analysis.load_run(path, exp["run_a"])
    empty = dataclasses.replace(report, hardware_stats=pd.DataFrame())
    fig = plots.hardware_compare_figure(
        [(report, plots.MODEL_COLOR, "SLURM"), (empty, plots.K8S_COLOR, "K8s")],
        [("gpu_sm_active_pct", "SM active (%)", (0, 105))],
    )
    assert len(fig.axes) == 2  # 1 signal row × 2 platform columns
    texts = [t.get_text() for t in fig.axes[1].texts]  # K8s column (empty)
    assert "no telemetry collected" in texts
    assert not fig.axes[0].texts or "no telemetry collected" not in [
        t.get_text() for t in fig.axes[0].texts
    ]  # SLURM column has data, no annotation


def test_compare_capacity_figure_merges_error_and_queue(fixture_db):
    # §15.1 merged capacity figure: N latency metrics stacked above ONE shared error + queue pair
    # (per-request, so not repeated per metric). For 2 metrics × 2 platforms → (2+2) rows × 2 cols.
    from tools.reports import plots

    path, exp = fixture_db
    report = analysis.load_run(path, exp["run_a"])
    fig = plots.compare_capacity_figure(
        [(report, plots.MODEL_COLOR, "SLURM"), (report, plots.K8S_COLOR, "K8s")],
        [("ttft_ms", 800.0), ("tpot_ms", 80.0)],
    )
    assert len(fig.axes) == 8  # (2 latency + error + queue) rows × 2 columns
    # each latency row carries an SLO line (a dashed red axhline)
    slo_lines = [ln for ln in fig.axes[0].get_lines() if ln.get_color() == plots.SLO_COLOR]
    assert slo_lines, "TTFT panel should carry its SLO line"


def test_queue_depth_vs_lambda():
    # §12.2 / §15.1 queue panel: per-λ mean (sustained) + max (peak) requests_waiting.
    import pandas as pd

    ss = pd.DataFrame([
        {"rate_lambda": 1.0, "requests_waiting": 0, "requests_running": 4, "gpu_cache_pct": 20.0},
        {"rate_lambda": 1.0, "requests_waiting": 0, "requests_running": 5, "gpu_cache_pct": 25.0},
        {"rate_lambda": 4.0, "requests_waiting": 12, "requests_running": 100, "gpu_cache_pct": 99.0},
        {"rate_lambda": 4.0, "requests_waiting": 8, "requests_running": 110, "gpu_cache_pct": 98.0},
    ])
    q = analysis.queue_depth_vs_lambda(ss)
    assert list(q["rate_lambda"]) == [1.0, 4.0]
    assert q[q["rate_lambda"] == 1.0].iloc[0]["waiting_mean"] == 0.0           # below saturation
    row4 = q[q["rate_lambda"] == 4.0].iloc[0]
    assert row4["waiting_mean"] == 10.0 and row4["waiting_max"] == 12.0        # queue > 0
    assert analysis.queue_depth_vs_lambda(pd.DataFrame()).empty                # no scrapes


# ----------------------------------------------------------- notebook execution


@pytest.mark.slow
def test_notebook_executes_headless(tmp_path, fixture_db):
    import nbformat

    from tools.reports.notebook import build_notebook
    from tools.reports.render import render_report

    path, exp = fixture_db
    template = tmp_path / "template.ipynb"
    nbformat.write(build_notebook(), str(template))

    out = tmp_path / "report_out"
    executed = render_report(template, path, exp["run_a"], out, exp["sessions_per_user_per_hour"])

    assert executed.exists()
    assert (out / "ttft.png").exists() and (out / "itl.png").exists()
    assert (out / "hardware.png").exists()  # fixture has telemetry
    # the executed notebook recorded the known λ* in a cell output
    nb = nbformat.read(str(executed), as_version=4)
    text = "\n".join(
        o.get("text", "") for c in nb.cells for o in c.get("outputs", [])
    )
    assert "λ* = 2.0" in text


def test_inject_params_emits_python_literals():
    """run_id=None must inject `run_id = None`, not JSON `null` (kernel NameError)."""
    from tools.reports.notebook import build_notebook
    from tools.reports.render import _inject_params

    nb = build_notebook()
    _inject_params(nb, {"db_path": "/x/run.db", "run_id": None, "out_dir": ".",
                        "sessions_per_user_per_hour": {"chat-short-turns": 3600.0}})
    src = next(c.source for c in nb.cells if "parameters" in c.get("metadata", {}).get("tags", []))
    assert "run_id = None" in src and "null" not in src
    compile(src, "<params>", "exec")  # valid Python


def test_template_notebook_committed_and_current():
    """The committed experiments/template_report.ipynb matches the builder (no drift)."""
    import nbformat

    from tools.common.config import REPO_ROOT
    from tools.reports.notebook import build_notebook

    committed = REPO_ROOT / "experiments" / "template_report.ipynb"
    assert committed.exists(), "run `python -m tools.reports.notebook` to generate it"
    sources_committed = [c.source for c in nbformat.read(str(committed), as_version=4).cells]
    sources_built = [c.source for c in build_notebook().cells]
    assert sources_committed == sources_built
