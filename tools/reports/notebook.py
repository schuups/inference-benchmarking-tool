"""Builds the §14.1 report notebook programmatically (M9).

`build_notebook()` returns an nbformat NotebookNode whose cells present every
§14.1 panel by calling the tested `analysis`/`plots` modules. The first code cell
is tagged `parameters` (papermill convention) so `render.py` can inject the DB
path / run_id / output dir; with the defaults it self-runs on the synthetic
fixture (so the template executes standalone for the M9 DoD).

Regenerate the committed template:  python -m tools.reports.notebook
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

_PARAMS = """\
# Parameters (papermill-injected by tools/reports/render.py). Defaults self-run on the fixture.
db_path = None
run_id = None
out_dir = "."
sessions_per_user_per_hour = {}
"""

_SETUP = """\
from pathlib import Path
import pandas as pd
from tools.reports import analysis, plots
plots.set_style()
out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
if db_path is None:                       # demo/self-test mode: build the fixture
    from tools.reports.fixtures import build_fixture_db
    db_path = str(out / "fixture_results.db")
    build_fixture_db(db_path)
    run_id = run_id or "fixtureA"
    sessions_per_user_per_hour = sessions_per_user_per_hour or {"chat-short-turns": 3600.0}
report = analysis.load_run(db_path, run_id)
mreq = analysis.measurement_requests(report)
print("loaded run", report.run_id, "—", len(mreq), "measurement-phase requests")
"""

_SCENARIO = """\
# Scenario & assumptions (§13.7) — read every chart below in this context.
display(pd.DataFrame(report.manifest.get("mix", [])))
for c in report.manifest.get("classes", []):
    print(f"# {c['name']} — {c.get('summary','')}")
    print("  modelled:", c.get("modelled"))
    print("  NOT modelled:", c.get("not_modelled"))   # must not be missed (§14.1)
    print("  assumptions:", c.get("assumptions"))
print("run assumptions:", report.manifest.get("run_assumptions"))
"""

_CONFIG = """\
# Configuration summary
display(pd.DataFrame([{"model": report.model, "backend": report.backend, **report.backend_config}]))
"""

_PRECHECKS = """\
# System pre-checks (§13.6) — warns/fails flagged at the top (§14.1)
sp = report.system_prechecks
if sp.empty:
    print("no system pre-checks recorded")
else:
    flagged = sp[sp["status"].isin(["warn", "fail"])]
    if not flagged.empty:
        print("⚠️  DEGRADED FOUNDATION — interpret all numbers below with care:")
        display(flagged[["metric", "measured", "expected", "status"]])
    display(sp[["metric", "measured", "expected", "tolerance_pct", "status"]])
"""

_MODEL_LOAD = """\
# Model loading times (§9.2), per instance
cols = ["instance_id", "node", "model_load_total_s", "model_load_weights_s",
        "model_load_engine_init_s", "model_load_cuda_graph_capture_s", "model_load_inductor_compile_s"]
display(report.instances[[c for c in cols if c in report.instances.columns]])
"""

_TTFT = """\
# TTFT vs λ with the per-class SLO line (§14.1)
ttft_slo = next((s["threshold"] for s in report.slos if s["metric"] == "ttft_ms"), None)
fig = plots.latency_figure(report, "ttft_ms", slo_threshold=ttft_slo)
fig.savefig(out / "ttft.png", bbox_inches="tight"); fig
"""

_ITL = """\
# Inter-token latency (TPOT/ITL) vs λ
tpot_slo = next((s["threshold"] for s in report.slos if s["metric"] == "tpot_ms"), None)
fig = plots.latency_figure(report, "tpot_ms", slo_threshold=tpot_slo)
fig.savefig(out / "itl.png", bbox_inches="tight"); fig
"""

_PER_CLASS = """\
# Per-class breakdown (mixed runs, §10.4/§14.1): TTFT percentiles by scenario
display(analysis.latency_vs_lambda(mreq, "ttft_ms", by_scenario=True))
display(analysis.failure_rate_vs_lambda(mreq, by_scenario=True))
"""

_SLO = """\
# SLO attainment per λ and the derived λ* (§12.4)
att = analysis.evaluate_slos(report)
lam_star = analysis.lambda_star(report)
print("λ* =", lam_star)
display(att)
"""

_USERS = """\
# Supportable-users estimate at λ* (§14.1) — edit sessions_per_user_per_hour above
users = analysis.supportable_users(report, lam_star, sessions_per_user_per_hour)
display(users if not users.empty else "λ* undefined — extend the sweep toward lower rates")
"""

_QUALITY = """\
# Response quality (§12.5): Stage-A gate, Stage-B scores, capacity-vs-quality
q = analysis.quality_summary(report)
if q["quality_flagged"]:
    print("⚠️  QUALITY-FLAGGED: Stage-A gate failed under on_fail=continue (§14.1)")
display(q["compare"][["suite", "eval_concurrency", "score"]] if not q["compare"].empty
        else "no Stage-B quality rows")
runs = analysis.list_runs(db_path)
if len(runs) > 1:
    print("Capacity vs quality across deployment configs:")
    display(analysis.capacity_vs_quality(db_path, runs, sessions_per_user_per_hour))
"""

_HARDWARE = """\
# Hardware utilisation vs λ — untapped headroom (§12.3/§14.1)
fig = plots.hardware_figure(report, ["gpu_sm_active_pct", "gpu_tensor_active_pct"])
if fig is not None:
    fig.savefig(out / "hardware.png", bbox_inches="tight")
fig if fig is not None else "no hardware telemetry recorded"
"""

_RAW = """\
# Raw per-rate-level table
raw = analysis.latency_vs_lambda(mreq, "ttft_ms").merge(
    analysis.failure_rate_vs_lambda(mreq), on="rate_lambda", how="outer", suffixes=("_ttft", "")
)
display(raw)
"""


def build_notebook() -> nbformat.NotebookNode:
    params = new_code_cell(_PARAMS)
    params.metadata["tags"] = ["parameters"]
    cells = [
        new_markdown_cell(
            "# Inference Benchmark Report\n\n"
            "Generated from the per-run / centralized results DB (SPECIFICATIONS.md §14.1). "
            "Latency/SLO math is over the measurement phase only (§11.2)."
        ),
        params,
        new_code_cell(_SETUP),
        new_markdown_cell("## Scenario & assumptions"),
        new_code_cell(_SCENARIO),
        new_markdown_cell("## Configuration"),
        new_code_cell(_CONFIG),
        new_markdown_cell("## System pre-checks"),
        new_code_cell(_PRECHECKS),
        new_markdown_cell("## Model loading times"),
        new_code_cell(_MODEL_LOAD),
        new_markdown_cell("## Latency vs λ"),
        new_code_cell(_TTFT),
        new_code_cell(_ITL),
        new_code_cell(_PER_CLASS),
        new_markdown_cell("## SLO attainment & λ*"),
        new_code_cell(_SLO),
        new_markdown_cell("## Supportable users"),
        new_code_cell(_USERS),
        new_markdown_cell("## Response quality"),
        new_code_cell(_QUALITY),
        new_markdown_cell("## Hardware utilisation"),
        new_code_cell(_HARDWARE),
        new_markdown_cell("## Raw data"),
        new_code_cell(_RAW),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> int:
    from tools.common.config import REPO_ROOT

    out = REPO_ROOT / "experiments" / "template_report.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), str(out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
