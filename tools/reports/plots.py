"""Matplotlib figures for the report notebook (§15.1). Headless (Agg backend).

Thin presentation over analysis.py: latency percentiles vs λ with the per-class
SLO line and a failure-rate panel, and the hardware-headroom overlay vs λ.
Styling defaults mirror reports/STYLE.md (SLO line dashed red #cc0000).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display, render to file
import matplotlib.pyplot as plt  # noqa: E402

from . import analysis  # noqa: E402

SLO_COLOR = "#cc0000"


def set_style() -> None:
    matplotlib.rcParams.update(
        {"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
         "axes.titlesize": 10, "legend.fontsize": 8}
    )


def latency_figure(report: analysis.ReportData, metric: str, slo_threshold: float | None = None):
    """p50/p95/p99 of `metric` vs λ (log x), an error-rate panel, and a request-queue-depth panel
    (§15.1). The three panels SHARE the λ axis so the latency knee lines up vertically with the
    request queue (`requests_waiting`) rising above 0 — the saturation onset (§12.2)."""
    mreq = analysis.measurement_requests(report)
    lat = analysis.latency_vs_lambda(mreq, metric)
    fail = analysis.failure_rate_vs_lambda(mreq)
    q = analysis.queue_depth_vs_lambda(report.server_stats)
    fig, (ax, axf, axq) = plt.subplots(
        3, 1, figsize=(7, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1.4]}
    )
    if not lat.empty:
        for p in ("p50", "p95", "p99"):
            ax.plot(lat["rate_lambda"], lat[p], marker="o", label=p)
    ax.set_xscale("log")  # shared by all panels
    if slo_threshold:
        ax.axhline(slo_threshold, ls="--", color=SLO_COLOR, label=f"SLO {slo_threshold:g}")
    ax.set_ylabel(f"{metric} (ms)")
    ax.set_title(f"{metric} vs λ — {report.run_id}")
    ax.legend()
    # error-rate panel (already a requirement)
    if not fail.empty:
        axf.plot(fail["rate_lambda"], fail["error_rate_pct"], marker="s", color="#cc6666")
    axf.set_ylabel("error %")
    # request-queue panel (new): mean = sustained backlog, max = peak. Knee aligns with mean > 0.
    if not q.empty:
        axq.plot(q["rate_lambda"], q["waiting_mean"], marker="o", color="#3366aa", label="mean waiting")
        axq.plot(q["rate_lambda"], q["waiting_max"], marker=".", ls=":", color="#999999", label="max waiting")
        axq.axhline(0, color="#cccccc", lw=0.8)
        axq.legend()
    axq.set_ylabel("queue (reqs)")
    axq.set_xlabel("λ (session starts/s)")
    fig.tight_layout()
    return fig


def hardware_figure(report: analysis.ReportData, signals: list[str]):
    """Telemetry signals vs λ — untapped-headroom overlay (§13.3/§15.1). None if no data."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plotted = False
    for signal in signals:
        hv = analysis.hardware_vs_lambda(report.hardware_stats, signal)
        if not hv.empty:
            ax.plot(hv["rate_lambda"], hv[signal], marker="o", label=signal)
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("λ (session starts/s)")
    ax.set_ylabel("value")
    ax.set_title(f"Hardware utilisation vs λ — {report.run_id}")
    ax.legend()
    fig.tight_layout()
    return fig
