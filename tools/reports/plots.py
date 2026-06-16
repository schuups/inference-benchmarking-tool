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


MODEL_COLOR = "#1f77b4"  # Apertus-70B → blue (STYLE.md); a 2nd platform overlays in a 2nd colour.


def latency_figure(
    report: analysis.ReportData, metric: str, slo_threshold: float | None = None,
    color: str = MODEL_COLOR, label: str | None = None,
):
    """p50/p95/p99 of `metric` vs λ + an error-rate panel + a request-queue panel, sharing the λ
    axis (§15.1) so the latency knee lines up vertically with the queue rising above 0 (§12.2).

    Each percentile-set is drawn in ONE colour — p50 thickest/opaque, p95/p99 progressively thinner
    and lighter — so a second platform (SLURM vs K8s) overlays cleanly in a second colour. Latency-y
    is log; error-y is linear 0–100%; queue-y is symlog (log, but still shows the 0 at low λ); the λ
    axis shows plain-number ticks only at the swept levels that have data."""
    import matplotlib.ticker as mticker

    mreq = analysis.measurement_requests(report)
    lat = analysis.latency_vs_lambda(mreq, metric)
    fail = analysis.failure_rate_vs_lambda(mreq)
    q = analysis.queue_depth_vs_lambda(report.server_stats)
    fig, (ax, axf, axq) = plt.subplots(
        3, 1, figsize=(7, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1.4]}
    )
    pfx = f"{label} " if label else ""
    # latency — same colour, p50 thickest/opaque → p99 thinnest/lightest
    for p, lw, alpha in (("p50", 2.6, 1.0), ("p95", 1.6, 0.70), ("p99", 1.0, 0.45)):
        if not lat.empty:
            ax.plot(lat["rate_lambda"], lat[p], marker="o", ms=4, color=color, lw=lw,
                    alpha=alpha, label=f"{pfx}{p}")
    ax.set_yscale("log")
    if slo_threshold:
        ax.axhline(slo_threshold, ls="--", color=SLO_COLOR, label=f"SLO {slo_threshold:g}")
    ax.set_ylabel(f"{metric} (ms)")
    ax.set_title(f"{metric} vs λ — {report.run_id}")
    ax.legend()
    # error-rate — linear, fixed 0–100%
    if not fail.empty:
        axf.plot(fail["rate_lambda"], fail["error_rate_pct"], marker="s", ms=4, color=color)
    axf.set_ylim(0, 100)
    axf.set_ylabel("error %")
    # request queue — symlog (logarithmic, but the linear window near 0 keeps the λ=low "queue 0" visible)
    if not q.empty:
        axq.plot(q["rate_lambda"], q["waiting_mean"], marker="o", ms=4, color=color, label=f"{pfx}mean")
        axq.plot(q["rate_lambda"], q["waiting_max"], marker=".", ls=":", color=color, alpha=0.5,
                 label=f"{pfx}max")
        axq.legend()
    axq.set_yscale("symlog", linthresh=1)
    axq.set_ylabel("queue (reqs)")
    # λ axis — log, but plain-number ticks ONLY at the swept levels that produced data
    ax.set_xscale("log")
    lambdas = sorted({float(x) for df in (lat, fail, q) if not df.empty for x in df["rate_lambda"]})
    if lambdas:
        axq.xaxis.set_major_locator(mticker.FixedLocator(lambdas))
        axq.xaxis.set_minor_locator(mticker.NullLocator())
        axq.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:g}"))
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
