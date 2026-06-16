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


# Coordinated, colour-blind-safe platform palette (Okabe-Ito) chosen to read well side by side and
# against the red SLO line (#cc0000): SLURM strong blue, K8s teal — both cool, distinct, far from red.
MODEL_COLOR = "#0072B2"   # SLURM / default (Okabe-Ito blue)
K8S_COLOR = "#009E73"     # K8s (Okabe-Ito teal)


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
        ax.axhline(slo_threshold, ls="--", color=SLO_COLOR, label=f"SLO {slo_threshold:g} ms")
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


def compare_latency_figure(entries, metric: str, slo_threshold: float | None = None):
    """Side-by-side comparison: ONE COLUMN per run (`entries` = list of (report, colour, label),
    e.g. SLURM left, K8s right), 3 rows (latency / error / queue) sharing the y-axis per row and the
    λ x-axis, so the platforms sit directly side by side (§15.1). Per column: p50 thickest/opaque →
    p99 thinnest/lightest; latency-y log, error-y linear 0–100%, queue-y symlog, plain λ ticks."""
    import matplotlib.ticker as mticker

    n = len(entries)
    fig, axes = plt.subplots(
        3, n, figsize=(4.3 * n, 7), sharex=True, sharey="row", squeeze=False,
        gridspec_kw={"height_ratios": [3, 1, 1.4]},
    )
    lambdas: set[float] = set()
    for col, (report, color, label) in enumerate(entries):
        axL, axE, axQ = axes[0][col], axes[1][col], axes[2][col]
        mreq = analysis.measurement_requests(report)
        lat = analysis.latency_vs_lambda(mreq, metric)
        fail = analysis.failure_rate_vs_lambda(mreq)
        q = analysis.queue_depth_vs_lambda(report.server_stats)
        for p, lw, alpha in (("p50", 2.6, 1.0), ("p95", 1.6, 0.70), ("p99", 1.0, 0.45)):
            if not lat.empty:
                axL.plot(lat["rate_lambda"], lat[p], marker="o", ms=4, color=color, lw=lw,
                         alpha=alpha, label=p)
        if slo_threshold:
            axL.axhline(slo_threshold, ls="--", color=SLO_COLOR, label=f"SLO {slo_threshold:g} ms")
        axL.set_yscale("log")
        axL.set_title(label)
        axL.legend(fontsize=7)
        if not fail.empty:
            axE.plot(fail["rate_lambda"], fail["error_rate_pct"], marker="s", ms=4, color=color)
        axE.set_ylim(0, 100)
        if not q.empty:
            axQ.plot(q["rate_lambda"], q["waiting_mean"], marker="o", ms=4, color=color, label="mean")
            axQ.plot(q["rate_lambda"], q["waiting_max"], marker=".", ls=":", color=color, alpha=0.5,
                     label="max")
            axQ.legend(fontsize=7)
        axQ.set_yscale("symlog", linthresh=1)
        axQ.set_xscale("log")
        axQ.set_xlabel("λ (session starts/s)")
        for df in (lat, fail, q):
            if not df.empty:
                lambdas |= {float(x) for x in df["rate_lambda"]}
    axes[0][0].set_ylabel(f"{metric} (ms)")
    axes[1][0].set_ylabel("error %")
    axes[2][0].set_ylabel("queue (reqs)")
    if lambdas:
        ticks = sorted(lambdas)
        for col in range(n):
            axes[2][col].xaxis.set_major_locator(mticker.FixedLocator(ticks))
            axes[2][col].xaxis.set_minor_locator(mticker.NullLocator())
            axes[2][col].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:g}"))
    fig.suptitle(f"{metric} vs λ — SLURM vs K8s")
    fig.tight_layout()
    return fig


def compare_capacity_figure(entries, metrics):
    """Merged side-by-side capacity figure: ONE COLUMN per run (`entries` = (report, colour, label),
    e.g. SLURM left, K8s right). Rows = each latency metric in `metrics` (= list of (metric_column,
    slo_threshold)) stacked above a SINGLE shared **error-rate** row and **request-queue** row.

    error-rate and queue are per-request (metric-independent), so they appear ONCE here instead of being
    repeated under every latency metric (the redundancy of rendering TTFT and TPOT as two separate
    figures). y-axis shared per row, λ x-axis shared; per column p50 thickest/opaque → p99 lightest;
    latency-y log + its SLO line, error-y linear 0–100%, queue-y symlog, plain λ ticks."""
    import matplotlib.ticker as mticker

    n = len(entries)
    nlat = len(metrics)
    nrows = nlat + 2  # latency rows + error + queue
    fig, axes = plt.subplots(
        nrows, n, figsize=(4.3 * n, 2.1 * nrows + 0.8), sharex=True, sharey="row", squeeze=False,
        gridspec_kw={"height_ratios": [3] * nlat + [1, 1.4]},
    )
    lambdas: set[float] = set()
    for col, (report, color, label) in enumerate(entries):
        mreq = analysis.measurement_requests(report)
        fail = analysis.failure_rate_vs_lambda(mreq)
        q = analysis.queue_depth_vs_lambda(report.server_stats)
        for mi, (metric, slo) in enumerate(metrics):
            axL = axes[mi][col]
            lat = analysis.latency_vs_lambda(mreq, metric)
            for p, lw, alpha in (("p50", 2.6, 1.0), ("p95", 1.6, 0.70), ("p99", 1.0, 0.45)):
                if not lat.empty:
                    axL.plot(lat["rate_lambda"], lat[p], marker="o", ms=4, color=color, lw=lw,
                             alpha=alpha, label=p)
                    lambdas |= {float(x) for x in lat["rate_lambda"]}
            if slo:
                axL.axhline(slo, ls="--", color=SLO_COLOR, label=f"SLO {slo:g} ms")
            axL.set_yscale("log")
            if mi == 0:
                axL.set_title(label)
            axL.legend(fontsize=7)
        axE = axes[nlat][col]
        if not fail.empty:
            axE.plot(fail["rate_lambda"], fail["error_rate_pct"], marker="s", ms=4, color=color)
            lambdas |= {float(x) for x in fail["rate_lambda"]}
        axE.set_ylim(0, 100)
        axQ = axes[nlat + 1][col]
        if not q.empty:
            axQ.plot(q["rate_lambda"], q["waiting_mean"], marker="o", ms=4, color=color, label="mean")
            axQ.plot(q["rate_lambda"], q["waiting_max"], marker=".", ls=":", color=color, alpha=0.5,
                     label="max")
            axQ.legend(fontsize=7)
            lambdas |= {float(x) for x in q["rate_lambda"]}
        axQ.set_yscale("symlog", linthresh=1)
        axQ.set_xscale("log")
        axQ.set_xlabel("λ (session starts/s)")
    for mi, (metric, _slo) in enumerate(metrics):
        axes[mi][0].set_ylabel(f"{metric.replace('_ms', '').upper()} (ms)")
    axes[nlat][0].set_ylabel("error %")
    axes[nlat + 1][0].set_ylabel("queue (reqs)")
    if lambdas:
        ticks = sorted(lambdas)
        for col in range(n):
            axes[-1][col].xaxis.set_major_locator(mticker.FixedLocator(ticks))
            axes[-1][col].xaxis.set_minor_locator(mticker.NullLocator())
            axes[-1][col].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:g}"))
    fig.suptitle("Capacity vs λ — SLURM vs K8s")
    fig.tight_layout()
    return fig


def throughput_figure(entries, title: str = "Token throughput vs λ — SLURM vs K8s"):
    """Input- and output-token throughput vs λ for one or more runs (`entries` = (report, colour,
    label)), overlaid — two panels: input tokens/s (top) and output tokens/s (bottom). Both rise with
    λ and plateau at the engine's ceiling past the knee (SLURM and K8s plateau together — same engine)."""
    import matplotlib.ticker as mticker

    fig, (axi, axo) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    lambdas: set[float] = set()
    for report, color, label in entries:
        t = analysis.throughput_vs_lambda(analysis.measurement_requests(report), report.measurement_s)
        if not t.empty:
            axi.plot(t["rate_lambda"], t["input_tok_s"], marker="o", color=color, label=label)
            axo.plot(t["rate_lambda"], t["output_tok_s"], marker="o", color=color, label=label)
            lambdas |= {float(x) for x in t["rate_lambda"]}
    axi.set_ylabel("input tokens/s")
    axi.set_title(title)
    axi.legend(fontsize=8)
    axo.set_ylabel("output tokens/s")
    axo.legend(fontsize=8)
    axo.set_xscale("log")
    axo.set_xlabel("λ (session starts/s)")
    if lambdas:
        ticks = sorted(lambdas)
        axo.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        axo.xaxis.set_minor_locator(mticker.NullLocator())
        axo.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:g}"))
    fig.tight_layout()
    return fig


def hardware_compare_figure(entries, signals):
    """Side-by-side hardware-telemetry comparison (§13.3/§15.1): ONE COLUMN per run
    (`entries` = list of (report, colour, label), e.g. SLURM left, K8s right), one ROW per signal
    (`signals` = list of (column, ylabel, ylim_or_None)), y-axis shared per row, λ x-axis shared.

    A run with no telemetry for a signal (e.g. the cross-cluster K8s gap) draws an empty panel
    annotated "no telemetry collected" rather than being silently dropped — the absence is a
    disclosed instrumentation gap, not a measured zero."""
    import matplotlib.ticker as mticker

    n = len(entries)
    nrows = len(signals)
    fig, axes = plt.subplots(
        nrows, n, figsize=(4.3 * n, 2.4 * nrows + 0.6), sharex=True, sharey="row",
        squeeze=False,
    )
    lambdas: set[float] = set()
    for col, (report, color, label) in enumerate(entries):
        for row, (signal, _ylabel, _ylim) in enumerate(signals):
            ax = axes[row][col]
            hv = analysis.hardware_vs_lambda(report.hardware_stats, signal)
            if hv.empty:
                ax.text(0.5, 0.5, "no telemetry collected", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="0.5", style="italic")
            else:
                ax.plot(hv["rate_lambda"], hv[signal], marker="o", ms=4, color=color)
                lambdas |= {float(x) for x in hv["rate_lambda"]}
            if row == 0:
                ax.set_title(label)
            if _ylim:
                ax.set_ylim(*_ylim)
    for row, (_signal, ylabel, _ylim) in enumerate(signals):
        axes[row][0].set_ylabel(ylabel)
    if lambdas:
        ticks = sorted(lambdas)
        for col in range(n):
            axes[-1][col].set_xscale("log")
            axes[-1][col].set_xlabel("λ (session starts/s)")
            axes[-1][col].xaxis.set_major_locator(mticker.FixedLocator(ticks))
            axes[-1][col].xaxis.set_minor_locator(mticker.NullLocator())
            axes[-1][col].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:g}"))
    fig.suptitle("Hardware telemetry vs λ — SLURM vs K8s")
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
