# Curated-report styling decisions (SPECIFICATIONS.md §15.3)

The AI coding assistant **reads this file first** when drafting any curated report in
`reports/<topic>/`, and honours its directives by default. The operator adds a rule here
once, and every subsequent report inherits it without re-asking. This file grows over time.

These rules govern **presentation only**. They never override the data-integrity contract
(§15.3): no styling rule may hide a SLO breach, a failure-rate spike, a degraded-foundation
pre-check, or any signal a stakeholder needs to see.

## Global decisions

- **SLO line**: always dashed red `#cc0000` (matches `tools/reports/plots.py`); the label includes the
  **unit** (e.g. `SLO 800 ms`).
- **λ axis**: log scale on vs-λ plots; label "λ (session starts/s)" (§12.3); show **plain-number ticks
  ONLY at the swept levels that produced data** (e.g. `0.5 / 1 / 2`) — no scientific `6×10⁻¹` clutter.
- **Per-model colour** (fixed across all reports): Apertus-70B → blue, Kimi-K2.6 → green,
  DeepSeek-V4-Pro → orange. Extend here as models are added.
- **Per-platform colour** (multi-platform comparisons of one model — governs over per-model there):
  SLURM → blue `#0072B2`, K8s → teal `#009E73` (Okabe-Ito, colour-blind-safe; both cool, chosen to read
  side by side and stay clear of the red SLO line). Extend here as platforms are added.
- **`not_modelled` disclosure**: always rendered in a warning-coloured / struck-through panel,
  never omitted (§15.1).
- **Uncollected data**: mark with ❓ "not captured", never a green ✅ — a check implies a *verified*
  value, so green for data that was never collected is misleading (data-integrity, §15.3).
- **Provenance block** at the end of every curated report (**mandatory**, §15.3): source `run_id`s, model +
  BackendConfig of each, curation date, **and an explicit relative link to each source `experiments/<run>/`
  folder + its `benchmark_config.yaml`** — a bare `run_id` is not enough; the reader must be able to click
  through to the config/sbatch/`engine.toml`/run DB. State where the per-run DB lives (in-repo path, or the
  cluster source it is re-pullable from — never cite data that exists only in a transient location). A report
  that doesn't link back to its source experiments is **incomplete**.
- **λ is session starts/s, not request load — always show the realized load too.** λ on a vs-λ axis counts
  *session starts*; the actual load is far higher (per-session turn fan-out, often ~10–100×). Every report
  must surface **both** of these alongside λ, never λ alone:
  - **requests/s** (effective request rate the engine serves — service-capped past the knee), and
  - **concurrent sessions** (live conversations — balloons past the knee as sessions accumulate).
  Plus the in-service `running` batch + queue on the capacity figure. `analysis.applied_load_vs_lambda` +
  `plots.applied_load_figure` produce these; the concurrent-sessions value at λ* must equal the §15.1
  supportable-users concurrent figure (same Little's-law definition).

## Per-audience sections

### Procurement / capacity planning
- Lead with the cost/throughput/users claim, not with TTFT.
- TTFT-vs-λ: show p50 and p95 only (drop p99 for clarity); call the drop out in the narrative.
- Hide the NVLink and PCIe panels unless interconnect sizing is the report's subject.
- Always pair a capacity gain with its measured quality delta (§13.5 capacity-vs-quality).

### Engineering / debugging
- Show all percentiles incl. p99 (and p99.9 when available).
- Keep the hardware-headroom overlays (SM-active, HBM bandwidth) — they explain saturation.

## Plots & axes

- **Capacity figure** = stacked panels sharing the λ axis: **latency** (p50/p95/p99) · **error-rate** ·
  **request-queue depth** — so the latency knee lines up vertically with the queue rising above 0 (§12.2
  saturation onset). When **several latency metrics** are shown for the same runs (e.g. TTFT *and* TPOT),
  stack them as rows above a **single shared error-rate + queue pair** (`compare_capacity_figure`) — error
  and queue are per-request, so repeating them under each metric is redundant; one latency row per metric,
  each with its own SLO line.
- **Percentiles**: one colour per run, **p50 thickest/opaque → p95 → p99 thinnest/lightest** (so a second
  platform/config overlays in a second colour without collisions).
- **Axes**: latency-y **log**; error-rate-y **linear, fixed 0–100 %**; queue-y **symlog** (logarithmic but
  the linear window near 0 keeps the "queue = 0" point visible).
- **Throughput figure**: input tokens/s and output tokens/s vs λ (two panels).
- **Hardware-telemetry figure**: one row per signal vs λ (at minimum **GPU utilization %** and **GPU power
  W**; add memory/temperature as relevant). In comparisons, one column per variant (`hardware_compare_figure`).
  A variant with no telemetry draws an **annotated empty panel** ("no telemetry collected"), never a dropped
  or zero-valued panel — the plot mirror of the ❓ rule (absence ≠ a measured zero). Disclose in the narrative
  which counters are coarse (`nvidia-smi`) vs unwired (DCGM SM-active / HBM BW / NVLink / PCIe → `NULL`).

## Comparison reports (multi-platform / multi-config)

- **Every section reports each variant.** Pre-checks, loading, capacity, queue, quality — each shows all
  compared targets (e.g. SLURM and K8s); tables carry one value per variant per row (`A / B`).
- **State that the offered load was identical** across targets when it was (same pool/seed/sweep/phases),
  so any delta is the serving stack, not the workload.
- **Side-by-side plots**: one column per variant (e.g. SLURM left, K8s right), y-axis **shared per row**,
  so panels are directly comparable. Overlay only single-line metrics (e.g. throughput).
- **Report = folder** `reports/<date>_<topic>/` with `report.md` + an `images/` subfolder.

## Notes

- These conventions were seeded from the **Apertus-70B single-node baseline (SLURM vs K8s)** report
  (2026-06-16). Seed further per-audience rules from each session's iterative adjustments.
