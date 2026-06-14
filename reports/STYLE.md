# Curated-report styling decisions (SPECIFICATIONS.md §15.3)

The AI coding assistant **reads this file first** when drafting any curated report in
`reports/<topic>/`, and honours its directives by default. The operator adds a rule here
once, and every subsequent report inherits it without re-asking. This file grows over time.

These rules govern **presentation only**. They never override the data-integrity contract
(§15.3): no styling rule may hide a SLO breach, a failure-rate spike, a degraded-foundation
pre-check, or any signal a stakeholder needs to see.

## Global decisions

- **SLO line**: always dashed red `#cc0000` (matches `tools/reports/plots.py`).
- **λ axis**: log scale on latency-vs-λ plots; label "λ (session starts/s)" (§12.3).
- **Per-model colour** (fixed across all reports): Apertus-70B → blue, Kimi-K2.6 → green,
  DeepSeek-V4-Pro → orange. Extend here as models are added.
- **`not_modelled` disclosure**: always rendered in a warning-coloured / struck-through panel,
  never omitted (§15.1).
- **Provenance block** at the end of every curated report: source `run_id`s, model +
  BackendConfig of each, curation date.

## Per-audience sections

### Procurement / capacity planning
- Lead with the cost/throughput/users claim, not with TTFT.
- TTFT-vs-λ: show p50 and p95 only (drop p99 for clarity); call the drop out in the narrative.
- Hide the NVLink and PCIe panels unless interconnect sizing is the report's subject.
- Always pair a capacity gain with its measured quality delta (§13.5 capacity-vs-quality).

### Engineering / debugging
- Show all percentiles incl. p99 (and p99.9 when available).
- Keep the hardware-headroom overlays (SM-active, HBM bandwidth) — they explain saturation.

## Notes

- First curated report is authored after E3 (§15.3); seed new per-audience rules from the
  iterative adjustments made during that session.
