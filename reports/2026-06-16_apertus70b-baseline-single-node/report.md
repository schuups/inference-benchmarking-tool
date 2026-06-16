# Apertus-70B — single-node baseline (256K window): SLURM vs K8s

**Curated synthesis (§15.3)** of the first full-settings capacity+quality run of the KV-cache grid's
**baseline cell** (KV-offloading *off*, KV-dtype *default*). This is the reference against which the
offloading and fp8 cells will be compared. Single GH200 node, TP4, vLLM 0.23.0.

> **Platform comparison — in progress.** This report compares the same baseline across both
> deployment platforms: **SLURM (clariden)** and **Kubernetes (breithorn)**. The SLURM results below
> are complete; the **K8s run is in flight** and the report will be enriched with its results and a
> side-by-side SLURM-vs-K8s comparison once it lands.

## Headline

- **Supportable load: λ\* = 0.5 session-starts/s** — the only swept level meeting **all** per-class
  SLOs. The capacity knee sits between λ=0.5 and λ=1.0.
- **Saturation onset = request-queue onset**, confirmed: the latency knee aligns vertically with the
  vLLM request queue (`num_requests_waiting`) rising above 0. This is the signal the sweep's adaptive
  early-stop keys on (§12.2), and it fired correctly here (skipped λ=4/8/16).
- **Quality intact**: gsm8k 0.72 (gate 0.732 / Stage-B 0.7187) — the baseline quality reference.

![ttft vs λ — latency / error / queue](images/baseline-ttft.png)

*(TPOT companion: `images/baseline-tpot.png`.)*

## Capacity — per-class SLO attainment

| λ (sessions/s) | chat TTFT p95 (SLO 800 ms) | chat TPOT p95 (SLO 80 ms) | agentic session-e2e p90 (SLO 600 s) | error % (SLO 1%) | verdict |
|---|---|---|---|---|---|
| **0.5** | **461 ms** ✅ | **32 ms** ✅ | **208 s** ✅ | **0.0%** ✅ | ✅ **meets all SLOs (λ\*)** |
| 1.0 | 198 000 ms ❌ | 131 ms ❌ | 639 s ❌ | 7.7% ❌ | ❌ saturated |
| 2.0 | 238 000 ms ❌ | 276 ms ❌ | 826 s ❌ | 72% ❌ | ❌ deep overload |
| 4 / 8 / 16 | — | — | — | — | **skipped** (adaptive early-stop, §12.2) |

The transition is abrupt: at λ=1.0 the engine is already catastrophically saturated (TTFT p95 jumps
from 0.46 s to ~198 s). λ* is therefore **bracketed but coarse** — only one sub-knee point (0.5) was
measured. A refinement pass (λ = 0.6 / 0.7 / 0.8) is needed to report a precise λ\*.

## Saturation onset = queue onset

| λ | mean queue (`requests_waiting`) | max queue | all SLOs met? |
|---|---|---|---|
| 0.5 | **0.0** | 0 | ✅ yes |
| 1.0 | 164 | 468 | ❌ no |
| 2.0 | 379 | 1150 | ❌ no |

The queue is empty at λ=0.5 (engine keeps up) and jumps the instant the SLOs break — TTFT climbs
because requests wait. The early-stop used this directly: λ=1.0 saturated (queue non-empty 84% of
measurement scrapes), λ=2.0 confirmed (88%) → the remaining higher λ were skipped as redundant
deep-overload.

## Model-loading breakdown (§10.2) — time-to-ready 314 s

| component | time | note |
|---|---|---|
| weights load | 108 s | 32.9 GiB → **0.30 GB/s effective** |
| engine init (profile + KV alloc + warmup) | 37 s | |
| CUDA-graph capture | 14 s | full graphs, custom all-reduce on (no E2b workarounds) |
| inductor compile | 12 s | |

The effective weight-load bandwidth (0.30 GB/s) is **well below** the §8.1 parallel-read floor
measured on the same mount (1.01 GB/s), so the load is **deserialization / single-stream-limited, not
mount-bandwidth-bound** — a target for the cold-start optimisation track, not a storage problem.

## Quality (§13.5) — gsm8k

| stage | sample | score | floor | status |
|---|---|---|---|---|
| Stage-A gate | 500 | **0.732** | 0.20 | pass |
| Stage-B compare | 1319 (full) | **0.7187** | — | reference |

This is the baseline quality the fp8 / KV-offload cells will be measured against (capacity-vs-quality, §15.1).

## Disclosures & limitations (not hidden, per §15.3)

- **Forced 256K on a natively-64K checkpoint.** Apertus-70B-Instruct-2509's `config.json` caps at
  64K (`max_position_embeddings=65536`); this run set `max_model_len=262144` +
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` to **validate the 256K serving pipeline ahead of the native-256K
  next Apertus**. Outputs on requests that exceed 64K (the long agentic-session tail) are degraded/garbage
  beyond the trained window. **Capacity timing and the queue/knee are unaffected** (forced output length,
  `ignore_eos`); **quality is unaffected** (gsm8k prompts are short, well inside 64K).
- **Latency at λ ≥ 1 is partly a client artifact.** At saturation the single-process load generator hit
  ~649 ms event-loop lag, so absolute latencies at the saturated levels are inflated. λ\* (=0.5) is below
  this regime; the knee/queue determination is engine-side and unaffected. Tracked: shard the load generator.
- **`agentic-coding` is approximate** (§11.7): multi-turn sessions with bursty fan-out, not a tool-calling
  state machine. λ\* → supportable-users mapping needs the per-class sessions/user/hour defaults (§15.1, TBD).
- **KV-cache-% panel empty for this run** — it predates the 0.23 metric-rename scraper fix
  (`gpu_cache_usage_perc` → `kv_cache_usage_perc`); subsequent runs capture it.
- The λ=4/8/16 rows are **skipped (not failed)** by the early-stop; treat their absence as "not measured."

## Provenance

- **run_id**: `20260616-095339_apertus-70b-instruct-2509_vllm_clariden_5124` (SLURM job 2543765, clariden)
- **model**: `swiss-ai/Apertus-70B-Instruct-2509` · image `vllm:0.23.0-alps.net.v1-gh200`
- **BackendConfig**: TP4, PP1, `max_model_len=262144`, `max_num_batched_tokens=16384`,
  `gpu_memory_utilization=0.90`, `enable_prefix_caching=true`, `safetensors_load_strategy=prefetch`;
  **no** KV-offloading, **no** `kv_cache_dtype` (baseline); `env: VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
- **workload**: 80% `chat-short-turns` + 20% `agentic-coding`; 100k-prompt real-text pool (WildChat + LongBench)
- **sweep**: λ=[0.5,1,2,4,8,16], warmup 300 s / measurement 600 s / drain 600 s, queue early-stop (stop after 2 saturated)
- **totals**: 15 245 requests, 3 λ levels run, 1134 sessions truncated at drain, quality_flagged=False
- **curated**: 2026-06-16. Source per-run DB + raw logs under `experiments/`.
