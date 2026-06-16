# Apertus-70B (256k context) - single-node baseline (SLURM vs K8s)

**Purpose.** Establish the **baseline operating point** for Apertus-70B served on a single GH200 node,
run at a **256K context window to approximate the upcoming Apertus-70B 1.5** (which will natively
support 256K). This checkpoint is natively 64K, so the 256K window is **forced**
(`VLLM_ALLOW_LONG_MAX_MODEL_LEN`; see Disclosures). This is the **reference against which the
KV-offloading and fp8 cells of the grid will be compared** (capacity and quality deltas, §15.1), and
the first run of a **SLURM-vs-K8s platform comparison** of the identical deployment.

**Setup.**

| | |
|---|---|
| Model | `swiss-ai/Apertus-70B-Instruct-2509` (70B dense), full **256K** context (`max_model_len=262144`) |
| Engine | vLLM **0.23.0**, TP4 × PP1 on **one GH200 node** (4× GH200, intra-node NVLink-C2C) |
| Image | `jfrog.svc.cscs.ch/ml/inference/vllm:0.23.0-alps.net.v1-gh200` — official base image vLLM 0.23 + [Alps Slingshot-11 netstack](../../tools/images/core/nvidia/netstack/v1) + baked Ray/NIXL/LMCache; self-contained (host CXI hook off) |
| Storage | weights from the **capstor** Lustre HF-cache (`/capstor/scratch/.../ibt/hf-cache`); dataset pool on capstor. (K8s variant: weights from a `ceph-corbo-cephfs` PVC.) |
| Platforms | **SLURM** (clariden) + **Kubernetes** (breithorn) — both complete; compared in §7 |

## 1. Pre-checks (§8 foundation gate)

The §8 gate runs **before** the engine is admitted, so the sweep can never measure a degraded
foundation. All checks passed on the populated 1-node clariden reference:

| check | measured | reference | verdict |
|---|---|---|---|
| NCCL all-reduce @128 MiB | 309.5 GB/s | 317.7 | ✅ pass |
| NCCL all-gather @128 MiB | 278.9 GB/s | 283.5 | ✅ pass |
| NCCL all-to-all @128 MiB | 307.8 GB/s | 306.2 | ✅ pass |
| NVSHMEM all-to-all latency @128 KiB *(not relevant for this model)* | 12.68 µs | — | ✅ pass |
| capstor sequential read (single-stream O_DIRECT) | 0.197 GB/s | floor 0.063 | ✅ pass |
| capstor parallel read (8-stream O_DIRECT) | 1.01 GB/s | — | ✅ pass |
| capstor buffered read (readahead) | 14.32 GB/s | — | ✅ pass |

Intra-node NVLink collectives are at reference; storage is healthy (well above the degraded-capstor
floor). Foundation sound → the capacity numbers below reflect the engine, not a broken substrate.

## 2. Model-loading time breakdown (§10.2) — time-to-ready 314 s

| component | time | note |
|---|---|---|
| weights load | 108 s | 32.9 GiB → **0.30 GB/s effective** |
| engine init (profile + KV alloc + warmup) | 37 s | |
| CUDA-graph capture | 14 s | full graphs, custom all-reduce **on** (no E2b workarounds on 0.23) |
| inductor compile | 12 s | |

The effective weight-load bandwidth (0.30 GB/s) is **well below** the §8 parallel-read floor on the same
mount (1.01 GB/s), so the load is **deserialization / single-stream-limited, not mount-bandwidth-bound**
— a target for the cold-start optimisation track, not a storage problem.

## 3. Loading scenario (the benchmark workload)

A realistic **mixed** workload, weighted by session starts (§12.3):

- **80% `chat-short-turns`** — short interactive turns (turn-1 ≈ 500 tokens).
- **20% `agentic-coding`** — long coding sessions (turn-1 ≈ 10 000 tokens).

Both are **multi-turn** with true conversational growth (`append_delta`: turn *N*'s prompt = the full
prior transcript, including the model's own outputs, + the new turn), so a session's context **grows
turn-over-turn** toward the 256K window — and *not every session fills it* (turn counts and lengths are
drawn from per-scenario distributions). Prompts are **real text**: chat from **WildChat-1M**, agentic
from **LongBench**. The pool is **100 000 turn records** (seed 1234, forced output length for sweep
reproducibility), generated once and shared across cells.

Load is offered as a **Poisson** arrival of session starts at rate **λ**, swept
`[0.5, 1, 2, 4, 8, 16]` sessions/s with **warmup 300 s / measurement 600 s / drain 600 s** per level.
The sweep uses the **sustained-queue adaptive early-stop** (§12.2): once the request queue is
persistently non-empty for two consecutive levels, the remaining higher λ are skipped as redundant
deep-overload.

## 4. Capacity — per-class SLO attainment

![ttft vs λ — latency / error / queue (shared λ axis), SLURM (blue) vs K8s (orange)](images/baseline-ttft.png)

*(TPOT companion: `images/baseline-tpot.png`. The figure overlays both platforms — SLURM blue, K8s
orange — which are nearly coincident; per-platform numbers below are SLURM, the side-by-side is §7.)*

- **Supportable load: λ\* = 0.5 session-starts/s** — the only swept level meeting **all** per-class SLOs.

| λ (sessions/s) | chat TTFT p95 (SLO 800 ms) | chat TPOT p95 (SLO 80 ms) | agentic session-e2e p90 (SLO 600 s) | error % (SLO 1%) | verdict |
|---|---|---|---|---|---|
| **0.5** | **461 ms** ✅ | **32 ms** ✅ | **208 s** ✅ | **0.0%** ✅ | ✅ **meets all SLOs (λ\*)** |
| 1.0 | 198 000 ms ❌ | 131 ms ❌ | 639 s ❌ | 7.7% ❌ | ❌ saturated |
| 2.0 | 238 000 ms ❌ | 276 ms ❌ | 826 s ❌ | 72% ❌ | ❌ deep overload |
| 4 / 8 / 16 | — | — | — | — | **skipped** (adaptive early-stop) |

The transition is abrupt: at λ=1.0 the engine is already catastrophically saturated (TTFT p95 jumps from
0.46 s to ~198 s). λ\* is **bracketed but coarse** — only one sub-knee point (0.5) was measured; a
refinement pass (λ = 0.6 / 0.7 / 0.8) is needed for a precise λ\*.

## 5. Saturation onset = queue onset

| λ | mean queue (`requests_waiting`) | max queue | all SLOs met? |
|---|---|---|---|
| 0.5 | **0.0** | 0 | ✅ yes |
| 1.0 | 164 | 468 | ❌ no |
| 2.0 | 379 | 1150 | ❌ no |

The queue is empty at λ=0.5 (engine keeps up) and jumps the instant the SLOs break — TTFT climbs because
requests wait. The early-stop used this directly: λ=1.0 saturated (queue non-empty 84% of measurement
scrapes), λ=2.0 confirmed (88%) → higher λ skipped.

## 6. Quality (§13.5) — gsm8k

| stage | sample | score | floor | status |
|---|---|---|---|---|
| Stage-A gate | 500 | **0.732** | 0.20 | pass |
| Stage-B compare | 1319 (full) | **0.7187** | — | reference |

This is the baseline quality the fp8 / KV-offload cells will be measured against (capacity-vs-quality, §15.1).

## 7. Platform comparison — SLURM vs K8s

The **identical** baseline (same model, image, BackendConfig, 256K window, shared 100k pool) was run on
both platforms. The K8s engine was deployed as a namespaced `ml` manifest and load-generated from a
clariden SLURM benchmarker over the breithorn ingress (then torn down). The §4/§5 figures overlay both
(SLURM blue, K8s orange) — the curves are nearly coincident.

**Capacity & quality — essentially identical:**

| | SLURM (clariden) | K8s (breithorn) |
|---|---|---|
| **λ\*** (meets all SLOs) | **0.5** | **0.5** |
| TTFT p50 / p95 @ λ=0.5 | 96 / 712 ms | 110 / 750 ms |
| queue mean @ λ=1.0 (saturated) | 164 | 170 |
| early-stop | [0.5,1,2] → skip 4/8/16 | [0.5,1,2] → skip 4/8/16 |
| gsm8k gate / Stage-B | 0.732 / 0.7187 | 0.738 / **0.7187** |
| requests / sessions truncated | 15 245 / 1134 | 15 277 / 1141 |

**Read:** both platforms land at the **same operating point** — λ\*=0.5, same knee, **identical Stage-B
quality** (0.7187, deterministic gsm8k). K8s adds only a **small fixed serving overhead at the operating
point** (~14 ms p50 / ~38 ms p95 higher TTFT at λ=0.5) from the ingress + cross-cluster load-gen hop; at
saturation the curves are indistinguishable. **No capacity or quality penalty** from the K8s path for
this single-node baseline.

**Path differences (not findings — K8s instrumentation gaps to close):**
- **Model-loading breakdown — SLURM only.** The K8s engine is deployed externally (kubectl), so the
  benchmarker can't read the pod's vLLM log to parse weights/capture/compile; the K8s
  `model_load_total_s` (110 s) is just the benchmarker's *ingress-readiness wait*, not the model load.
  Follow-up: parse `kubectl logs` for the K8s breakdown.
- **Hardware telemetry — K8s empty.** The in-pod sampler writes to the pod's `/results` PVC, which the
  clariden benchmarker doesn't read, so K8s `hardware_stats` has 0 rows (SLURM has the full §13.3 sampling).

## Disclosures & limitations (not hidden, per §15.3)

- **Forced 256K on a natively-64K checkpoint.** `config.json` caps at 64K (`max_position_embeddings=65536`);
  this run set `max_model_len=262144` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` to **validate the 256K serving
  pipeline ahead of Apertus-70B 1.5** (the upcoming version, which will natively support 256K). Outputs on requests exceeding 64K (the long
  agentic-session tail) are degraded beyond the trained window. **Capacity timing and the queue/knee are
  unaffected** (forced output length, `ignore_eos`); **quality is unaffected** (gsm8k prompts are short).
- **Latency at λ ≥ 1 is partly a client artifact.** At saturation the single-process load generator hit
  ~649 ms event-loop lag, inflating absolute latencies at the saturated levels. λ\* (=0.5) is below this
  regime; the knee/queue determination is engine-side and unaffected. Tracked: shard the load generator.
- **`agentic-coding` is approximate** (§11.7): multi-turn sessions with bursty fan-out, not a tool-calling
  state machine. λ\* → supportable-users mapping needs per-class sessions/user/hour defaults (§15.1, TBD).
- **KV-cache-% panel empty for this run** — it predates the 0.23 metric-rename scraper fix
  (`gpu_cache_usage_perc` → `kv_cache_usage_perc`); subsequent runs capture it.
- λ=4/8/16 rows are **skipped (not failed)** by the early-stop; treat their absence as "not measured".

## Provenance

- **run_id (SLURM)**: `20260616-095339_apertus-70b-instruct-2509_vllm_clariden_5124` (job 2543765, clariden)
- **run_id (K8s)**: `20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423` (benchmarker job 2545821 on clariden; engine on breithorn ns `ml`, deployed via kubectl + torn down post-run)
- **model**: `swiss-ai/Apertus-70B-Instruct-2509` · image `vllm:0.23.0-alps.net.v1-gh200`
- **BackendConfig**: TP4, PP1, `max_model_len=262144`, `max_num_batched_tokens=16384`,
  `gpu_memory_utilization=0.90`, `enable_prefix_caching=true`, `safetensors_load_strategy=prefetch`;
  **no** KV-offloading, **no** `kv_cache_dtype` (baseline); `env: VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
- **workload**: 80% `chat-short-turns` + 20% `agentic-coding`; 100k-prompt real-text pool (WildChat + LongBench)
- **sweep**: λ=[0.5,1,2,4,8,16], warmup 300 s / measurement 600 s / drain 600 s, queue early-stop (stop after 2 saturated)
- **totals**: 15 245 requests, 3 λ levels run, 1134 sessions truncated at drain, quality_flagged=False
- **curated**: 2026-06-16. Source per-run DB + raw logs under `experiments/`.
