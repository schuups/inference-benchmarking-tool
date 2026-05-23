# Inference Benchmarking — Specification

This document enumerates all requirements captured through design and implementation.

## Table of Contents

1. [Guiding Principles](#1-guiding-principles)
2. [Deployment Targets](#2-deployment-targets)
3. [Resource Identification](#3-resource-identification)
4. [Resource Lifecycle (Cleanup)](#4-resource-lifecycle-cleanup)
5. [Server Configuration Options](#5-server-configuration-options)
6. [Prompt Generation](#6-prompt-generation)
7. [Inductor Pre-compilation Primer](#7-inductor-pre-compilation-primer)
8. [Load Generation](#8-load-generation)
9. [Measurement](#9-measurement)
10. [Benchmarker Infrastructure](#10-benchmarker-infrastructure)
11. [Results](#11-results)
12. [Reporting](#12-reporting)
13. [Cluster-Specific Constraints (Alps / GH200)](#13-cluster-specific-constraints-alps--gh200)
14. [Known Issues & Workarounds](#14-known-issues--workarounds)
15. [Open Items](#15-open-items)

---

## 1. Guiding Principles

- **Laptop-orchestrated**: all coordination runs on the operator's laptop; no coordinator node
  is allocated on the cluster.
- **Backend-agnostic**: vLLM, sglang, and NVIDIA Dynamo are first-class backends. Adding a new
  one requires a new EDF template, sbatch template, and K8s deployment template only.
- **Open-loop load generation**: requests are issued at a fixed Poisson rate λ, independent of
  server completions. This faithfully models queuing behaviour under overload.
- **Reproducible by config**: a single YAML file fully specifies a sweep. Re-running the same
  file must produce comparable results.
- **Separation of concerns**: the inference server runs on its own (GPU) allocation. The
  **Benchmarker** is a separate SLURM allocation hosting the **dataset generator** and the
  **load generator** as distinct, sequential phases — prompt preparation always completes
  before measurement begins. No benchmark CPU work competes with GPU inference, and dataset
  generation never overlaps with load generation.
- **Clean cluster state**: all deployed resources must be cleaned up after every run — on both
  success and failure paths. No orphaned jobs, pods, services, secrets, or scratch directories.

---

## 2. Deployment Targets

### 2.1 SLURM

Applies to both SLURM clusters (`clariden`, `beverin`).

- All jobs (inference server, Benchmarker, NCCL benchmarks, image builds) submit to the
  `normal` partition.
- Account: `csstaff` (or `a-csstaff`). Never use other accounts.
- **Time-limit alignment**: every job in a single experiment — inference server, Benchmarker,
  and any K8s-deployed components — must be configured with the **same** time limit, set
  conservatively enough to cover: model load + CUDA graph capture + inductor compilation
  primer + dataset generation + full sweep.
- Multi-node support via Ray: `tensor_parallel_size` / `gpus_per_node` determine node count.

Access is via distinct FirecREST MCP servers, both registered in Claude Code:

| Cluster | Hardware | FirecREST MCP server |
|---|---|---|
| `clariden` | NVIDIA Grace-Hopper (GH200) | "ML Platform" |
| `beverin` | AMD MI300A | "HPC Platform" |

Per-cluster constraints (driver versions, capstor paths, vLLM flag compatibility) live in §13.

### 2.2 Kubernetes (`breithorn`)

`breithorn` is the single Kubernetes target. It hosts multiple GPU node types:

| Node type | Hardware | Availability |
|---|---|---|
| `gh200` | NVIDIA Grace-Hopper | Available |
| `mi300a` | AMD MI300A | Planned (not yet available) |

- `nodeSelector: beta.kubernetes.io/instance-type: <type>` targets a specific node type.
- A single experiment may pin different components to different node types (e.g. GH200 prefill
  + MI300A decode for prefill-disaggregation studies). The deployment manifest sets the
  `nodeSelector` per component.
- Time limit on K8s-deployed components must match the SLURM `server_time_limit` /
  `benchmarker_time_limit` of the same experiment (see §2.1).
- `kubectl apply --validate=false` is required (Rancher API unreachable from operator laptop).
- The cluster is typically at near-100% GPU utilisation; benchmark scheduling must account for
  this. Orphaned pods from failed runs consume GPUs indefinitely.
- `VLLM_ENABLE_CUDA_COMPATIBILITY=1` must **not** be set for current GH200 drivers (causes
  Error 803). It was required for the old driver 525/535 nodes, which have since been replaced.

---

## 3. Resource Identification

All benchmark-created resources must be labelled so they can be discovered and cleaned up:

- **K8s**: every object (Deployment, Service, Ingress, PVC, Secret) carries
  `app.kubernetes.io/managed-by: inference-benchmarking`.
- **SLURM**: both server and benchmarker sbatch include `#SBATCH --comment=inference-benchmarking`.
- K8s discovery: `kubectl get all,ingress,secret -n ml -l app.kubernetes.io/managed-by=inference-benchmarking`

---

## 4. Resource Lifecycle (Cleanup)

Teardown must run on **both success and failure**, after results are downloaded.

### 4.1 K8s teardown

- Delete: Deployment, Service, Ingress, TLS Secret (`<name>-cert`)
- Cancel the SLURM benchmarker job
- Delete the capstor scratch run directory

### 4.2 SLURM teardown

- Cancel all server jobs
- Cancel the benchmarker job
- Delete the capstor scratch run directory

### 4.3 K8s PVC

- Model-cache PVCs (`model-cache-<model-slug>`) are **intentionally kept** across runs to avoid
  repeated 20-30 min weight downloads.

### 4.4 Run ID uniqueness

- Run IDs include: timestamp + model slug + backend + deployment + 4-hex random suffix.
- The random suffix prevents collision when multiple coordinators start within the same second.

---

## 5. Server Configuration Options

All fields are optional (sensible defaults apply). Wired from `BackendConfig` through
coordinator → backends → Jinja2 templates.

| Field | vLLM flag | Notes |
|---|---|---|
| `tensor_parallel_size` | `--tensor-parallel-size` | |
| `max_model_len` | `--max-model-len` | |
| `max_num_batched_tokens` | `--max-num-batched-tokens` | Must equal `max_model_len` for long-context (avoids chunked-prefill rejection) |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | |
| `kv_cache_dtype` | `--kv-cache-dtype` | e.g. `"fp8"`. Doubles KV capacity but worsens per-request latency due to higher batch concurrency. |
| `enable_prefix_caching` | `--enable-prefix-caching` | Default True. Set False to isolate TTFT from cache artefacts (but prefer unique prompts instead). |
| `safetensors_load_strategy` | `--safetensors-load-strategy` | `"prefetch"` recommended on Lustre/capstor; vLLM misidentifies capstor as CEPH and disables prefetch otherwise. |
| `kv_offloading_size` | `--kv-offloading-size` | Total GiB across all TP ranks (e.g. `400` = 100 GiB/GPU for TP=4). Uses GH200 Grace DRAM at 900 GB/s via NVLink-C2C. |
| `kv_offloading_backend` | `--kv-offloading-backend` | `"native"` (default) or `"lmcache"`. |
| `kv_transfer_config` | `--kv-transfer-config` | JSON string for NIXL KV transfer. |
| `swap_space_gb` | (removed in vLLM v1) | Replaced by `kv_offloading_size`. |
| `speculative_decoding.draft_model` | part of `--speculative-config` JSON | vLLM v0.20+: `--speculative-model` and `--num-speculative-tokens` were removed; use `--speculative-config '{"model":..., "num_speculative_tokens":N, "draft_tensor_parallel_size":M}'` |
| `speculative_decoding.num_speculative_tokens` | part of `--speculative-config` JSON | |
| `speculative_decoding.draft_tensor_parallel_size` | part of `--speculative-config` JSON | Running draft at TP=1 on shared GPUs reduces KV concurrency; use TP matching target only if draft runs on dedicated GPUs. |

---

## 6. Prompt Generation

> Being edited in `SPECIFICATIONS_prompt_generation.md`. Will be merged back here when ready.

---

## 7. Inductor Pre-compilation Primer

vLLM v1 (vllm-cxi v0.20+) uses `torch.inductor` to JIT-compile CUDA kernels for large prefill
sequences (> 512 tokens) **lazily** — on the first request that triggers the path. This one-time
compilation takes ~60 s for 70B target, plus additional time for the draft model path in speculative
decoding.

### Requirements

- The benchmarker must send a **priming request** (20K-token prompt, `max_tokens=1`) before
  the sweep begins, and wait up to 300 s for it to complete.
- The primer must run **inside the container** (`srun --environment=...`); `python3` is not in
  the system PATH on compute nodes outside containers.
- After the primer completes, the first measurement request should exhibit genuine ~1-2 s TTFT
  (not 60 s compile delay).

---

## 8. Load Generation

### 8.1 Server readiness and model-loading tracking

Before the sweep starts, the load generator must, for **each** deployed instance:

- Wait for `/health 200`. Per-instance wait bounded by `server_ready_timeout_s` (default 3600 s;
  see §10.1). If any instance fails to come ready within the timeout, the experiment aborts.
- Parse the per-instance model-loading breakdown from the backend's structured logs / runtime
  API and persist one row per instance into the `instances` table (§11.2) with the
  `model_load_*` fields populated (§10.2).
- Run the inductor pre-compilation primer (§7).

The sweep begins only once **all** instances are ready, profiled, and primed.

### 8.2 Sweep structure

- **Warmup phase**: requests sent but metrics excluded. Long enough for:
  - Inductor JIT compilation to complete after primer (≥ 1 full round of compilation per model)
  - KV cache and queue to reach steady state
- **Measurement phase**: TTFT, ITL, E2E recorded per request.
- **Drain phase**: in-flight requests after measurement window are allowed to complete up to
  `drain_timeout_s`.
- `request_timeout_s`: client-side TTFT hard cutoff; exceeded requests recorded as `success=0`.

### 8.3 Open-loop Poisson arrivals

- Inter-arrival times drawn from `Exp(1/λ)`.
- Each arriving request is routed to one of N server instances according to `routing_strategy`.

### 8.4 Routing strategies

- `random` (default): uniformly random instance selection per request.
- `session_affinity`: `prompt_idx % N` — same prompt always routes to the same instance.
  Enables meaningful prefix-cache benefit across multi-turn sessions. Useful to measure
  the effect of session affinity vs random routing in multi-instance deployments.

---

## 9. Measurement

### 9.1 Metrics recorded per request

- `ttft_ms`: time from send to first token (authoritative SLO metric)
- `tpot_ms`: inter-token latency (mean across output tokens)
- `e2e_ms`: total request time
- `input_tokens`, `output_tokens`, `success`, `error`

### 9.2 Request error tracking

Every failed request (`success=0`) is **kept** in the `requests` table (§11.3) — never dropped —
with its `error` column populated. The classification is used both for diagnosis and for
reporting error rates per λ level (see §12.1).

The `error` column carries `<class>:<detail>`, with class drawn from:

| Class | Triggered by |
|---|---|
| `timeout` | Client-side `request_timeout_s` exceeded before the first token. |
| `http_<status>` | Non-2xx HTTP response (e.g. `http_429` for queue saturation, `http_500` for server crash). |
| `connection` | TCP refusal / reset / DNS failure / TLS handshake error. |
| `server` | 2xx response but the payload signals an error (truncated stream, malformed SSE, etc.). |
| `unknown` | Anything else; the raw exception message is appended for triage. |

Reports aggregate by class so the reader can distinguish queue saturation (`http_429` / `timeout`)
from server-side failure (`http_5xx` / `connection`).

### 9.3 Server metrics (scraped periodically)

- `requests_running`, `requests_waiting`, `gpu_cache_pct`, `spec_accept_rate`

---

## 10. Benchmarker Infrastructure

### 10.1 Health-check timeout

The benchmarker health check must wait at least as long as `server_ready_timeout_s` (default 3600 s)
before giving up. The former 10-minute hardcoded limit was too short for servers with speculative
decoding (dual model load + CUDA graph capture ≥ 15 min).

### 10.2 Model loading time tracking

The Benchmarker records both the **total** time-to-ready and its **individual components**,
so that optimization decisions are driven by data — knowing whether to attack weight load,
graph capture, or compilation requires measuring each in isolation. Components that a given
backend cannot expose are stored as `NULL` rather than collapsed into another bucket.

A single experiment may deploy **multiple instances** of the same configuration (e.g. when
testing request routing across replicas). Each instance is measured independently, so the
load data is stored **per instance** in the dedicated `instances` table (§11.2).

All fields share the `model_load_` prefix to make the group easy to identify:

| Field | Measures |
|---|---|
| `model_load_total_s` | Total: from job start to first `/health 200`. |
| `model_load_weights_s` | Reading model weights from storage into device memory. |
| `model_load_engine_init_s` | Engine/runtime startup not attributable to the other components (tokenizer load, KV cache allocation, distributed init). |
| `model_load_cuda_graph_capture_s` | CUDA graph capture phase. |
| `model_load_inductor_compile_s` | `torch.inductor` JIT compilation primer (large-prefill path; add the draft-model path for speculative decoding). |

- Each component must be parsed from the backend's structured logs or runtime API (per
  backend; see backend-specific notes in §13).
- Reports show both the total and the per-component stack, per instance, with the totals
  aggregated across instances for the experiment summary (see §12).

---

## 11. Results

Per-run results live in a SQLite database file (`run_<id>.db`) with four tables:
`experiments` (one row per sweep), `instances` (one row per deployed server instance),
`requests` (one row per issued request), and `server_stats` (periodic samples of
server-side counters).

### 11.1 `experiments` table

One row per sweep — the configuration and overall outcome of the run.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, PK | Unique identifier (`timestamp + model_slug + backend + deployment + 4-hex random`; see §4.4). |
| `model` | TEXT | Model identifier (HuggingFace ID or path). |
| `backend` | TEXT | Inference engine (`vllm`, `sglang`, `dynamo`). |
| `backend_config` | TEXT (JSON) | Serialized `BackendConfig` — all fields from §5. |
| `dataset_config` | TEXT (JSON) | Serialized dataset configuration (§6). |
| `rate_levels` | TEXT (JSON) | List of λ values (req/s) swept in this run. |
| `warmup_s` | INTEGER | Warmup phase duration in seconds (metrics excluded; see §8.2). |
| `measurement_s` | INTEGER | Measurement phase duration in seconds. |
| `created_at` | TEXT (ISO 8601) | Experiment start timestamp. |

### 11.2 `instances` table

One row per deployed server instance for the experiment. A single experiment may deploy
multiple instances of the same configuration (routing tests, disaggregation studies, multi-
replica deployments); each instance has its own load profile (§10.2).

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Per-experiment instance identifier (stable across the run). Composite PK with `run_id`. |
| `endpoint` | TEXT | URL the load generator targets for this instance (`host:port`). |
| `node` | TEXT | Hosting node — SLURM node name or K8s pod / node-type. `NULL` if not applicable. |
| `model_load_total_s` | REAL | Total time-to-ready for this instance (§10.2). |
| `model_load_weights_s` | REAL | Weights load subcomponent (§10.2). |
| `model_load_engine_init_s` | REAL | Engine/runtime startup subcomponent (§10.2). |
| `model_load_cuda_graph_capture_s` | REAL | CUDA graph capture subcomponent (§10.2). |
| `model_load_inductor_compile_s` | REAL | Inductor compilation primer subcomponent (§10.2). |

Loading-time components a backend cannot expose are stored `NULL` (see §10.2).

### 11.3 `requests` table

One row per issued request — the per-request latency record.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `rate_lambda` | REAL | λ value (req/s) of the sweep step this request belongs to. |
| `request_id` | INTEGER | Per-rate-level request index (monotonic). |
| `ttft_ms` | REAL | Time to first token, milliseconds — authoritative SLO metric (§9.1). |
| `tpot_ms` | REAL | Inter-token latency, mean across the request's output tokens. |
| `e2e_ms` | REAL | End-to-end request time, milliseconds. |
| `input_tokens` | INTEGER | Number of input tokens. |
| `output_tokens` | INTEGER | Number of generated output tokens. |
| `success` | INTEGER | `1` if completed within timeouts; `0` if client-side `request_timeout_s` exceeded or the server returned an error. |
| `error` | TEXT | Error message or class when `success=0`; `NULL` otherwise. |

### 11.4 `server_stats` table

Periodic samples of server-side counters during a sweep step. Sampling cadence is
backend-dependent.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `rate_lambda` | REAL | λ value (req/s) of the sweep step being sampled. |
| `ts` | TEXT (ISO 8601) | Sample timestamp. |
| `requests_running` | INTEGER | Requests currently executing on the server. |
| `requests_waiting` | INTEGER | Requests queued on the server. |
| `gpu_cache_pct` | REAL | KV cache utilization, percent. |
| `spec_accept_rate` | REAL | Speculative-decoding token acceptance rate; `NULL` if speculative decoding disabled. |

### 11.5 Experiment directories

Each completed sweep produces an `experiments/YYYY-MM-DD_description/` folder containing:

- `benchmark_config.yaml` (copy of the input config for provenance)
- the run's SQLite DB file (`run_<id>.db`)
- deployment artifacts used for the run (sbatch scripts, Kubernetes YAML, Dockerfile)
- the executed report notebook and its rendered outputs (see §12)

---

## 12. Reporting

The Reports generator produces a Jupyter notebook from the centralized results database
and writes it back into the experiment directory.

### 12.1 Report notebook (`experiments/template_report.ipynb`)

Every experiment report must include:

- Experiment title and description
- Configuration summary table (model, TP, KV dtype, spec dec, SLO, etc.)
- **Model loading times**: per instance (from the `instances` table, §11.2),
  `model_load_total_s` plus the per-component breakdown (`model_load_weights_s`,
  `model_load_engine_init_s`, `model_load_cuda_graph_capture_s`,
  `model_load_inductor_compile_s`) — see §10.2.
- TTFT p50/p95 vs λ plot (log scale) with SLO line
- ITL p50/p95 vs λ plot
- Failure rate bar chart (bottom panel of each plot)
- Raw per-rate-level data table

### 12.2 Notebook output

The executed notebook (`report.ipynb`) and its rendered plots (`ttft.png`, `itl.png`)
are written into the corresponding `experiments/YYYY-MM-DD_description/` folder (§11.5).

---

## 13. Cluster-Specific Constraints (Alps / GH200)

### 13.1 vLLM flag compatibility (vllm-cxi v0.20.x)

| Flag | Status | Notes |
|---|---|---|
| `--speculative-model` / `--num-speculative-tokens` | ❌ Removed | Use `--speculative-config '{"model":..., "num_speculative_tokens":N, "draft_tensor_parallel_size":M}'` |
| `--swap-space` | ❌ Removed | Use `--kv-offloading-size` |
| `--kv-offloading-size` | ✅ Available since v0.20.0 | Total GiB across all TP ranks |
| `--kv-cache-dtype fp8` | ✅ Works | Native FP8 hardware on SM90 (GH200) |
| `--enable-prefix-caching` | ✅ Works | |
| `--safetensors-load-strategy` | ✅ Works | |
| `--speculative-config` | ✅ Works | JSON string |

### 13.2 Capstor filesystem

- capstor is **Lustre**, not Ceph. vLLM misidentifies it and disables auto-prefetch.
- Use `--safetensors-load-strategy=prefetch` to force Lustre-optimised parallel shard loading.
- Expected gain: ~10–20 s on a 131 GiB checkpoint.

### 13.3 GH200 KV cache capacity (70B, TP=4)

```
Available KV per GPU  = (96 GiB × gpu_memory_utilization) − (140 GiB / 4 GPUs)
                      ≈ 51 GiB at 0.90

KV per 25K-token request per GPU  ≈ 80 KB/tok × 25,000 = 1.91 GiB
Max concurrent requests            ≈ 51 / 1.91  ≈ 28

With --kv-offloading-size 400 (100 GiB/GPU via Grace DRAM):
Additional KV capacity  = 100 / 1.91 ≈ 52 additional slots
Total concurrent        ≈ 80 slots
```

### 13.4 Speculative decoding with shared GPUs

- Running draft at TP=4 on the **same** 4 GPUs as the 70B target is counterproductive:
  NCCL allreduce overhead for 5 draft passes per cycle (320 extra syncs) erases the
  memory bandwidth benefit.
- Optimal configuration when GPUs are shared: **draft TP=1**, accept reduced GPU-0 KV capacity.
- Optimal configuration when dedicated GPUs are available: **draft on separate node/GPUs**.

### 13.5 Inductor compilation timing

- **Target model (70B)**: first large-prefill request triggers ~60 s compile.
- **Draft model (8B)**: adds separate compile path, typically 30–60 s additional.
- Total primer wait on a cold server: up to 120 s. Primer timeout is set to 300 s.
- After compilation, TTFT drops to ~1.7 s for a 25K-token prompt at idle.
- Compilation artifacts are **not persisted** (`local_cache_dir: None`) — every restart re-compiles.

---

## 14. Known Issues & Workarounds

| Issue | Workaround |
|---|---|
| Prefix cache inflates TTFT at low rates with small prompt pools | Use `num_prompts ≥ pool coverage` + unique headers; or `dataset_source: longbench` |
| Filler-text synthetic prompts → ~0% speculative acceptance rate | Use `dataset_source: longbench` for speculative decoding experiments |
| `datasets` library absent from vllm-cxi container | LongBench downloaded via `urllib` + `zipfile` (no library needed) |
| vLLM misidentifies capstor as CEPH → disables prefetch | Set `safetensors_load_strategy: "prefetch"` explicitly |
| Same-second coordinator starts → run_id collision → teardown deletes sibling's capstor dir | Run ID includes 4-hex random suffix |
| Benchmarker health check (old: 10 min) too short for speculative dec startup | Health check timeout now equals `server_ready_timeout_s` |
| `benchmarker_time_limit` of 1 h too short when model load takes >30 min | Set ≥ 1.5 h for speculative decoding experiments |

---

## 15. Open Items

See `TODOs.md` for tracked future work. Key items:

- **Persist inductor compilation cache** (`--compilation-config '{"local_cache_dir": "..."}'`)
  to avoid 60 s recompile on every server restart.
- **Trim CUDA graph capture sizes** to match actual concurrency ceiling (~28 slots for 70B 25K
  context); captures 1–512 is wasteful, ~32 suffices.
- **Test `--kv-offloading-size 400`** for GH200 KV extension via Grace DRAM.
  Implementation complete, not yet run.
- **Session affinity experiment**: compare round-robin vs sticky routing for multi-instance
  deployments to quantify prefix-cache benefit in production.
- **NIXL disaggregated prefill/decode**: vLLM v1 startup logs show `NIXL is available`;
  configuration via `--kv-transfer-config` not yet implemented.
