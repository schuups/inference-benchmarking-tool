# Inference Benchmarking — Specification

This document enumerates all requirements captured through design and implementation.
It supersedes the scattered notes in ARCHITECTURE.md (which describes an earlier design state)
and consolidates everything into a single reference.

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
- **Separation of concerns**: the inference server and the load generator always run as separate
  SLURM allocations; load-generator CPU never competes with GPU inference.
- **Clean cluster state**: all deployed resources must be cleaned up after every run — on both
  success and failure paths. No orphaned jobs, pods, services, secrets, or scratch directories.

---

## 2. Deployment Targets

### 2.1 SLURM (clariden / Alps)

- Server job submitted to the `debug` partition (GPU); benchmarker to `normal` (CPU).
- Account: `csstaff` (or `a-csstaff`). Never use other accounts.
- Both `debug` and `normal` partition time limits must be respected (`debug` ≤ 1.5 h on clariden).
- Multi-node support via Ray: `tensor_parallel_size` / `gpus_per_node` determine node count.
- `server_time_limit` and `benchmarker_time_limit` must be set conservatively; exceeding the
  debug partition hard limit returns `sbatch: error: Requested time limit is invalid`.
- The `benchmarker_time_limit` must be long enough for: model load + CUDA graph capture +
  inductor compilation primer + prompt generation + full sweep.

### 2.2 Kubernetes (Alps, namespace `ml`)

- GPU nodes are `nid006xxx` (arm64 GH200, driver 590 / CUDA 13.1). **Not** `breithorn-worker-*`
  (CPU-only).
- `nodeSelector: beta.kubernetes.io/instance-type: gh200` is required to target GPU nodes.
- `VLLM_ENABLE_CUDA_COMPATIBILITY=1` must **not** be set for driver 590; it causes Error 803.
  (It was required for the old driver 525/535 nodes, which have since been replaced.)
- `kubectl apply --validate=false` is required (Rancher API unreachable from operator laptop).
- K8s cluster is typically at near-100% GPU utilisation; benchmark scheduling must account for
  this. Orphaned pods from failed runs consume GPUs indefinitely.

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

## 6. Load Generation & Measurement

### 6.1 Sweep structure

- **Warmup phase**: requests sent but metrics excluded. Long enough for:
  - Inductor JIT compilation to complete after primer (≥ 1 full round of compilation per model)
  - KV cache and queue to reach steady state
- **Measurement phase**: TTFT, ITL, E2E recorded per request.
- **Drain phase**: in-flight requests after measurement window are allowed to complete up to
  `drain_timeout_s`.
- `request_timeout_s`: client-side TTFT hard cutoff; exceeded requests recorded as `success=0`.

### 6.2 Open-loop Poisson arrivals

- Inter-arrival times drawn from `Exp(1/λ)`.
- Each arriving request is routed to one of N server instances according to `routing_strategy`.

### 6.3 Routing strategies

- `random` (default): uniformly random instance selection per request.
- `session_affinity`: `prompt_idx % N` — same prompt always routes to the same instance.
  Enables meaningful prefix-cache benefit across multi-turn sessions. Useful to measure
  the effect of session affinity vs random routing in multi-instance deployments.

### 6.4 Metrics recorded per request

- `ttft_ms`: time from send to first token (authoritative SLO metric)
- `tpot_ms`: inter-token latency (mean across output tokens)
- `e2e_ms`: total request time
- `input_tokens`, `output_tokens`, `success`, `error`

### 6.5 Server metrics (scraped periodically)

- `requests_running`, `requests_waiting`, `gpu_cache_pct`, `spec_accept_rate`

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

## 8. Prompt Generation

### 8.1 Location

Prompts are generated **on the SLURM benchmarker node** from `dataset_config` in `run_config.json`.
This avoids the FirecREST 5 MB direct-upload limit and allows arbitrarily large prompt pools.
The coordinator no longer uploads `prompts.json`.

### 8.2 Prompt uniqueness requirement

Every prompt must start with a distinct token block so that the vLLM prefix cache does not serve
synthetic cache hits. Requirements:
- Each prompt begins with a unique `[prompt-NNNNNN]` or `[session-NNNNNN]` header.
- Without this, filler-text prompts share identical first blocks → 100% cache hit rate →
  TTFT drops to ~100 ms regardless of server load (artefact, not real performance).

### 8.3 Supported dataset sources

Controlled by `dataset_config.dataset_source` in the benchmark YAML:

| Value | Description |
|---|---|
| `"synthetic"` (default) | Filler text with unique `[prompt-NNNNNN]` headers. No network required. |
| `"longbench"` | LongBench code tasks (`lcc` + `repobench-p`) downloaded from HuggingFace via `urllib` as a single `data.zip`. No extra libraries required. Falls back to synthetic on failure. |

### 8.4 LongBench specifics

- Tasks: `lcc` (Long Code Completion, real Python/C++/Java files, ~13–22 K tokens) and
  `repobench-p` (repository-level Python completion, ~14–22 K tokens).
- Content: real GitHub repositories, appropriate for speculative decoding acceptance rate
  measurement with same-family draft/target model pairs.
- Length filter: examples are accepted if their token count falls within 40–160% of `input_length.mean`.
- Pool is repeated (with unique session headers) if `num_prompts` exceeds available examples.

### 8.5 Notes on dataset suitability

- **Synthetic prompts**: acceptable for latency and throughput benchmarking but produce
  near-zero speculative decoding acceptance rates (random text is unpredictable).
- **LongBench / real code**: required for meaningful speculative decoding acceptance rate
  measurements. Both Apertus-8B and Apertus-70B were trained on the same data, so
  same-family speculative acceptance rates should be 0.5–0.7 on real code.

---

## 9. Benchmarker Infrastructure

### 9.1 Health-check timeout

The benchmarker health check must wait at least as long as `server_ready_timeout_s` (default 3600 s)
before giving up. The former 10-minute hardcoded limit was too short for servers with speculative
decoding (dual model load + CUDA graph capture ≥ 15 min).

### 9.2 Model loading time tracking

- The benchmarker records `SERVER_LOAD_TIME_S` = time from job start to first `/health 200`.
- Passed to the load generator via `--server-load-time-s`.
- Stored in `experiments.server_load_time_s` (SQLite column, backward-compatible via `ALTER TABLE`).
- Shown in every report notebook under "Model loading times".

---

## 10. Results & Reporting

### 10.1 Database schema

SQLite per-run file (`run_<id>.db`):

```
experiments   run_id, model, backend, backend_config, dataset_config,
              rate_levels, warmup_s, measurement_s, created_at, server_load_time_s
requests      run_id, rate_lambda, request_id, ttft_ms, tpot_ms, e2e_ms,
              input_tokens, output_tokens, success, error
server_stats  run_id, rate_lambda, ts, requests_running, requests_waiting,
              gpu_cache_pct, spec_accept_rate
```

### 10.2 Report notebook (`experiments/template_report.ipynb`)

Every experiment report must include:

- Experiment title and description
- Configuration summary table (model, TP, KV dtype, spec dec, SLO, etc.)
- **Model loading times** per environment (from `server_load_time_s`)
- TTFT p50/p95 vs λ plot (log scale) with SLO line
- ITL p50/p95 vs λ plot
- Failure rate bar chart (bottom panel of each plot)
- Raw per-rate-level data table

### 10.3 Experiment directories

Each completed sweep produces an `experiments/YYYY-MM-DD_description/` folder containing:
- `benchmark_config.yaml` (copy for provenance)
- `report.ipynb` (executed notebook)
- `ttft.png`, `itl.png`

---

## 11. Cluster-Specific Constraints (Alps / GH200)

### 11.1 vLLM flag compatibility (vllm-cxi v0.20.x)

| Flag | Status | Notes |
|---|---|---|
| `--speculative-model` / `--num-speculative-tokens` | ❌ Removed | Use `--speculative-config '{"model":..., "num_speculative_tokens":N, "draft_tensor_parallel_size":M}'` |
| `--swap-space` | ❌ Removed | Use `--kv-offloading-size` |
| `--kv-offloading-size` | ✅ Available since v0.20.0 | Total GiB across all TP ranks |
| `--kv-cache-dtype fp8` | ✅ Works | Native FP8 hardware on SM90 (GH200) |
| `--enable-prefix-caching` | ✅ Works | |
| `--safetensors-load-strategy` | ✅ Works | |
| `--speculative-config` | ✅ Works | JSON string |

### 11.2 Capstor filesystem

- capstor is **Lustre**, not Ceph. vLLM misidentifies it and disables auto-prefetch.
- Use `--safetensors-load-strategy=prefetch` to force Lustre-optimised parallel shard loading.
- Expected gain: ~10–20 s on a 131 GiB checkpoint.

### 11.3 GH200 KV cache capacity (70B, TP=4)

```
Available KV per GPU  = (96 GiB × gpu_memory_utilization) − (140 GiB / 4 GPUs)
                      ≈ 51 GiB at 0.90

KV per 25K-token request per GPU  ≈ 80 KB/tok × 25,000 = 1.91 GiB
Max concurrent requests            ≈ 51 / 1.91  ≈ 28

With --kv-offloading-size 400 (100 GiB/GPU via Grace DRAM):
Additional KV capacity  = 100 / 1.91 ≈ 52 additional slots
Total concurrent        ≈ 80 slots
```

### 11.4 Speculative decoding with shared GPUs

- Running draft at TP=4 on the **same** 4 GPUs as the 70B target is counterproductive:
  NCCL allreduce overhead for 5 draft passes per cycle (320 extra syncs) erases the
  memory bandwidth benefit.
- Optimal configuration when GPUs are shared: **draft TP=1**, accept reduced GPU-0 KV capacity.
- Optimal configuration when dedicated GPUs are available: **draft on separate node/GPUs**.

### 11.5 Inductor compilation timing

- **Target model (70B)**: first large-prefill request triggers ~60 s compile.
- **Draft model (8B)**: adds separate compile path, typically 30–60 s additional.
- Total primer wait on a cold server: up to 120 s. Primer timeout is set to 300 s.
- After compilation, TTFT drops to ~1.7 s for a 25K-token prompt at idle.
- Compilation artifacts are **not persisted** (`local_cache_dir: None`) — every restart re-compiles.

---

## 12. Known Issues & Workarounds

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

## 13. Open Items

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
