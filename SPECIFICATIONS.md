# Inference Benchmarking — Specification

This document enumerates all requirements captured through design and implementation.

## Table of Contents

1. [Guiding Principles](#1-guiding-principles)
2. [Deployment Targets](#2-deployment-targets)
3. [Resource Identification](#3-resource-identification)
4. [Resource Lifecycle (Cleanup)](#4-resource-lifecycle-cleanup)
5. [Server Configuration Options](#5-server-configuration-options)
6. [Features Under Test](#6-features-under-test)
7. [Prompt Generation](#7-prompt-generation)
8. [System Performance Pre-checks](#8-system-performance-pre-checks)
9. [Inductor Pre-compilation Primer](#9-inductor-pre-compilation-primer)
10. [Load Generation](#10-load-generation)
11. [Measurement](#11-measurement)
12. [Benchmarker Infrastructure](#12-benchmarker-infrastructure)
13. [Results](#13-results)
14. [Reporting](#14-reporting)
15. [Cluster-Specific Constraints (Alps / GH200)](#15-cluster-specific-constraints-alps--gh200)
16. [Known Issues & Workarounds](#16-known-issues--workarounds)
17. [Open Items](#17-open-items)

---

## 1. Guiding Principles

- **Laptop-orchestrated**: all coordination runs on the operator's laptop; no coordinator node
  is allocated on the cluster.
- **Backend-agnostic**: vLLM, sglang, and NVIDIA Dynamo are first-class backends. Adding a new
  one requires a new EDF template, sbatch template, and K8s deployment template only.
- **Open-loop stochastic load generation**: requests are issued by configurable arrival
  processes (Poisson and burst-aware variants — §10.3) at mean rate λ, independent of server
  completions. This faithfully models queuing behaviour — backlog formation, saturation,
  latency amplification — under both steady and bursty load.
- **Reproducible by config**: a single YAML file fully specifies a sweep. Re-running the same
  file must produce comparable results.
- **Scenario-disclosed results**: every result carries a **scenario manifest** stating what
  the scenario models (e.g. agentic-coding-style large prompts with follow-up turns), what
  it explicitly does *not* model (e.g. no image inputs, no audio, no reasoning traces), and
  the numeric assumptions baked in (e.g. follow-up turn probability, max output tokens).
  Reports surface this manifest prominently so downstream plots cannot be read out of
  context (§13.8, §14.1).
- **Separation of concerns**: the inference server runs on its own (GPU) allocation. The
  **Benchmarker** is a separate SLURM allocation hosting the **dataset generator** and the
  **load generator** as distinct, sequential phases — prompt preparation always completes
  before measurement begins. No benchmark CPU work competes with GPU inference, and dataset
  generation never overlaps with load generation.
- **Validated foundation**: every experiment is preceded by synthetic micro-benchmarks
  (NCCL, GPU memory, system memory, storage, network) that verify baseline hardware
  performance against per-system reference values (§8). The pre-checks run in the
  **exact software environment** the LLM engine will run in moments later — same image,
  same env vars, same mounts, same NUMA pinning, same library / NCCL configuration —
  so what is measured is the foundation the engine actually sits on, not a sibling
  environment. A degraded foundation surfaces a warning before the LLM sweep runs and
  offers the operator the chance to abort.
- **Observed execution**: throughout every sweep, GPU, CPU, memory, storage, and network
  telemetry is sampled on every inference-server node (§11.5). Untapped hardware headroom
  is then distinguishable from genuine saturation when interpreting latency / throughput
  results.
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

Per-cluster constraints (driver versions, capstor paths, vLLM flag compatibility) live in §15.

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

## 6. Features Under Test

The framework systematically benchmarks the following inference-serving features. Each
feature is exercised by one or more scenarios (see the scenario taxonomy in the README)
and configured via the knobs in §5. Coverage is the basis of the comparative evidence
the tool produces — adding a new feature requires extending this list, defining its
config knobs in §5, binding it to at least one scenario, and surfacing its marginal
effect in reports (§14).

| Feature | Why it matters | Where configured | Procurement implication |
|---|---|---|---|
| **Automatic prefix caching** | Reduces TTFT for sessions sharing prompt prefixes; critical for chat and AI-assisted coding. | `enable_prefix_caching` (§5) | Cache-friendly KV memory hierarchy; cache hit-rate as a procurement metric. |
| **KV-cache offloading** | Extends effective KV capacity by spilling to host DRAM / unified memory; trades per-request latency for concurrency. | `kv_offloading_size`, `kv_offloading_backend` (§5) | **Memory-layer sizing decisions** — HBM vs Grace-DRAM vs CXL. Offloading bandwidth profiles drive host-DRAM-per-GPU sizing and the choice of unified-memory / CXL fabrics for next-generation systems. |
| **KV-cache reuse across requests** | Identical or partially-overlapping prefixes from different requests reuse already-computed KV; effectiveness depends on routing. | `enable_prefix_caching` (§5) + `routing_strategy` (§10.4) | KV memory pressure under realistic locality; informs replica-pool sizing. |
| **Speculative decoding** | Improves decode throughput when a smaller draft model proposes tokens accepted by the target. | `speculative_decoding.*` (§5) | Compute headroom for draft model; memory budget for two-model deployments. |
| **Continuous batching** | Schedules new requests into running batches without waiting for current ones to finish — the dominant throughput optimization for online serving. | Backend default; not directly exposed | Scheduler responsiveness characterisation; admission-control budget. |
| **MoE expert routing** | Token-to-expert dispatch and load balance govern memory pressure and inter-GPU traffic. | Model-internal | Interconnect sizing for all-to-all expert traffic; hot-expert memory pressure. |
| **Quantization (weights / KV / activation)** | Trades model fidelity and memory footprint against throughput. | `kv_cache_dtype` (§5); weight quantization via backend | Memory hierarchy: lower-precision math support vs higher-precision storage. |
| **Disaggregated prefill / decode** | Splits compute-heavy prefill from memory-bandwidth-heavy decode across different accelerator classes. | `kv_transfer_config` (§5); per-component `nodeSelector` (§2.2) | Heterogeneous accelerator procurement; interconnect bandwidth between roles. |
| **Multi-replica routing and session affinity** | Distributes load across replicas; `session_affinity` preserves prefix-cache hits at the cost of fairness. | `routing_strategy` (§10.4) | Ingress / load-balancer requirements; cache-locality vs replica-fairness trade-off. |
| **Schema-constrained decoding (JSON / XML)** | Forces structured outputs for tool calls; some backends parse natively, others post-hoc. | Per-request decoding parameters | Backend-specific tool-calling cost; structured-output validity rate. |
| **Hardware elasticity / auto-scaling** | Time-to-scale-up, request loss during scale events, and pre-warmed-pool sizing — critical for bursty workloads. | Cluster-level orchestration | **Elasticity requirements for CSCS vClusters** — auto-scale latency budgets, pre-warmed-pool sizing, scale-down hysteresis. |

Each feature's contribution to latency, throughput, error rate, and hardware utilisation
(§11.5) is recorded per sweep. Reports plot the marginal effect of enabling / disabling
individual features so procurement evidence can isolate the value of each.

---

## 7. Prompt Generation

### 7.1 Location and ownership

Prompts are produced on the **Benchmarker** SLURM allocation by its **dataset-generator**
subcomponent, from `dataset_config` in the benchmark YAML, sequentially before the
load-generator phase starts (per §1's separation of concerns). Generating on the cluster
sidesteps the FirecREST 5 MB direct-upload limit and allows arbitrarily large prompt
pools — the coordinator never ships prompt data to the cluster.

### 7.2 Key concepts

Three artefacts cooperate to produce a benchmark dataset:

- **Scenario registry** (`tool/scenarios/<slug>.yaml`) — the canonical declaration of what
  each scenario is: its source, length distributions, multi-turn structure, session mode,
  tool catalog (if agentic), modalities (if multimodal), and the `modelled` / `not_modelled`
  lists that feed the scenario manifest (§13.8).
- **`dataset_config`** in the benchmark YAML — the per-run knobs: which scenario to run,
  the master seed, `num_prompts`, and any per-run overrides to registry defaults.
- **Dataset generator** — reads both, materializes the prompt pool on capstor scratch,
  and emits the scenario manifest as a structured side-effect for the experiment row (§13.1).

The registry is data, not code: adding a new scenario does not require editing the
generator. Registry entries are versioned with the repo; changes to a scenario's
`modelled` / `not_modelled` lists are reviewable in PRs.

### 7.3 Scenario registry

Each registry entry is a YAML file with the schema below. Fields not relevant to a given
scenario are omitted (e.g. `tools` only applies to agentic scenarios).

```yaml
name: agentic-coding-large-prompt
summary: AI-assisted coding session with large initial prompt and short follow-ups.
maturity: established                # established | emerging | exploratory
modalities: [text]                   # text, image (audio and video deferred — §7.11)

source:
  kind: longbench                    # see §7.5
  config: { tasks: [lcc, repobench-p] }

input_length:
  distribution: lognormal            # lognormal | normal | fixed
  params: { mean: 20000, sigma: 0.3, min: 8000, max: 64000 }

output_length:
  distribution: lognormal
  params: { mean: 512, sigma: 0.4, min: 32, max: 4096 }

session:
  mode: sequential                   # sequential | open_loop  (see §7.9)
  turns_per_session:
    distribution: lognormal
    params: { mean: 3, sigma: 0.4, min: 1, max: 12 }
  prefix_strategy: append_delta      # only supported strategy (see §7.9)
  think_time_ms: { distribution: lognormal, params: { mean: 1500, sigma: 0.4 } }  # sequential mode only

tools: []                            # only for agentic scenarios (§7.10)

manifest:
  modelled:
    - "long multi-turn prompts (8K–64K input tokens)"
    - "follow-up turns reusing the initial context"
  not_modelled:
    - "no image inputs"
    - "no tool-call interleaving"
```

`assumptions` is **not** stored in the registry — it is computed at runtime from the
actual `dataset_config` consumed (§7.13).

### 7.4 dataset_config schema

The benchmark YAML's `dataset_config` block:

| Field | Type | Required | Notes |
|---|---|---|---|
| `scenario` | string | yes | Must match a registered scenario name. |
| `num_prompts` | integer | yes | Size of the generated prompt pool. The load generator's request stream draws from this pool. |
| `seed` | integer | yes | Master seed; sub-seeds derived deterministically (§7.12). |
| `input_length` | object | no | Per-run override of the registry's `input_length` distribution. |
| `output_length` | object | no | Per-run override of the registry's `output_length` distribution. |
| `session` | object | no | Per-run override of session fields. |
| `tokenizer_id` | string | no | Override the tokenizer (defaults to the target model's; see §7.8). |
| `source_overrides` | object | no | Source-specific overrides (e.g. LongBench task subset). |

Any field absent from `dataset_config` is inherited from the scenario registry.
Validation aborts the run before submission if `scenario` is missing or refers to an
unknown registry entry.

### 7.5 Dataset sources

The `source.kind` enum, with v1 scope:

| `kind` | Description |
|---|---|
| `synthetic` | Filler text with unique `[prompt-NNNNNN]` headers. No network required. Acceptable for latency/throughput; near-zero speculative-decoding acceptance rate (§7.15). |
| `longbench` | LongBench code tasks (`lcc`, `repobench-p`) downloaded from HuggingFace via `urllib` + `zipfile`. Real GitHub source; suitable for speculative-decoding measurements. |
| `reasoning_trace_replay` | Replays recorded reasoning traces from public datasets (GSM8K-with-cot, MATH, AIME, R1-distill traces). Output length is taken from the recorded target and overrides the scenario's `output_length`. |
| `tool_registry` | Drives agentic scenarios; prompts are generated from a tool catalog + task template registry (§7.10). |
| `image_corpus` | Image inputs paired with text prompts. v1: fixed corpus (e.g. COCO); image-token cost counted into `input_tokens` (§7.11). |

**Audio and video are intentionally not in v1.** Tracked as future work in `TODOs.md`.
A scenario declaring `modalities: [audio]` or `modalities: [video]` is rejected at
registry-load time until support lands.

### 7.6 Prompt uniqueness requirement

Every prompt must start with a distinct token block so that the engine's automatic prefix
cache — one of the features under test (§6 features table), implemented by vLLM, SGLang,
and equivalents — does not serve synthetic cache hits.

- Single-turn scenarios: each prompt begins with a unique `[prompt-NNNNNN]` header.
- Multi-turn scenarios: each session begins with a unique `[session-NNNNNN]` header that
  is reused across the session's turns, so the engine's prefix cache *does* hit on the
  shared session prefix (this is the locality the benchmark is meant to expose; see §7.9).
- Without this discipline, filler-text prompts share identical first blocks → 100% cache
  hit rate → TTFT drops to ~100 ms regardless of server load (artefact, not real
  performance).

### 7.7 Length distributions and output length control

**Input length** distribution shape is **per scenario** (declared in the registry; §7.3).
Supported shapes: `lognormal` (truncated), `normal` (truncated), `fixed`. Heavy-tailed
`lognormal` matches observed LLM-workload distributions and is the recommended default;
`fixed` is useful for isolation studies.

**Output length** is controlled per request:

- Each prompt carries a target `max_tokens` sampled from the scenario's `output_length`
  distribution.
- The load generator sends `max_tokens=<sampled>` **and** `ignore_eos=True`, forcing the
  model to emit exactly that many decode tokens.

`ignore_eos` makes decode cost reproducible across runs and across models — measured
TPOT and `output_tokens` no longer depend on each model's stopping behaviour for a given
prompt.

Sources that carry ground-truth output lengths (`reasoning_trace_replay`,
`tool_registry`) override the sampled value with the recorded target.

### 7.8 Tokenization

Length filtering, length-distribution sampling, and the `input_tokens` field in the
generated dataset all use the **target model's tokenizer**, loaded by HuggingFace ID.
The tokenizer is fetched on the Benchmarker at dataset-generation time.

- Changing the target model invalidates the generated dataset and triggers regeneration
  (the dataset is per-run on capstor scratch; §7.14).
- For draft/target same-family pairs (e.g. Apertus-8B draft + Apertus-70B 1.5 target),
  tokenizers are identical and only one is loaded.
- For cross-family draft/target pairs the **target's** tokenizer is authoritative; any
  draft-tokenizer mismatch is logged but does not block the run.

### 7.9 Multi-turn / session structure

Multi-turn scenarios produce N turns per session, where N is sampled from the scenario's
`session.turns_per_session` distribution (§7.3).

**Prefix strategy** is always **`append_delta`**: turn K+1's prompt = full prior
transcript + new user turn. The engine's prefix cache reuses the shared prefix naturally
— exactly as real chat / agentic clients do. (A `regenerate` strategy was considered
but rejected: it defeats the prefix cache and is better expressed as a separate ablation
by disabling prefix caching at the backend.)

**Session mode** governs how follow-up turns interact with the load generator's open-loop
arrival process (§10.3):

| `session.mode` | Follow-up behaviour | Use for |
|---|---|---|
| `open_loop` (default) | All turns of a session are scheduled by the arrival process; turn K+1 fires on its own schedule regardless of when turn K completed. Preserves open-loop queueing semantics throughout. | RAG-style independent queries against a shared long-lived prefix; reasoning workloads; any scenario where turn ordering is incidental. |
| `sequential` | Turn K+1 is sent only after turn K's response has been received, plus a `think_time_ms` delay sampled from the registry's distribution. Introduces closed-loop coupling *within a session*; cross-session arrivals remain open-loop (session starts are still Poisson per §10.3). | Conversational chat; agentic-coding follow-ups; any scenario where a follow-up message cannot meaningfully precede its predecessor's response. |

Sequential sessions document their closed-loop-within-session coupling in the
`assumptions` field of the scenario manifest (§7.13) so reports interpret tail-latency
results accordingly.

### 7.10 Agentic scenarios

Agentic scenarios fan one user request into many model invocations (think → tool call →
tool result → think …). The unit of measurement is the **agent task** (§13.7); child
model invocations are recorded in the `requests` table (§13.3) with their parent's
`agent_task_id`.

**Tool registry.** `tool/tools/<tool-name>.yaml` declares each available tool:

```yaml
name: read_file
schema:                              # JSON schema for the tool call
  type: object
  properties:
    path: { type: string }
  required: [path]
result_size:                         # distribution of injected tool-result token counts
  distribution: lognormal
  params: { mean: 2000, sigma: 0.5, min: 50, max: 32000 }
result_content_source:               # how the synthesized tool-result body is produced
  kind: longbench                    # synthetic | longbench | static
```

Each scenario in the registry references the tools it offers via
`tools: [read_file, run_tests, …]`. This produces the **bimodal output distribution** the
README §3 calls out (tiny structured tool calls drawn from the schema + large injected
results drawn from each tool's `result_size`) without requiring scenario-level
distribution tuning.

**Fan-out template.** Each agentic scenario declares a fan-out template specifying the
shape of the agent task: turn types (`think`, `tool_call`, `tool_result`), allowed
transitions, and the distribution over the number of think/tool cycles per task. The
template is consumed by the load generator to drive `sequential` execution of the task's
child requests. (Template DSL grammar is unspecified — tracked in `TODOs.md`.)

**Schema-constrained decoding.** When the model emits a tool call, the load generator
requests structured-output decoding against the tool's JSON schema. The result's validity
is recorded per request in `requests.structured_output_valid` (§13.3) and aggregated per
task into `agent_tasks.task_tool_calls_valid` (§13.7).

### 7.11 Multimodal prompts (v1)

v1 supports **image** inputs alongside text; **audio and video are deferred** (tracked
in `TODOs.md`).

- `image_corpus`: image inputs drawn from a fixed corpus (e.g. COCO) and paired with a
  text prompt template from the scenario registry. The number of images per prompt
  follows a scenario-declared distribution.
- Image token cost is computed by the target model's tokenizer (or the backend's
  multimodal-aware reporter when the tokenizer cannot tokenize raw image bytes) and
  counted into `input_tokens`. Backends that do not surface per-modality token counts
  fall back to a model-card-derived per-image token estimate, recorded in the manifest's
  `assumptions`.
- A scenario declaring `modalities: [audio]` or `modalities: [video]` is rejected at
  registry-load time until the corresponding modality support lands.

### 7.12 Seeding and determinism

`dataset_config.seed` is a single integer. The generator derives per-axis sub-seeds
deterministically by hashing the master seed with a constant per-axis salt:

```python
sub_seed(seed, axis) = blake2b(f"{seed}:{axis}".encode(), digest_size=8).digest()
# axes: "header", "length_input", "length_output", "selection", "turns",
#       "tool_choice", "image_pick", "thinktime"
```

Reproducibility contract:

- Same `dataset_config` + same scenario-registry revision + same target tokenizer →
  identical prompt pool (byte-for-byte).
- Changing the target model triggers regeneration (different tokenizer → different
  length filtering and different sampled lengths).

### 7.13 Scenario manifest emission

Per §13.8, every experiment row carries a structured `scenario_manifest` JSON object.
The dataset generator produces it as a side-effect of running, combining:

- `name`, `summary`, `maturity`, `modelled`, `not_modelled` — copied **verbatim** from
  the scenario registry entry.
- `assumptions` — auto-filled at runtime from the `dataset_config` actually consumed,
  including: input / output length distribution shape + parameters; turns-per-session
  distribution; session mode; prefix strategy; source `kind` + relevant source config;
  master seed; tokenizer ID; modalities; tool list (agentic only). Sufficient to
  reconstruct what the run measured without re-reading the registry at a specific revision.

Validation: the run aborts before load-generation if any required manifest field is
missing or fails schema validation.

### 7.14 Persistence and source-failure semantics

**Persistence.** Generated datasets live on the Benchmarker's capstor scratch directory
for the duration of the run, reused across sweep steps within the experiment, and
deleted by the §4 teardown. They are **not** copied into `experiments/<run>/` — only the
manifest (persisted on `experiments.scenario_manifest`) and the master seed are needed
for reproduction; the dataset can always be regenerated from those plus the scenario-
registry revision recorded alongside.

**Source-failure semantics.** A failure of any dataset source (LongBench download error,
HuggingFace unreachable, tool registry malformed, image corpus mount missing, etc.)
**aborts the run** with a clear error. There is no silent fallback to synthetic data —
this avoids the trap where, e.g., a speculative-decoding experiment silently degrades to
filler text and reports ~0% acceptance as if it were a property of the model.

### 7.15 Notes on dataset suitability

- **Synthetic prompts**: acceptable for latency and throughput benchmarking but produce
  near-zero speculative-decoding acceptance rates (random text is unpredictable).
- **LongBench / real code**: required for meaningful speculative-decoding acceptance rate
  measurements. For same-family draft/target pairings (e.g. Apertus-8B as the draft for
  the Apertus-70B 1.5 target — Apertus-8B is of interest exclusively in the draft role),
  acceptance rates on real code are typically in the 0.5–0.7 range.
- **Reasoning-trace replay**: required when measuring speculative-acceptance for
  reasoning workloads — synthetic prompts produce the same near-zero rate as for code.

---

## 8. System Performance Pre-checks

Synthetic micro-benchmarks executed in the **exact software environment in which the LLM
engine will be launched moments later** — same container image, same environment variables,
same mount points, same NUMA / CPU pinning, same library load paths, same NCCL / RCCL
configuration — immediately before the LLM engine binary starts. Without this gate, a
degraded NCCL fabric, slow weight storage, or under-clocked memory silently biases LLM
benchmark results — good throughput / latency numbers measured on top of a degraded
foundation are misleading. Running the checks in any sibling environment (a different image,
a stripped init container, a clean shell on the same node) measures something other than
what the engine will sit on, and defeats the gate.

### 8.1 Scope

Pre-checks cover the hardware planes whose performance directly bounds LLM serving:

| Plane | Benchmark | Validates |
|---|---|---|
| GPU collectives | NCCL / RCCL all-reduce and all-gather, multiple message sizes | TP communication bandwidth across the deployed GPU group |
| GPU memory | `bandwidthTest` (or device-side STREAM) | HBM read/write bandwidth per GPU |
| System memory | STREAM | Host DRAM bandwidth (Grace DRAM on GH200; node RAM elsewhere) |
| Storage | Sequential read against the model-weights mount | Capstor / PVC throughput (validates `safetensors_load_strategy=prefetch` effectiveness) |
| Network (multi-node only) | Point-to-point `iperf3` between TP nodes | Inter-node link (IB / Ethernet) bandwidth |

Each pre-check must run on the **exact** topology *and* the **exact** software environment
the experiment will use: same node(s), same GPUs, same container image, same environment
variables, same mounts, same NUMA / CPU pinning, same library and NCCL/RCCL configuration,
same TP size. Running it anywhere else measures a different stack from the one the LLM
engine will actually sit on, and defeats the gate.

### 8.2 Execution

- Pre-checks run in the **same process environment** the LLM engine will run in — same
  image, same env, same mounts, same pinning. Concretely:
  - **SLURM**: invoked via `srun --environment=<engine-edf>` against the same EDF file the
    engine job uses, in the same allocation, so all `--container-mounts`, `--env`, and
    NUMA/CPU bindings are identical.
  - **Kubernetes**: invoked as a **pre-launch command in the engine's own pod and
    container** (e.g. `command: ["bash", "-c", "run_prechecks && exec <engine>"]`),
    *not* as a separate init container. The engine sidecar's env vars, volume mounts,
    `securityContext`, and resource requests are then guaranteed to be in effect during
    the checks.
- Total wall-clock budget for the full pre-check suite: ≤ 120 s (configurable via
  `system_prechecks_timeout_s`).
- Pre-check output is streamed to the structured log and persisted in the `system_prechecks`
  table (§13.6).

### 8.3 Reference values

Per-system reference values live in `tool/system_prechecks_reference.yaml`. The file is the
authoritative ground truth and is updated as new systems are characterised. Each entry carries:

- `expected` — measured median on a known-good run
- `tolerance_pct` — acceptable negative deviation (e.g. `-10` ⇒ warn if measured < 0.9 × expected)
- `source` — short note (date + `run_id`) on where the reference came from

Initial table (placeholder values — populated after first characterisation runs on each system):

| Cluster | Node | TP | Benchmark | Size | Expected | Tolerance |
|---|---|---|---|---|---|---|
| `clariden` | GH200 | 4 | NCCL all-reduce | 16 MiB | TBD GB/s | -10% |
| `clariden` | GH200 | 4 | NCCL all-reduce | 256 MiB | TBD GB/s | -10% |
| `clariden` | GH200 | 1 | HBM bandwidth | — | TBD GB/s | -5% |
| `clariden` | GH200 | 1 | Grace DRAM bandwidth | — | TBD GB/s | -10% |
| `clariden` | GH200 | 1 | Capstor sequential read | 1 MiB blocks | TBD GB/s | -15% |
| `beverin` | MI300A | 4 | RCCL all-reduce | 16 MiB | TBD GB/s | -10% |
| `beverin` | MI300A | 1 | HBM bandwidth | — | TBD GB/s | -5% |
| `breithorn` | gh200 | 4 | NCCL all-reduce | 16 MiB | TBD GB/s | -10% |

### 8.4 Outcome and abort flow

For each pre-check metric:

- **pass** — measured within tolerance band. Recorded; sweep proceeds.
- **warn** — measured below tolerance. A warning is emitted; the benchmarker pauses and
  surfaces the discrepancy to the coordinator on the laptop. The operator chooses to abort
  the experiment or proceed. Non-interactive runs default to the value of
  `prechecks_on_warn` in the benchmark YAML (`abort` or `continue`; default `abort`).
- **fail** — measured well below tolerance (e.g. < 50% of expected) or the benchmark binary
  itself errored. The experiment is aborted by default; override with `prechecks_on_fail:
  continue` in the benchmark YAML.

All measurements (pass, warn, fail) are persisted in `system_prechecks` (§13.6) so that
later analysis can correlate a degraded foundation with anomalous LLM benchmark results.

### 8.5 Skipping pre-checks

Pre-checks add up to ~120 s per experiment. They can be disabled via:

- CLI: `--skip-system-prechecks`
- YAML: `skip_system_prechecks: true`

Use sparingly — only when the exact same hardware + image + runtime environment + topology
combination was validated within the same session. Any change to the image, env vars,
mounts, NUMA pinning, NCCL/RCCL settings, driver, or node set invalidates a prior pass
and the checks should be re-run.

---

## 9. Inductor Pre-compilation Primer

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

## 10. Load Generation

### 10.1 Server readiness and model-loading tracking

Before the sweep starts, the load generator must, for **each** deployed instance:

- Wait for `/health 200`. Per-instance wait bounded by `server_ready_timeout_s` (default 3600 s;
  see §12.1). If any instance fails to come ready within the timeout, the experiment aborts.
- Parse the per-instance model-loading breakdown from the backend's structured logs / runtime
  API and persist one row per instance into the `instances` table (§13.2) with the
  `model_load_*` fields populated (§12.2).
- Run the inductor pre-compilation primer (§9).

The sweep begins only once **all** instances are ready, profiled, and primed.

### 10.2 Sweep structure

- **Warmup phase**: requests sent but metrics excluded. Long enough for:
  - Inductor JIT compilation to complete after primer (≥ 1 full round of compilation per model)
  - KV cache and queue to reach steady state
- **Measurement phase**: TTFT, ITL, E2E recorded per request.
- **Drain phase**: in-flight requests after measurement window are allowed to complete up to
  `drain_timeout_s`.
- `request_timeout_s`: client-side TTFT hard cutoff; exceeded requests recorded as `success=0`.

### 10.3 Open-loop stochastic arrivals

The load generator supports **configurable arrival processes**, selected per sweep step via
`arrival_process` in the benchmark YAML. The chosen process and its parameters are serialized
into `experiments.scenario_manifest.assumptions` (§13.8) so the conditions a result was
measured under are always recoverable.

| `arrival_process` | Description |
|---|---|
| `poisson` (default) | Memoryless Poisson at rate λ; inter-arrivals drawn from `Exp(1/λ)`. Statistically-independent baseline. |
| `burst_mmpp` | Two-state on/off Markov-Modulated Poisson Process at mean rate λ, with configurable burst factor (peak-to-mean ratio) and mean burst / idle durations. Models agentic and batch-API traffic patterns. |
| `pareto` | Heavy-tailed inter-arrivals at mean rate λ, configurable Pareto shape α. Models long-tail idle gaps interleaved with bursts. |

Each arriving request is routed to one of N server instances per `routing_strategy` (§10.4).

### 10.4 Routing strategies

- `random` (default): uniformly random instance selection per request.
- `session_affinity`: `prompt_idx % N` — same prompt always routes to the same instance.
  Enables meaningful prefix-cache benefit across multi-turn sessions. Useful to measure
  the effect of session affinity vs random routing in multi-instance deployments.

---

## 11. Measurement

### 11.1 Metrics recorded per request

- `ttft_ms`: time from send to first token (authoritative SLO metric)
- `tpot_ms`: inter-token latency (mean across output tokens)
- `e2e_ms`: total request time
- `input_tokens`, `output_tokens`, `success`, `error`

### 11.2 Request error tracking

Every failed request (`success=0`) is **kept** in the `requests` table (§13.3) — never dropped —
with its `error` column populated. The classification is used both for diagnosis and for
reporting error rates per λ level (see §14.1).

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

### 11.3 Server metrics (scraped periodically)

- `requests_running`, `requests_waiting`, `gpu_cache_pct`, `spec_accept_rate`

### 11.4 Agentic task metrics

Agentic and tool-calling scenarios fan a single user request out into many model invocations
(think → tool call → tool result → think …). In these scenarios the unit of measurement is
the **agent task**, not the individual model request: each task is recorded once in the
`agent_tasks` table (§13.7) and its child model invocations are recorded in the `requests`
table (§13.3), each carrying the parent's `agent_task_id`.

Per-task metrics are defined in the `agent_tasks` schema (§13.7) and include task-level
wall-clock latency, total invocations, aggregate input/output tokens, tool calls emitted
and validated against schema, and task success.

For schema-constrained decoding (JSON / XML), the load generator additionally records, per
request, whether the structured output validated against its schema, via
`requests.structured_output_valid` (§13.3) — `NULL` when the request is not under schema
constraint — so the cost of schema constraints can be analysed alongside the standard
latency metrics.

### 11.5 Hardware utilization sampling

While the LLM benchmark records request-level latency, the underlying hardware may be
under-utilised even when latency targets are met — meaning more concurrent traffic could be
served on the same allocation. To detect such headroom, the benchmarker samples host-side
hardware telemetry on every inference-server node for the full duration of every sweep step
(warmup + measurement + drain). Samples are stored in the `hardware_stats` table (§13.5).

Sampled signals — **GPU** (one row per GPU per sample):

| Signal | Meaning |
|---|---|
| `gpu_util_pct` | Coarse GPU activity (DCGM `DCGM_FI_DEV_GPU_UTIL`) |
| `gpu_mem_used_gb`, `gpu_mem_pct` | Device memory occupancy |
| `gpu_power_w`, `gpu_temp_c` | Power draw and thermal state |
| `gpu_sm_active_pct` | Fraction of cycles with ≥ 1 warp resident (DCGM `DCGM_FI_PROF_SM_ACTIVE`) |
| `gpu_tensor_active_pct` | Tensor-core pipe activity (DCGM `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`) |
| `gpu_dram_bw_gbs` | HBM read+write bandwidth |
| `nvlink_rx_gbs`, `nvlink_tx_gbs` | NVLink throughput per direction |
| `pcie_rx_gbs`, `pcie_tx_gbs` | PCIe throughput per direction |

Sampled signals — **System** (one row per node per sample):

| Signal | Meaning |
|---|---|
| `cpu_util_pct` | Aggregate CPU utilisation |
| `cpu_iowait_pct` | Fraction of CPU time blocked on I/O |
| `ram_used_gb`, `ram_pct` | Host memory occupancy |
| `ram_bw_gbs` | Host memory bandwidth (when perf counters available; else `NULL`) |
| `storage_read_gbs`, `storage_read_iops` | Read activity on the model-weights mount |
| `net_rx_gbs`, `net_tx_gbs` | Aggregate node network throughput |

Sampling cadence: **1 Hz** by default (configurable via `hardware_sampling_interval_s`).

Data sources by platform:

| Platform | GPU source | System source |
|---|---|---|
| NVIDIA (GH200) | DCGM in-container (recommended) or `nvidia-smi dmon` fallback | `psutil` + `/proc` |
| AMD (MI300A) | `rocm-smi` | `psutil` + `/proc` |

Signals a platform cannot expose are stored `NULL`. Reports (§14) overlay these signals
against λ so that the "p95 TTFT meets SLO but GPU SM-active is 35%" case (untapped headroom)
is immediately visible to the reader.

---

## 12. Benchmarker Infrastructure

### 12.1 Health-check timeout

The benchmarker health check must wait at least as long as `server_ready_timeout_s` (default 3600 s)
before giving up. The former 10-minute hardcoded limit was too short for servers with speculative
decoding (dual model load + CUDA graph capture ≥ 15 min).

### 12.2 Model loading time tracking

The Benchmarker records both the **total** time-to-ready and its **individual components**,
so that optimization decisions are driven by data — knowing whether to attack weight load,
graph capture, or compilation requires measuring each in isolation. Components that a given
backend cannot expose are stored as `NULL` rather than collapsed into another bucket.

A single experiment may deploy **multiple instances** of the same configuration (e.g. when
testing request routing across replicas). Each instance is measured independently, so the
load data is stored **per instance** in the dedicated `instances` table (§13.2).

All fields share the `model_load_` prefix to make the group easy to identify:

| Field | Measures |
|---|---|
| `model_load_total_s` | Total: from job start to first `/health 200`. |
| `model_load_weights_s` | Reading model weights from storage into device memory. |
| `model_load_engine_init_s` | Engine/runtime startup not attributable to the other components (tokenizer load, KV cache allocation, distributed init). |
| `model_load_cuda_graph_capture_s` | CUDA graph capture phase. |
| `model_load_inductor_compile_s` | `torch.inductor` JIT compilation primer (large-prefill path; add the draft-model path for speculative decoding). |

- Each component must be parsed from the backend's structured logs or runtime API (per
  backend; see backend-specific notes in §15).
- Reports show both the total and the per-component stack, per instance, with the totals
  aggregated across instances for the experiment summary (see §14).

---

## 13. Results

Per-run results live in a SQLite database file (`run_<id>.db`) with seven tables:
`experiments` (one row per sweep), `instances` (one row per deployed server instance),
`requests` (one row per issued request), `server_stats` (periodic samples of
server-side counters), `hardware_stats` (periodic samples of host hardware telemetry),
`system_prechecks` (one row per pre-check metric), and `agent_tasks` (one row per
agentic task in tool-calling scenarios; parent of multiple `requests` rows).

### 13.1 `experiments` table

One row per sweep — the configuration and overall outcome of the run.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, PK | Unique identifier (`timestamp + model_slug + backend + deployment + 4-hex random`; see §4.4). |
| `model` | TEXT | Model identifier (HuggingFace ID or path). |
| `backend` | TEXT | Inference engine (`vllm`, `sglang`, `dynamo`). |
| `backend_config` | TEXT (JSON) | Serialized `BackendConfig` — all fields from §5. |
| `dataset_config` | TEXT (JSON) | Serialized dataset configuration (§7). |
| `scenario` | TEXT | Scenario slug (e.g. `agentic-coding-large-prompt`, `chat-multimodal`, `reasoning-traces`). See §13.8. |
| `scenario_manifest` | TEXT (JSON) | Structured disclosure of what the scenario models, what it omits, and the numeric assumptions baked in. See §13.8. |
| `rate_levels` | TEXT (JSON) | List of λ values (req/s) swept in this run. |
| `warmup_s` | INTEGER | Warmup phase duration in seconds (metrics excluded; see §10.2). |
| `measurement_s` | INTEGER | Measurement phase duration in seconds. |
| `created_at` | TEXT (ISO 8601) | Experiment start timestamp. |

### 13.2 `instances` table

One row per deployed server instance for the experiment. A single experiment may deploy
multiple instances of the same configuration (routing tests, disaggregation studies, multi-
replica deployments); each instance has its own load profile (§12.2).

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Per-experiment instance identifier (stable across the run). Composite PK with `run_id`. |
| `endpoint` | TEXT | URL the load generator targets for this instance (`host:port`). |
| `node` | TEXT | Hosting node — SLURM node name or K8s pod / node-type. `NULL` if not applicable. |
| `model_load_total_s` | REAL | Total time-to-ready for this instance (§12.2). |
| `model_load_weights_s` | REAL | Weights load subcomponent (§12.2). |
| `model_load_engine_init_s` | REAL | Engine/runtime startup subcomponent (§12.2). |
| `model_load_cuda_graph_capture_s` | REAL | CUDA graph capture subcomponent (§12.2). |
| `model_load_inductor_compile_s` | REAL | Inductor compilation primer subcomponent (§12.2). |

Loading-time components a backend cannot expose are stored `NULL` (see §12.2).

### 13.3 `requests` table

One row per issued request — the per-request latency record.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `rate_lambda` | REAL | λ value (req/s) of the sweep step this request belongs to. |
| `request_id` | INTEGER | Per-rate-level request index (monotonic). |
| `ttft_ms` | REAL | Time to first token, milliseconds — authoritative SLO metric (§11.1). |
| `tpot_ms` | REAL | Inter-token latency, mean across the request's output tokens. |
| `e2e_ms` | REAL | End-to-end request time, milliseconds. |
| `input_tokens` | INTEGER | Number of input tokens. |
| `output_tokens` | INTEGER | Number of generated output tokens. |
| `success` | INTEGER | `1` if completed within timeouts; `0` if client-side `request_timeout_s` exceeded or the server returned an error. |
| `error` | TEXT | Error message or class when `success=0`; `NULL` otherwise. |
| `agent_task_id` | TEXT | Foreign key to `agent_tasks.agent_task_id` for requests issued as part of an agentic task; `NULL` for non-agentic scenarios. See §13.7. |
| `structured_output_valid` | INTEGER | `1` if the request used schema-constrained decoding and the output validated; `0` if the output failed schema validation; `NULL` if not under schema constraint. |

### 13.4 `server_stats` table

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

### 13.5 `hardware_stats` table

Periodic samples of host-side hardware telemetry on the inference-server node(s) during a
sweep step. See §11.5 for sampling cadence and per-signal meaning. GPU-scoped rows carry
a non-`NULL` `gpu_index`; node-scoped rows carry `gpu_index = NULL`.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Instance whose host this sample belongs to. |
| `rate_lambda` | REAL | λ value (req/s) of the sweep step being sampled. |
| `ts` | TEXT (ISO 8601) | Sample timestamp. |
| `gpu_index` | INTEGER | GPU device index for GPU rows; `NULL` for node-wide rows. |
| `gpu_util_pct` | REAL | Coarse GPU activity (§11.5). |
| `gpu_mem_used_gb`, `gpu_mem_pct` | REAL | Device memory occupancy. |
| `gpu_power_w`, `gpu_temp_c` | REAL | Power and thermal. |
| `gpu_sm_active_pct`, `gpu_tensor_active_pct` | REAL | DCGM profiling counters. |
| `gpu_dram_bw_gbs` | REAL | HBM bandwidth. |
| `nvlink_rx_gbs`, `nvlink_tx_gbs` | REAL | NVLink throughput. |
| `pcie_rx_gbs`, `pcie_tx_gbs` | REAL | PCIe throughput. |
| `cpu_util_pct`, `cpu_iowait_pct` | REAL | Node-level CPU. |
| `ram_used_gb`, `ram_pct`, `ram_bw_gbs` | REAL | Node-level RAM. |
| `storage_read_gbs`, `storage_read_iops` | REAL | Model-weights mount read activity. |
| `net_rx_gbs`, `net_tx_gbs` | REAL | Node network throughput. |

Signals a platform cannot expose are stored `NULL`.

### 13.6 `system_prechecks` table

One row per pre-check metric per instance (see §8). Used both for warning the operator about
a degraded foundation and for later correlation with anomalous sweep results.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Instance the check ran against. |
| `metric` | TEXT | Metric identifier (e.g. `nccl_allreduce_16MiB_GBs`). |
| `measured` | REAL | Measured value. |
| `expected` | REAL | Expected value from reference table (§8.3); `NULL` if no reference. |
| `tolerance_pct` | REAL | Negative deviation tolerance (e.g. `-10`); `NULL` if no reference. |
| `status` | TEXT | `pass`, `warn`, or `fail` (§8.4). |
| `ts` | TEXT (ISO 8601) | Time the check completed. |

### 13.7 `agent_tasks` table

One row per agentic task. A task is the unit of measurement for tool-calling scenarios:
a single user request that fans out into many model invocations (think → tool → result →
think …). Individual model invocations remain in the `requests` table (§13.3), each
carrying `agent_task_id` as a foreign key back to this table.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `agent_task_id` | TEXT, PK | UUID identifying the task; composite PK with `run_id`. |
| `rate_lambda` | REAL | λ value (req/s) of the sweep step the task started in. |
| `task_e2e_ms` | REAL | Wall-clock time from the user's request to the final response. |
| `task_invocations` | INTEGER | Number of model invocations the task fanned out to. |
| `task_input_tokens_total` | INTEGER | Sum of `input_tokens` across all child invocations. |
| `task_output_tokens_total` | INTEGER | Sum of `output_tokens` across all child invocations. |
| `task_tool_calls_emitted` | INTEGER | Tool calls emitted by the model across all invocations. |
| `task_tool_calls_valid` | INTEGER | Subset of `task_tool_calls_emitted` that parsed successfully against the declared schema. |
| `task_success` | INTEGER | `1` if the task completed successfully; `0` otherwise. |

For non-agentic scenarios this table is empty; all per-request data lives in §13.3.

### 13.8 Scenario manifest

Every result carries a structured **scenario manifest** that discloses what the benchmarked
scenario models, what it explicitly does *not* model, and the numeric assumptions baked in.
Without this, a reader looking at a plot has no principled way to know whether the result
applies to *their* workload — e.g. a Pareto frontier built from text-only large-prompt
agentic-coding traffic may be wildly off for a chat workload dominated by short turns and
image inputs.

The manifest is stored in `experiments.scenario_manifest` as a JSON object with the
following required keys:

| Field | Type | Semantic |
|---|---|---|
| `name` | string | Same value as `experiments.scenario`; included for self-containment. |
| `summary` | string | One- to two-sentence human description of the scenario. |
| `maturity` | string | One of `established` (validated against real-workload telemetry), `emerging` (early-signal, partially validated), `exploratory` (anticipated future pattern with no validation yet). Reports must visually flag `emerging` and `exploratory` Pareto frontiers as forward-looking so procurement readers can distinguish validated patterns from early signals. |
| `modelled` | list[string] | Aspects of real workload that the scenario *does* exercise — e.g. `"large multi-turn prompts (16K–32K input tokens)"`, `"follow-up turns reusing the initial context"`. |
| `not_modelled` | list[string] | Aspects the scenario explicitly does *not* cover — e.g. `"no image inputs"`, `"no audio inputs"`, `"no reasoning / thinking traces"`, `"no tool-call interleaving"`. |
| `assumptions` | list[string] | Numeric or structural assumptions baked in — e.g. `"follow-up turn probability = 0.4"`, `"max output tokens = 4096"`, `"input length distribution: lognormal, mean=20K, σ=0.3"`, `"system prompt length: 1.2K tokens, identical across sessions"`. |

Validation: a run aborts before submission if `scenario` or `scenario_manifest` is missing
or fails schema validation. There is no implicit default — every benchmark must declare
what it is and is not.

### 13.9 Experiment directories

Each completed sweep produces an `experiments/YYYY-MM-DD_description/` folder containing:

- `benchmark_config.yaml` (copy of the input config for provenance)
- the run's SQLite DB file (`run_<id>.db`)
- deployment artifacts used for the run (sbatch scripts, Kubernetes YAML, Dockerfile)
- the executed report notebook and its rendered outputs (see §14)

---

## 14. Reporting

The Reports generator produces a Jupyter notebook from the centralized results database
and writes it back into the experiment directory.

### 14.1 Report notebook (`experiments/template_report.ipynb`)

Every experiment report must include:

- Experiment title and description
- **Scenario & assumptions panel** (from `experiments.scenario` and
  `experiments.scenario_manifest`, §13.8): scenario name, one-line summary, the
  `modelled` list, the `not_modelled` list, and the `assumptions` list — surfaced
  near the top of the report, before any plot, so every downstream chart is read in
  the context of what the scenario actually does and does not cover. Items in
  `not_modelled` must be visually distinguished (e.g. struck-through or in a
  warning-coloured panel) so a reader cannot miss them.
- Configuration summary table (model, TP, KV dtype, spec dec, SLO, etc.)
- **System pre-checks** (from `system_prechecks`, §13.6): table of pre-check metrics with
  measured / expected / status — warns and fails flagged prominently at the top of the
  report so a degraded foundation is impossible to overlook when interpreting downstream
  numbers.
- **Model loading times**: per instance (from the `instances` table, §13.2),
  `model_load_total_s` plus the per-component breakdown (`model_load_weights_s`,
  `model_load_engine_init_s`, `model_load_cuda_graph_capture_s`,
  `model_load_inductor_compile_s`) — see §12.2.
- TTFT p50/p95 vs λ plot (log scale) with SLO line
- ITL p50/p95 vs λ plot
- Failure rate bar chart (bottom panel of each plot)
- **Hardware utilization** (from `hardware_stats`, §13.5), per λ level, overlaid against
  TTFT/ITL so untapped headroom is visible at a glance:
  - GPU SM-active and tensor-active vs λ (the key headroom indicator — SLO met with these
    well below 100% means the system can take more traffic on the same allocation)
  - GPU power, HBM bandwidth, and memory occupancy vs λ
  - NVLink and PCIe throughput vs λ
  - Node CPU, RAM, storage-read, network rx/tx vs λ
- Raw per-rate-level data table

### 14.2 Notebook output

The executed notebook (`report.ipynb`) and its rendered plots (`ttft.png`, `itl.png`,
`hardware.png`, `prechecks.png`) are written into the corresponding
`experiments/YYYY-MM-DD_description/` folder (§13.9).

---

## 15. Cluster-Specific Constraints (Alps / GH200)

### 15.1 vLLM flag compatibility (vllm-cxi v0.20.x)

| Flag | Status | Notes |
|---|---|---|
| `--speculative-model` / `--num-speculative-tokens` | ❌ Removed | Use `--speculative-config '{"model":..., "num_speculative_tokens":N, "draft_tensor_parallel_size":M}'` |
| `--swap-space` | ❌ Removed | Use `--kv-offloading-size` |
| `--kv-offloading-size` | ✅ Available since v0.20.0 | Total GiB across all TP ranks |
| `--kv-cache-dtype fp8` | ✅ Works | Native FP8 hardware on SM90 (GH200) |
| `--enable-prefix-caching` | ✅ Works | |
| `--safetensors-load-strategy` | ✅ Works | |
| `--speculative-config` | ✅ Works | JSON string |

### 15.2 Capstor filesystem

- capstor is **Lustre**, not Ceph. vLLM misidentifies it and disables auto-prefetch.
- Use `--safetensors-load-strategy=prefetch` to force Lustre-optimised parallel shard loading.
- Expected gain: ~10–20 s on a 131 GiB checkpoint.

### 15.3 GH200 KV cache capacity (70B, TP=4)

```
Available KV per GPU  = (96 GiB × gpu_memory_utilization) − (140 GiB / 4 GPUs)
                      ≈ 51 GiB at 0.90

KV per 25K-token request per GPU  ≈ 80 KB/tok × 25,000 = 1.91 GiB
Max concurrent requests            ≈ 51 / 1.91  ≈ 28

With --kv-offloading-size 400 (100 GiB/GPU via Grace DRAM):
Additional KV capacity  = 100 / 1.91 ≈ 52 additional slots
Total concurrent        ≈ 80 slots
```

### 15.4 Speculative decoding with shared GPUs

- Running draft at TP=4 on the **same** 4 GPUs as the 70B target is counterproductive:
  NCCL allreduce overhead for 5 draft passes per cycle (320 extra syncs) erases the
  memory bandwidth benefit.
- Optimal configuration when GPUs are shared: **draft TP=1**, accept reduced GPU-0 KV capacity.
- Optimal configuration when dedicated GPUs are available: **draft on separate node/GPUs**.

### 15.5 Inductor compilation timing

- **Target model (70B)**: first large-prefill request triggers ~60 s compile.
- **Draft model (8B)**: adds separate compile path, typically 30–60 s additional.
- Total primer wait on a cold server: up to 120 s. Primer timeout is set to 300 s.
- After compilation, TTFT drops to ~1.7 s for a 25K-token prompt at idle.
- Compilation artifacts are **not persisted** (`local_cache_dir: None`) — every restart re-compiles.

---

## 16. Known Issues & Workarounds

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

## 17. Open Items

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
- **Characterise pre-check reference values** for `tool/system_prechecks_reference.yaml` on
  each cluster (clariden GH200, beverin MI300A, breithorn gh200) — placeholders in §8.3
  must be replaced with measured medians plus tolerances before the foundation gate becomes
  enforceable.
