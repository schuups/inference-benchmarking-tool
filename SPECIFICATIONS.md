# Inference Benchmarking Tool - Specifications

This document enumerates all requirements captured so far.

## Table of Contents

1. [Guiding Principles](#1-guiding-principles)
2. [Architecture](#2-architecture)
3. [Project folder structure](#3-project-folder-structure)
4. [Pre-flight checks](#4-pre-flight-checks)
5. [Planner](#5-planner)
6. [Deployment Targets](#6-deployment-targets)
7. [Resources Lifecycle and Cleanup](#7-resources-lifecycle-and-cleanup)
8. [System Performance Pre-checks](#8-system-performance-pre-checks)
9. [Backends and models under test](#9-backends-and-models-under-test)
10. [Inference Engine Bring-up](#10-inference-engine-bring-up)
11. [Prompt Generation](#11-prompt-generation)
12. [Load Generation](#12-load-generation)
13. [Measurement](#13-measurement)
14. [Results](#14-results)
15. [Reporting](#15-reporting)
16. [Experiment Plans](#16-experiment-plans)
17. [Findings Records](#17-findings-records)
18. [Known Issues & Workarounds](#18-known-issues--workarounds)

---

## 1. Guiding Principles

- **Laptop-orchestrated**: the Coordinator runs on the operator's laptop — submits the
  Benchmarker job, monitors execution, collects per-run results into the centralised DB,
  tears down. Cluster-side sequencing (engine spawn, dataset gen → load gen) is the
  Benchmarker's job. The laptop carries orchestration and the results DB; the cluster
  carries the work.
- **Backend-agnostic**: vLLM, sglang, and NVIDIA Dynamo are first-class backends. Others
  can be added later by providing blueprint / example files.
- **Open-loop stochastic load generation**: sessions arrive at mean rate λ via Poisson or
  burst-aware processes (§12.3), independent of server completions (λ counts **session
  starts** — see §12.3 *What λ counts*). This captures the
  **queuing dimension** of real load — backlog, saturation, latency amplification. It is
  one of two axes of realism. The other — semantic realism: prompt content, multi-turn
  structure, fan-out, modality mix — lives in the scenario registry (§11) and is equally
  necessary.
- **Mixed workloads by default**: an experiment's traffic is a weighted blend of
  scenarios declared in `scenario_mix` (§11.4) — e.g. 80% agentic-coding + 20% chat —
  so cross-class interference (long agentic prefills inflating chat TTFT on a shared
  instance) is part of every measurement rather than an afterthought. Per-class SLOs
  (§13.4) and per-class report panels (§15.1) keep each class's experience separately
  visible. A single-scenario run is the degenerate mix with one entry.
- **Reproducible by config**: a single YAML file fully specifies the experiment and its
  parameter sweep. Re-running the same file must produce comparable results.
- **Scenario-disclosed results**: every plot is published with its scenario manifest,
  which describes the experimental context — including the assumptions made — so the
  reader can readily interpret the results (§14.7, §15.1).
- **Quality-disclosed capacity**: capacity gains from quality-impacting configurations
  (weight quantization, KV dtype, …) are published alongside the measured
  response-quality change **in the same report** — a faster deployment whose answers
  degraded is not an improvement. A pre-sweep sanity gate also protects every sweep
  from measuring a corrupted deployment (§13.5, §15.1).
- **Separation of concerns**: the **Benchmarker** runs as its own SLURM allocation,
  sequencing three phases — **dataset generation**, then spawning the **inference
  deployment** on a separate GPU allocation, then **load generation**. GPUs are not
  occupied until the dataset is ready (no idle during prompt prep), and the Benchmarker's
  separate allocation keeps its CPU work from competing with engine GPU inference.
- **Validated foundation**: every experiment is preceded by micro-benchmarks
  (NCCL / RCCL collectives, NVSHMEM, storage — §8) checked against per-system references.
  They run in the engine's container session — same libfabric / CUDA / NCCL / mounts /
  NUMA — so the measured foundation is the one the engine sits on. A degraded foundation
  pauses the sweep and offers the operator an abort.
- **Observed execution**: GPU, CPU, memory, storage, network, and power telemetry is
  sampled per inference-server node throughout every sweep (§13.3), so untapped headroom
  is distinguishable from saturation.
- **Clean cluster state**: all deployed resources must be cleaned up after every run — on both
  success and failure paths. No orphaned jobs, pods, services, secrets, or scratch directories.
- **Flexible process**: tool development and experiment design + execution follow a soft
  sequential process — the canonical phase sequence and the skip / rewind rules live in
  `CLAUDE.md` *How we work together*.

---

## 2. Architecture

The framework is **laptop-orchestrated, cluster-executed**. The operator's laptop carries
orchestration and the results database; a dedicated **Benchmarker** — always its own
**SLURM** allocation — spawns the inference deployment(s) under test, drives load against
them, and measures. The deployment under test runs on **SLURM or Kubernetes**, one engine
or many, a single GPU through many nodes; the Benchmarker spawns it with `sbatch` or
`kubectl` accordingly. The numbered sections that follow specify each piece in detail (the
component roster also lives in `CLAUDE.md`); this section gives the overall shape and the
engineering forces that drive it.

```
  LAPTOP — orchestration + results database (never allocated on a cluster)
  ═══════════════════════════════════════════════════════════════════════════════
     Pre-flight (§4) ──▶ Planner (§5) ──▶ Coordinator (§7) ──▶ Reports (§15)
     creds & capacity    renders the      submit · monitor ·   reads central DB
                         experiment dir   collect · teardown   ──▶ notebook + plots

     Cleaner (§7.7) — operator-approved sweep of whatever teardown missed
                             │
                             │  FirecREST MCP — the Coordinator always reaches the
                             ▼  Benchmarker over SLURM (decision 5)
  BENCHMARKER — one allocation per experiment · ALWAYS SLURM · runs from a staged venv
  ═══════════════════════════════════════════════════════════════════════════════
     ① dataset generation (CPU, §11)   — prompt pool built first, so the GPUs the
                                          engine will hold sit idle as briefly as possible
     ② spawn the engine under test ─────────────────┐   via  sbatch (SLURM target)
     ③ load generation + quality eval (§12, §13.5)   │    or  kubectl (K8s target)
            │ writes                                 ▼
     per-run SQLite DB (§14)             ENGINE DEPLOYMENT UNDER TEST — SLURM *or* K8s
            │                            ═══════════════════════════════════════════
            │ downloaded                 engine vLLM / SGLang / Dynamo × replicas
            ▼  (staged, compressed)      + ingress / routing (§10, §16.2)
     central results DB                  §8 pre-checks + hw sampler (§13.3) run in
            │                            the SAME container session as the engine
            ▼
     report notebook (§15)
```

### Constraints the architecture addresses

- **Measurement must not perturb the system under test.** The Benchmarker runs as its
  **own allocation**, separate from the inference deployment, so its CPU-bound dataset and
  load-generation work never competes with the engine's GPUs (§1 *Separation of concerns*).
- **GPU time is the scarce, costly resource.** The three Benchmarker phases are
  **sequential** — the CPU-bound prompt pool is built *before* any GPU deployment is
  spawned, so accelerators are never idle during prompt prep (§1, §11).
- **The deployment under test is highly variable.** Backend (vLLM / SGLang / Dynamo),
  topology (1 GPU → multi-node), replica count, and platform all change per experiment, so
  deployment artifacts are **rendered from templates** by the Planner (§5) and the
  Benchmarker spawns them as SLURM jobs or K8s manifests; multi-instance deployments map to
  `instances` rows (§14.2, §16.2).
- **The Benchmarker is always SLURM; only the engine target varies.** The Coordinator
  always reaches the Benchmarker over **SLURM / FirecREST** (decision 5); the
  SLURM-vs-Kubernetes choice is a property of the **engine deployment under test**, which
  the Benchmarker spawns with `sbatch` or `kubectl` from inside its allocation (§7, §10).
  One control path, two engine-deployment targets.
- **The measured foundation must be the one actually served.** The §8 collective / storage
  pre-checks run **in the engine's own container session** (identical libfabric / CUDA /
  NCCL / mounts / NUMA) and gate the sweep on a degraded foundation (§1, §8.2).
- **One image, both platforms.** Engine images are **self-contained** — the network stack
  is baked in and container-engine hooks are disabled — so the same image is correct on
  SLURM and K8s with no host-injection dependency (§9.1).
- **Runs are long; laptop links are flaky.** Coordinator state is **resumable**, the per-run
  DB moves over the **staged, compressed** transfer path (direct transfer is ~5 MB-capped),
  and the central-DB merge is **idempotent** and `run_id`-keyed, so a reattach never
  double-counts (§7, §14).
- **No orphaned cluster state, ever.** Teardown runs on **both success and failure paths**;
  a manual, operator-approved **Cleaner** reclaims anything that escaped (§1, §7.3–7.7).
- **Real load has queueing dynamics.** Load is **open-loop** — arrivals (Poisson /
  burst-aware) are independent of completions — so backlog, saturation, and tail
  amplification are observed rather than truncated away (§1, §12.3).
- **Evidence must compose across experiments.** Each run writes a self-contained per-run
  SQLite DB on the §14 schema; these fold into **one central results DB** from which reports
  are synthesised (§14, §15).
- **Reproducible and self-describing.** A single YAML fully specifies an experiment;
  datasets are **seeded** (§11.8) and engine versions / image provenance are pinned (§9.1);
  every plot ships with its scenario manifest (§1, §14.7).
- **Capacity claims are quality-disclosed.** A default-on pre-sweep **sanity gate** plus a
  post-sweep **quality comparison** against the deployed endpoint are reported alongside
  capacity, so a faster-but-degraded config cannot masquerade as a win (§1, §13.5).
- **Headroom must be distinguishable from saturation.** Per-node GPU / CPU / memory /
  network / power telemetry is sampled throughout each sweep by a stdlib-only sampler
  backgrounded in the engine container (§1, §13.3).
- **Operable by an assistant, yet fully scriptable.** Every step is also a plain CLI /
  Python entry point; the assistant drives the SLURM / FirecREST path in-session — there is
  no hidden autonomous daemon (§5, §7).

---

## 3. Project folder structure

### 3.1 Laptop (repository root)

| Folder | Purpose |
|---|---|
| `.venv/` | uv-managed Python 3.14 virtualenv. Activation and invocation details in *Constants* below. |
| `examples/` | Reference deployments and build scripts (Docker image builds, K8s and SLURM deployments, NCCL pre-checks). Claude consults these when building the benchmarking tool itself. |
| `firecrest-mcp/` | FirecREST MCP server. **Started manually by the operator** before a session — not auto-managed. |
| `tools/` | Implementation of the laptop-side components (Coordinator, Planner, Pre-flight checker, Cleaner, Reports generator). |
| `experiments/` | Per-experiment folders (`YYYY-MM-DD_description/`) with config, deployment artifacts, raw results. See §7.2 for the run-ID format and §14.8 for the directory contents. |
| `reports/` | Curated, audience-facing reports synthesised from one or many `experiments/` runs (§15.3). |

The repository root also carries `SPECIFICATIONS.md`, `CLAUDE.md`, `TODOs.md`, `README.md`, and `IMPLEMENTATION_PLAN.md` as the five authoritative documents.

### 3.2 Remote (cluster scratch)

On SLURM clusters, **all project files live under a single folder** — `/capstor/scratch/cscs/$USER/ibt/` (Lustre, HDD; see §6.1), the configured `scratch_base` (§3.3) — keeping the operator's scratch root uncluttered. Each experiment creates a run-specific subdirectory there holding the Benchmarker's working files (sbatch + EDF copies), the dataset generator's prompt pool, and the inference deployment's container working directory. Alongside the run dirs live the shared `collective-tests-cache/` (one entry per stack fingerprint, §8.2), `hf-cache/`, the host-side `venv/` (Benchmarker runtime), and `image-builds/` staging.

On Kubernetes (`breithorn`), the equivalent layout lives under Ceph-backed PVCs scoped per experiment.

Remote scratch is **transient** for a given run: per-run subdirectories are reclaimed by the cleanup phases in §7.

### 3.3 Global configuration (`tools/common/global.yaml`)

A single version-controlled YAML holding the **environmental constants shared by every
experiment** — values that are properties of the operating environment, not of any one
experiment:

| Key | Content |
|---|---|
| `clusters` | Cluster catalogue: type (`slurm` / `k8s`), FirecREST platform, partition / namespace, `gpus_per_node`. Drives per-cluster validation (e.g. the §6.1 TP ≤ `gpus_per_node` rule). |
| `slurm.account` | The only permitted account (§6.1). |
| `scratch_base` | Capstor scratch base (§3.2) under which run directories, prompt pools, and caches live. |
| `collective_tests_cache_dir` | Persistent compiled-binaries cache for §8.2. |
| `registry.jfrog_base` | JFrog publish base for built images (§9.1): `https://jfrog.svc.cscs.ch/artifactory/ml/inference`. |

What deliberately does **not** belong here: anything swept or experiment-specific —
model, BackendConfig, `scenario_mix`, SLOs, `quality_eval`, rate levels, image tags.
Those live in each experiment's benchmark YAML so a run stays fully reproducible from
that one file. The global values consumed by a run are copied into the experiment
directory (§14.8) for provenance.

Loaded and validated by `tools/common/config.py`; every component (Planner, Coordinator,
Cleaner, pre-flight checks) reads constants from it rather than duplicating literals.

---

## 4. Pre-flight checks

A laptop-side **Pre-flight checker** runs at the start of every new session to verify
that the target systems are ready to accept work. It catches operator-environment problems
(missing credentials, unreachable APIs, exhausted K8s capacity) early.

Distinct from the *System Performance Pre-checks* in §8: those run **inside the engine
container on the cluster** and validate hardware; the Pre-flight checks run **on the
laptop** and validate access to the targets.

Required checks (fail-fast; the operator sees a single error message naming the first
failing check):

| Plane | Check | Validates |
|---|---|---|
| Auth — ML Platform | FirecREST MCP "ML Platform" server responds | Credentials for `clariden` and `bristen` (both SLURM clusters under MLP — one credential set covers both, per §6.1) |
| Auth — HPC Platform | FirecREST MCP "HPC Platform" server responds | Credentials for `beverin` |
| Auth — K8s | `kubectl get nodes` against `breithorn` succeeds | kubeconfig present; Rancher cluster reachable |
| K8s capacity (`breithorn` only) | At least one node has all GPUs free | Schedulability. If free GPUs are fragmented (scattered across nodes with none aggregated per-node), the operator defrags via external K8s tools before retrying. |
| Filesystem | The capstor scratch dir (`/capstor/scratch/cscs/$USER/`) exists and is writable | The dataset generator can write the prompt pool to capstor scratch. |
| Podman storage config | `~/.config/containers/storage.conf` is present (or `$XDG_CONFIG_HOME/containers/storage.conf` if `XDG_CONFIG_HOME` is set) | Required for podman container operations on Alps (image build, EDF import). Contents per the [CSCS container docs](https://docs.cscs.ch/build-install/containers/). |
| Auth — JFrog | `jf rt ping` succeeds against the configured `cscs-jfrog` server | Credentials and a correct Artifactory URL for image push/pull (§9.1, §3.3). Catches the doubled-URL misconfiguration observed 2026-06-12. |

Implementation: `tools/pre-flight-checks.py`.

---

## 5. Planner

A laptop-side **Planner** prepares the artifacts the rest of the workflow consumes — it
takes the operator's intent and produces a self-contained experiment directory:

- The backend container EDF (SLURM `--environment=` TOML) or K8s deployment / service /
  ingress manifests, with §16.2 server-config knobs rendered into engine flags.
- The Benchmarker sbatch — **always SLURM** (the Benchmarker is never a K8s pod) — with the
  dataset prep → engine spawn → load gen chain wired up; the engine spawn it performs is
  `sbatch` or `kubectl` depending on the engine's target (§6, §8.2).
- The benchmark YAML's `dataset_config` block (§11.4), including the resolved scenario
  references (one per `scenario_mix` entry).

The Planner runs entirely on the laptop, against Jinja2 templates checked into the repo,
and does not touch the cluster — its output is config; nothing is submitted. The operator
drives the Planner through Claude Code (`/plan` or freeform conversation) or via a direct
CLI entry point; both paths produce the same artifacts. Planner output is then handed off
to the Coordinator at submission time.

The Planner does **not** persist any state of its own — every artifact it produces lives
in the experiment directory (§14.8), so a re-render is fully reproducible from the
benchmark YAML alone.

---

## 6. Deployment Targets

### 6.1 SLURM

Applies to all three SLURM clusters (`clariden`, `bristen`, `beverin`).

- All jobs (inference deployment, Benchmarker, image builds) submit to the cluster's
  default partition: **`normal`** on `clariden` and `bristen`, **`mi300`** on `beverin`.
  NCCL/RCCL benchmarks are **not** a separate job — they run inside the engine's container
  instance as part of the System Performance Pre-checks (§8).
- Account: `csstaff` (or `a-csstaff`). Never use other accounts.
- **Time-limit alignment**: every SLURM job in a single experiment — inference deployment
  and Benchmarker — must be configured with the **same** time limit, set conservatively
  enough to cover (chronologically): dataset generation + model load + CUDA graph capture +
  inductor compilation primer + quality gate (§13.5 Stage A) + full sweep + quality
  comparison (§13.5 Stage B) + results finalisation (writing the per-run DB and staging
  outputs into the experiment directory). Coordinator-driven cleanup runs
  *after* the SLURM job exits and is outside the time limit (§7).
- **Multi-node support via Ray.** Total GPUs = `tensor_parallel_size` × `pipeline_parallel_size`
  × `data_parallel_size` (× `expert_parallel_size` for MoE); node count = total GPUs /
  `gpus_per_node`. On Alps, **`tensor_parallel_size` must not exceed `gpus_per_node`** (= 4):
  TP's per-layer all-reduce is bandwidth-heavy, and the NVLink-C2C ↔ Slingshot 11 gap
  (~7–8×) makes cross-node TP impractical. Cross-node scale-out is via PP or DP, whose
  communication patterns tolerate the slower fabric.

Access is via FirecREST MCP. The MCP servers are scoped per **platform**, not per cluster —
clusters that share a platform share the same FirecREST credentials and MCP server:

| Cluster | Hardware | GPUs/node | Platform (FirecREST MCP server) |
|---|---|---|---|
| `clariden` | NVIDIA Grace-Hopper (GH200) | 4 | ML Platform |
| `bristen` | NVIDIA A100 | 4 | ML Platform |
| `beverin` | AMD MI300A | 4 | HPC Platform |

`bristen` runs A100 (sm_80) — vLLM kernels that assume Hopper (sm_90) features (e.g. FP8
data paths, certain FlashAttention-3 paths) do not apply there; treat any GH200-specific
guidance (§17.1, §17.2, §10.3) as not portable to `bristen`.

**Weight-storage mounts.** Both `capstor` and `iopsstor` are CSCS-managed parallel
**Lustre** filesystems exposed to SLURM jobs (Kubernetes uses Ceph-backed PVCs instead).
They differ in storage tier:

| Mount | Backend | Profile | Use for |
|---|---|---|---|
| `capstor` | Spinning-disk Lustre | High peak sequential bandwidth; slower random I/O and metadata | Bulk model weights, large sequential reads |
| `iopsstor` | All-flash Lustre | Lower peak sequential ceiling; fast random / small I/O / metadata | Workloads dominated by many small reads (heavy fine-grained checkpoints, KV-state spills) |

- vLLM misidentifies Lustre as Ceph on **both** mounts and disables auto-prefetch. Set
  `--safetensors-load-strategy=prefetch` to force Lustre-optimised parallel shard loading.
- Expected weight-load gain: ~10–20 s on a 131 GiB checkpoint on `capstor`. `iopsstor`
  delivers a similar or larger gain when shard files are small or many.
- Choose `iopsstor` over `capstor` when first-byte / metadata latency dominates startup.

### 6.2 Kubernetes (`breithorn`)

`breithorn` is the single Kubernetes target. It currently hosts one GPU node type:

| Node type | Hardware | Availability |
|---|---|---|
| `gh200` | NVIDIA Grace-Hopper | Available |

An `mi300a` (AMD MI300A) node type is **planned for the coming months** — once available
it will enable prefill-disaggregation studies (e.g. GH200 prefill paired with MI300A
decode, with KV transferred over Slingshot 11).

- `nodeSelector: beta.kubernetes.io/instance-type: <type>` targets a specific node type.
- A single experiment may pin different components to different node types (once
  mi300a is available, see above). The deployment manifest sets the `nodeSelector` per
  component.
- Time limit on K8s-deployed components must match the SLURM `server_time_limit` /
  `benchmarker_time_limit` of the same experiment (see §6.1).

**The Benchmarker stays on SLURM even for a K8s engine target.** Only the engine deployment
lives on `breithorn`; the Benchmarker is still a SLURM allocation that deploys the engine
via `kubectl` and drives load against it. This requires (a) the engine to expose an endpoint
reachable from the Benchmarker — a NodePort / LoadBalancer / Ingress, **not** the in-cluster
`*.svc` Service DNS — and (b) a `kubectl` context for `breithorn` usable from the SLURM job.
Both are E5 deliverables; the cross-platform endpoint / credentials mechanics are tracked in
`TODOs.md`.

---

## 7. Resources Lifecycle and Cleanup

Resources created during a run are labelled at creation (§7.1) and assigned a unique
run ID (§7.2). Per-run teardown (§7.3 → §7.6) runs on both success and failure paths at
end-of-run. Periodic operator-driven cleanup (§7.7) reclaims state that escaped per-run
teardown.

### 7.1 Resource identification

All benchmark-created resources must be labelled / tagged at creation so they can be
discovered and reclaimed later:

- **K8s**: every object (Deployment, Service, Ingress, PVC, Secret) carries
  `app.kubernetes.io/managed-by: inference-benchmarking`.
- **SLURM**: both inference-deployment and Benchmarker sbatch include
  `#SBATCH --comment=inference-benchmarking`.
- **JFrog**: container images pushed for benchmark runs are tagged under a benchmark
  path / prefix (e.g. tags prefixed `inference-benchmarking-<run_id>-`) so the JFrog
  API can list them by prefix.
- K8s discovery: `kubectl get all,ingress,secret -n ml -l app.kubernetes.io/managed-by=inference-benchmarking`.

### 7.2 Run ID uniqueness

- Run IDs include: timestamp + model slug + backend + deployment + 4-hex random suffix.
- The random suffix prevents collision when multiple Coordinators start within the same
  second.

### 7.3 Benchmarker teardown

- Cancel the Benchmarker SLURM job (`scancel <job_id>`).
- Delete the Benchmarker's capstor scratch run directory
  (`/capstor/scratch/cscs/$USER/ibt/<run_id>/`), which holds the dataset, working files,
  and load-gen state.

The Benchmarker spawned the inference deployment(s) but its cancellation does **not**
automatically cancel them — the per-target sections below handle that explicitly.

### 7.4 Inference deployment teardown — SLURM

- Cancel all inference-deployment SLURM jobs (`scancel`).

### 7.5 Inference deployment teardown — Kubernetes

- Delete: Deployment, Service, Ingress, TLS Secret (`<name>-cert`).

### 7.6 K8s PVC retention

Model-cache PVCs (`model-cache-<model-slug>`) are **intentionally kept** across runs to
avoid repeated 20–30 min weight downloads. The Coordinator's per-run teardown leaves
them in place.

### 7.7 Periodic cleanup (Cleaner)

State that escapes the Coordinator's per-run teardown — Coordinator killed mid-run,
network failure during teardown, older runs predating a teardown fix — is reclaimed by
a separate **Cleaner** that is **run manually by the operator** — never by Claude, never
by the Coordinator automatically.

Each Cleaner invocation has **two stages**:

1. **Identification** (always executed, read-only). Lists candidate resources discovered
   via the §7.1 labels / tags and a configurable age threshold (default 24 h). Output is
   a candidate report shown to the operator; nothing is modified.
2. **Cleanup / pruning execution** (requires **manual operator approval**). Once the
   operator confirms the candidate list, the script deletes the resources.

Implementation: `tools/cleaner.py` (script invoked from the laptop). Claude's role is to
**periodically remind the operator to run the cleanup** (e.g. via a scheduled wake-up
that surfaces the reminder); Claude does **not** execute the cleanup itself.

Resource classes the Cleaner identifies (and, on approval, prunes):

| Class | Discovery | Notes |
|---|---|---|
| K8s objects (Deployment, Service, Ingress, TLS Secret) | `kubectl get ... -l app.kubernetes.io/managed-by=inference-benchmarking` | Skip Model-cache PVCs (§7.6 keeps them intentionally). |
| SLURM scratch dirs under `/capstor/scratch/cscs/$USER/ibt/` | Match the run-ID pattern (§7.2); skip dirs owned by an active job. | |
| JFrog images tagged for benchmark runs | JFrog API filtered by the benchmark tag prefix (§7.1). | Skip the most recent N tags per repository. |

Cleaner actions are logged on the laptop but are not persisted to the per-run results DB.

---

## 8. System Performance Pre-checks

Synthetic micro-benchmarks executed in the engine's own container instance immediately
before the engine binary starts (mechanism in §8.2) — so the foundation measured is the
one the engine actually sits on. Without this gate, a degraded NCCL fabric or slow
weight storage would silently bias benchmark results: good throughput / latency numbers
measured on top of a degraded foundation are misleading.

### 8.1 Scope

Pre-checks cover the three planes whose performance directly bounds LLM serving:

| Plane | Benchmark | Validates |
|---|---|---|
| Collective communication | NCCL / RCCL `all_reduce`, `all_gather`, and `alltoall` (see `examples/nccl-tests/`), run at the engine's rank topology | Intra-node (NVLink / NVLink-C2C on GH200, Infinity Fabric on MI300A) and inter-node (Slingshot 11 on Alps) bandwidth in one pass. The three-collective set covers TP all-reduce, sequence-parallel / weight-gather, and MoE expert dispatch; the planner appends `sendrecv` for the inter-node PP point-to-point link when `pipeline_parallel_size > 1` (and `reduce_scatter` / `broadcast` can be added similarly). |
| GPU-initiated one-sided RMA (vendor SHMEM) | SHMEM perftest binaries shipped with the engine image — **NVSHMEM** on NVIDIA targets, **ROCm SHMEM** on AMD targets (e.g. `device/coll/alltoall_latency`, `device/pt-to-pt/shmem_put_bw`) — run at the engine's rank topology | Put / get bandwidth and one-sided all-to-all latency. Used by MoE engines that bypass NCCL / RCCL for expert dispatch (DeepEP and equivalents). Skipped with a warning if the engine image lacks the relevant SHMEM library; `shmem_required: true` to enforce. |
| Storage | Sequential read against the engine's model-weights mount (`capstor` / `iopsstor` on SLURM, Ceph PVC on K8s) | Read throughput as seen by the engine — **contextualises the model-loading times collected later** (§10.2), so a slow `model_load_weights_s` can be attributed to either storage or engine overhead. On Lustre also validates that `safetensors_load_strategy=prefetch` is taking effect (see §6.1). |

GPU memory bandwidth (`bandwidthTest`) and host DRAM bandwidth (STREAM) are intentionally
**not** in the pre-check suite — they are stable per-SKU characteristics that rarely
degrade in isolation without also degrading the NCCL / RCCL bandwidth above, so a separate
test adds maintenance without adding signal.

### 8.2 Execution

- Pre-checks run as a **dedicated step inside the engine's allocation**, immediately before —
  and gating — the engine step, using the **same EDF / container** as the engine (same image,
  libfabric / CUDA / NCCL / OpenMPI build, mounts, `--network`, `--mpi`). The dynamic linker
  resolves the identical stack, so the foundation measured is the one the engine sits on. The
  two steps differ only in **rank topology**, by design:
  - **Pre-check step (SLURM)**: `srun --environment=<engine-edf> --ntasks-per-node=<gpus_per_node>
    --mpi=pmix bash run_system_prechecks` — **one rank per GPU** (SPMD, PMIx-bootstrapped), the
    topology the collective / SHMEM micro-benchmarks need to measure the real per-GPU fabric
    (intra-node NVLink *and* inter-node Slingshot) and to wire up multi-PE NVSHMEM. A non-zero
    exit (warn/fail, §8.4) aborts the job before the engine starts.
  - **Engine step (SLURM)**: `srun --environment=<engine-edf> --ntasks-per-node=1 --mpi=pmix bash -c
    "<engine>"` — one task per node (vLLM drives its GPUs in-process, or via Ray for multi-node).
    The allocation is sized `--ntasks-per-node=<gpus_per_node>` so the pre-check step gets one task
    per GPU; the engine step simply uses fewer.
  - **Kubernetes**: an equivalent pre-check container/step runs before the engine container in
    the engine's own pod (same image, env, mounts, `securityContext`); K8s rank-topology details
    are finalised at E2c.

  This replaces the earlier welded `run_system_prechecks && exec <engine>` single-invocation model,
  which could only give the pre-checks the engine's **1-task-per-node** topology — adequate
  intra-node (one process drives all local GPUs over NVLink) but unable to measure the cross-node
  fabric or wire up multi-PE NVSHMEM (both collapsed to 1 rank/node). The dedicated step keeps the
  same *foundation* while measuring it at the engine's true per-GPU rank topology.
- The collective benchmark uses the upstream test suite for the engine's stack —
  [`NVIDIA/nccl-tests`](https://github.com/NVIDIA/nccl-tests) on NVIDIA targets (`clariden`,
  `bristen`, `breithorn`) and [`ROCm/rccl-tests`](https://github.com/ROCm/rccl-tests) on
  AMD targets (`beverin`). It is compiled inside the engine container at first use and
  cached on persistent storage. No dedicated test image. Because binaries built against
  one CUDA / NCCL (or ROCm / RCCL) / OpenMPI / libfabric combination are not ABI-portable
  to another, the cache directory is keyed by a **stack fingerprint** — a hash of the
  versions of CUDA or ROCm, NCCL or RCCL, OpenMPI, libfabric, and the test-suite tag
  detected in the engine container. Changing the engine image lands in a new cache
  directory; older caches survive in parallel and can be reused if the engine reverts.
  Configuration knobs flow from the benchmark YAML / sweep:
  - `collective_tests_version` — git tag to build (e.g. `nccl-tests` `2.17.1` or the
    matching `rccl-tests` tag). Pinned per experiment.
  - `collective_tests_cache_dir` — persistent path where compiled binaries live (default
    `/capstor/scratch/cscs/$USER/ibt/collective-tests-cache` on SLURM; a PVC mount on K8s).
    Shared across experiments; safe to delete to force a rebuild.

  The pre-check script attempts to **install missing build tools** (`make`, `g++`, OpenMPI
  dev, `curl`, `tar`) inside the engine container via `apt-get` / `dnf` / `yum`, rank 0 only
  (other ranks wait on a sentinel, so apt is not hammered by `N` ranks). **This fallback
  only works when the container runs as root.** On the CSCS Container Engine the container
  runs as the invoking user (non-root), so the install does not succeed and the build fails
  for want of the toolchain (a missing `mpi.h` / `g++` / NCCL headers — observed at E1 on the
  stock vLLM image). Because the dedicated pre-check step always runs **one rank per GPU**, the
  collective build is always the **MPI flavor** (`NCCL_TESTS_MPI=1`; each rank drives a single
  GPU via `-g 1`), which requires the image's MPI built against the Alps libfabric/CXI stack.
  The **engine image must therefore pre-ship the build toolchain** — the repo-built Alps image
  (§9.1) does; stock vendor images (§9.2) do not, so `skip_system_prechecks: true` is required
  when running on them (§8.5).
- The NVSHMEM benchmark uses the perftest binaries that ship with the engine image's
  NVSHMEM SDK (no separate build), bootstrapped over the dedicated step's SLURM PMIx ranks
  (`PMIX_MCA_psec=native`, `NVSHMEM_DISABLE_CUDA_VMM=1`; libfabric/cxi from the image's own
  Alps stack, host hooks disabled). `device/coll/alltoall_latency` (the MoE all-to-all
  collective) runs across all PEs of the step; `device/pt-to-pt/shmem_put_bw` needs **exactly
  two PEs**, so it runs in its own 2-rank step (its inter-node form gives the PP-link put
  bandwidth). If NVSHMEM is absent the row is skipped with a warning; set `shmem_required:
  true` in the benchmark YAML to make absence a failure.
- Total wall-clock budget for the full pre-check suite: ≤ 120 s (configurable via
  `system_prechecks_timeout_s`), achievable only when the collective-tests cache and the
  build toolchain are already in place. A cache miss — first-time collective-tests build
  or a missing build toolchain — flips the run into **smoke-test mode**: the pipeline
  still executes end-to-end (caches are warmed, the engine is exercised), but **results
  are not persisted**. A clear warning is surfaced to the operator at both launch and
  termination so the smoke-test status is unmissable. The operator re-runs the experiment
  afterwards.
- The `run_system_prechecks` script owns the full result lifecycle: running each
  benchmark, **collecting** its output, **parsing** it into per-metric values, comparing
  against `tools/system_prechecks_reference.yaml` (§8.3), and **storing** one row per
  metric in the `system_prechecks` table (§14.6). Streaming a verbose log alongside is
  fine; the table is the structured source of truth.

### 8.3 Reference values

The grading rule in §8.4 (pass / warn / fail) compares each measurement against an entry
in `tools/system_prechecks_reference.yaml` — **this is what makes pre-check results
actionable rather than informational**, and what flags an unacceptable deviation.

Each entry carries:

- `expected` — measured median on a known-good characterisation run (units inline).
- `tolerance_pct` — acceptable deviation. **Negative** for higher-is-better metrics
  (bandwidth, throughput): `-10` warns when measured falls below 0.9 × expected.
  **Positive** for lower-is-better metrics (latency): `+20` warns when measured rises
  above 1.2 × expected.
- `source` — date + `run_id` of the characterisation run that produced the reference.

Short example (full content lives in `tools/system_prechecks_reference.yaml`):

```yaml
- { cluster: clariden, scope: "4× GH200, 1 node",  benchmark: NCCL all_reduce,          size: 128 MiB, expected: TBD GB/s busbw, tolerance_pct: -10, source: TBD }
- { cluster: clariden, scope: "4× GH200, 1 node",  benchmark: NVSHMEM alltoall_latency, size: 128 KiB, expected: TBD µs,         tolerance_pct: +20, source: TBD }
- { cluster: beverin,  scope: "4× MI300A, 1 node", benchmark: RCCL all_reduce,          size: 128 MiB, expected: TBD GB/s busbw, tolerance_pct: -10, source: TBD }
- { cluster: beverin,  scope: "4× MI300A, 1 node", benchmark: ROCm SHMEM alltoall_latency, size: 128 KiB, expected: TBD µs,      tolerance_pct: +20, source: TBD }
```

All entries currently carry `expected: TBD` — placeholders until first characterisation
runs on each system. Until populated the gate (§8.4) is **unenforceable** and
measurements log as informational only.

### 8.4 Outcome and abort flow

For each pre-check metric:

- **pass** — measured within tolerance band. Recorded; sweep proceeds.
- **warn** — measured below tolerance. A warning is emitted; the benchmarker pauses and
  surfaces the discrepancy to the coordinator on the laptop. The operator chooses to abort
  the experiment or proceed. Non-interactive runs default to the value of
  `system_prechecks_on_warn` in the benchmark YAML (`abort` or `continue`; default `abort`).
- **fail** — measured well below tolerance (e.g. < 50% of expected) or the benchmark binary
  itself errored. The experiment is aborted by default; override with `system_prechecks_on_fail:
  continue` in the benchmark YAML.

All measurements (pass, warn, fail) are persisted in `system_prechecks` (§14.6) so that
later analysis can correlate a degraded foundation with anomalous LLM benchmark results.

### 8.5 Skipping pre-checks

Pre-checks add ~120 s per experiment. They can be disabled by setting
`skip_system_prechecks: true` in the benchmark YAML.

Use sparingly — only when the exact same hardware + image + runtime environment + topology
combination was validated within the same session. Any change to the image, env vars,
mounts, NUMA pinning, NCCL/RCCL settings, driver, or node set invalidates a prior pass
and the checks should be re-run.

---

## 9. Backends and models under test

This section enumerates the inference engines and the LLM models the benchmarker covers.
These are **stable system-under-test facts** — the operational reference for what the
framework can deploy and measure. The experiment-design surfaces (which feature to
compare, which BackendConfig knobs to sweep, which scenarios to use) live in §16.

### 9.1 Backends

The benchmark YAML's `backend:` field selects the inference engine. Each backend is
wired via its own Jinja2 EDF / K8s template (§10 *Inference Engine Bring-up*); its
sweepable configuration knobs live in §16.2.

| Backend | Planner template | Status |
|---|---|---|
| **vLLM** | `tools/templates/slurm/vllm.edf.j2` | active |
| **SGLang** | `tools/templates/slurm/sglang.edf.j2` (TBD) | planned |
| **Nvidia Dynamo** | `tools/templates/slurm/dynamo.edf.j2` (TBD) | planned |

(The SLURM engine EDF per backend; the SLURM job wrapper is `tools/templates/slurm/engine.sbatch.j2`
and the Kubernetes engine manifest is `tools/templates/k8s/engine.yaml.j2`. The Benchmarker job —
always SLURM — is the common `tools/templates/benchmarker.sbatch.j2`.)

The specific backend version and variant (e.g. one vLLM release vs a newer one, or
upstream vs a CSCS-patched build) is declared **per experiment** in the benchmark YAML and sweeps as a
deployment-sweep dimension — comparing versions of one backend, or comparing across
backends, are both first-class experiment shapes.

#### Image lifecycle and registry

Every inference-engine image the framework tests is **built from sources checked into
this repository**, organised as a small catalogue under `tools/images/`
(`tools/images/README.md` carries the operational detail):

- **Shared `core/`** holds the vendor-scoped, **version-tagged Alps network stack** at
  `core/<vendor>/netstack/<v>/` — the Containerfile, per-component build `phases/`,
  source `patches/`, and the baked `runtime/` env tuning — plus generic env/warning
  installers in `core/common/`. The netstack version (`v1`, `v2`, …) is a **maturity
  axis independent of the engine**: bumping a component pin, a Slingshot release, or a
  tuning choice (e.g. re-enabling NCCL LL128) lands as a new `vN` sibling so the prior
  one stays reproducible, and the same engine can be built on two netstacks to isolate
  the stack's contribution.
- **Each image is a thin per-image directory** — a `manifest.yaml` (identity = vendor ×
  backend × backend-version × netstack-version; base image; baked-in component pins;
  published tag; provenance; sanity status), an optional `variant/`, and `tests/`. The
  Containerfile lives once per netstack version; the manifest selects the base and any
  pin overrides.

**Self-contained images.** The network libraries (libfabric/CXI, patched NCCL +
aws-ofi-nccl, NVSHMEM, UCX/UCC/OpenMPI) **and** the runtime tuning env are baked into the
image, so it is correct under any launch — login shell, non-login `bash -c` on SLURM, or
a K8s pod — with **no container-engine hook and no `--environment` injection**. This is
essential for the K8s target, where there is nowhere to inject such adjustments. The
launch contract for a self-contained image is:

- EDF / pod annotation **`com.hooks.cxi.enabled = "false"`** and **no** `aws_ofi_nccl`
  hook annotation — the libraries are in the image, not injected by the engine. With the
  CXI hook disabled the devices remain accessible (`fi_info -p cxi` still enumerates the
  NICs), so disabling it costs nothing.
- Launch collectives with **`srun --network=disable_rdzv_get`**, matching the baked
  `FI_CXI_RDZV_PROTO=alt_read` (a runtime warning fires otherwise).
- The baked env (`NCCL_NET=AWS Libfabric`, `FI_PROVIDER=cxi`, `FI_CXI_*`, `OMPI_MCA_*`,
  `PMIX_MCA_psec`, `NVSHMEM_*`) applies via `/etc/profile.d` and `BASH_ENV` as *defaults*
  (set-if-unset), so any knob can be overridden per launch.

This **supersedes the older hook-injection model** in `examples/nccl-tests/README.md`
(which enables the `aws_ofi_nccl` hook and relies on host libraries); that guidance
applies only to stock images that ship without the Alps stack.

**Build, push, sanity.** Builds run under podman in node-local `/dev/shm` — a
RAM-resident layer cache, lost when the node is released — driven over **SSH +
`srun --overlap` into a held SLURM allocation** (FirecREST exposes no `srun`). The
per-phase Containerfile makes a failed or edited phase rebuild only from that layer down,
so a held node is reused across iterations. `tools/images/build.sh` reads a manifest,
stages the composed context, and runs `podman build` + `podman push`; one hard
constraint — the **built NCCL version must match the base image's torch-bundled NCCL**,
so the aws-ofi-nccl plugin is ABI-compatible with the NCCL actually loaded at runtime.

**Post-push acceptance gate + maturity.** Before an image is used, a short-lived 2-node
job (`tools/images/sanity.sbatch`) validates that the baked stack loads and **inter-node
collectives reach the Slingshot reference bandwidth** — NCCL `all_reduce` busbw near the
per-system reference (§8.3), *not* the ~5 GB/s "plugin didn't fire" floor — plus OSU and
NVSHMEM over CXI. Image status then distinguishes **`verified`** (sanity green —
functionally usable, performance scaling not yet characterised) from **`benchmarked`**
(additionally validated in inference performance-scaling experiments). To re-test a
rebuilt image under an unchanged tag, the sanity EDF is pinned by the **registry digest**
to bypass the engine's stale image cache (§18.1).

The image **registry** is the **CSCS JFrog Artifactory** (`registry.jfrog_base` in
`tools/common/global.yaml`, §3.3); images are referenced from the engine EDF / K8s
manifest by canonical tag or digest. Full build provenance — netstack source revision,
component pins, base image digest, published registry digest, build date — lives in each
image's `manifest.yaml` (durable; the `/capstor/.../ibt` build scratch is ephemeral) and
is recorded per experiment alongside the BackendConfig, so the exact stack any experiment
ran on is recoverable.

**Build-when-needed.** Most experiments deploy a pre-built image straight from JFrog.
When an experiment requires a change — a new pin, patch, base, or backend variant —
Claude carries the build through during experiment preparation: update the netstack /
manifest, build + push via `build.sh`, run the sanity gate, and update the planner
template's image reference to the new tag.

### 9.2 Models

The first-pass model set the benchmarker covers. Each entry pins the operational
information the Planner needs to render an engine launch (HuggingFace ID, tokenizer,
context length) plus the experiment-design information the operator needs to pick
scenarios and BackendConfig combinations (role, thinking mode, MoE structure).

Candidate future model families are tracked in `TODOs.md` (*Candidate models*) and enter
this table when they come under active measurement; this section is the **operational
subset**.

| Model | Role | HuggingFace ID | Tokenizer | Context | Thinking mode | MoE | Scenarios |
|---|---|---|---|---|---|---|---|
| **Apertus-70B** | target | `swiss-ai/Apertus-70B-Instruct-2509` | Apertus family (multilingual, 1000+ languages) | 65,536 tokens | No dedicated thinking mode (base model) | No | `long-context-followup`, `chat-short-turns` (with the caveat below on `thinking: true`). Excluded from `agentic-coding` — Apertus is not used by operators as a coding model. |
| **Apertus-8B**     | draft (same-family with Apertus-70B) | `swiss-ai/Apertus-8B-Instruct-2509` | Apertus family — **identical to the 70B**, tokenizer loaded once (§11.6) | 65,536 tokens | No | No | Always paired with Apertus-70B as the draft for speculative-decoding experiments |
| **Kimi-K2.6**      | target | `moonshotai/Kimi-K2.6` | Kimi family | 262,144 tokens (256K) | Yes — deeper reasoning and planning; strong on agentic, multi-step workflows | Yes (MoE; expert routing exercised by §16.1 *MoE expert routing* row) | `agentic-coding`, `chat-short-turns`, `long-context-followup`. `thinking: true` scenarios are most representative on Kimi-K2.6 because the widened output distribution matches the model's actual think+answer behaviour. |

**Notes on scenario / model pairing:**

- A scenario's `thinking: true` flag is a **workload declaration**, not a model-capability
  query. The dataset generator widens output sampling regardless of whether the model
  has a dedicated thinking mode (§11.6). When the model under test is **not** a thinking
  model (e.g. Apertus-70B), `thinking: true` simulates "what happens if this model
  were forced to emit thinking-length outputs" — a useful stress-test of decode capacity,
  but the class's `not_modelled` (§14.7) should disclose that the model's own emissions
  would ordinarily be shorter.
- Speculative-decoding experiments require a draft/target pair from this table. Only
  the **Apertus-8B → Apertus-70B** pairing is in scope for v1 (same-family, identical
  tokenizer). Kimi-K2.6 has no in-scope draft; cross-family pairings (e.g. a smaller
  open model as draft for an MoE target) are deferred until acceptance-rate baselines
  are characterised.
- Tokenizer-loading consequences for these pairings are documented in §11.6.
- Adding a new model to this set is a planner-template + benchmark-YAML change; no spec
  edit is required unless the model introduces a new capability dimension (e.g. native
  thinking mode toggle, new MoE topology) that the framework should sweep over.
- **Pipeline-validation smoke runs are exempt from these pairings.** The
  `smoke-synthetic` scenario (single-turn, synthetic source, `exploratory`) may be run
  against any model purely to validate the pipeline end-to-end (e.g. experiment E1 in
  `IMPLEMENTATION_PLAN.md`); its results are never published as findings or used for
  capacity / Pareto / procurement claims. The same exemption extends to the **engine
  image**: a pipeline-validation run may deploy a stock vendor image (e.g. NGC vLLM)
  while the repo-built lineage iterates; every graded run uses repo-built JFrog images
  per §9.1. Such stock-image runs must set `skip_system_prechecks: true` — stock images
  lack the build toolchain the §8 collective checks need (§8.2).

---

## 10. Inference Engine Bring-up

Once the System Performance Pre-checks (§8) pass, the Benchmarker hands control to the
inference engine in the same container session. This section covers the full bring-up:
how the launch command is constructed, how readiness is tracked, how model-loading time
is decomposed for diagnostics, and how the Inductor JIT-compile cost is paid up front
via a priming request — so that the sweep that follows measures steady-state behaviour.

The engine launch command is rendered by the Planner (§5) from:

- the **backend choice** in the benchmark YAML (`backend: vllm | sglang | dynamo`),
- the **BackendConfig** knobs (§16.2) — varied across deployments within an experiment (one deployment per combination; λ is then swept inside each deployment, see §12),
- backend-specific Jinja2 templates checked into `tools/` (one EDF template + one
  K8s manifest template per backend, per §1's *Backend-agnostic* principle).

The rendered command is concatenated with `run_system_prechecks && exec <engine>` (§8.2),
so pre-checks and the engine launch share the same container session with identical
libfabric / CUDA / NCCL / mounts / NUMA.

Backend-version compatibility of individual flags (which were removed, renamed, or made
mandatory in which release) is captured alongside the configuration surface in §16.2.

### 10.1 Health-check timeout

The benchmarker health check must wait at least as long as `server_ready_timeout_s`
(default 3600 s) before giving up — long enough to cover dual model load + CUDA graph
capture for speculative-decoding configurations, which can exceed 15 min.

### 10.2 Model loading time tracking

The Benchmarker records both the **total** time-to-ready and its **individual components**
(weight load, graph capture, compilation, …) so that optimisation efforts have a
per-component baseline to compare against and measure progress over. Components that a
given backend cannot expose are stored as `NULL` rather than collapsed into another bucket.

The schema lives in §14.2 (the `instances` table). Each `model_load_*` column is parsed
from the backend's structured logs or runtime API (per backend; for vLLM specifics see
§16.2).

A single experiment may deploy **multiple instances** of the same configuration (routing
tests, replica-sizing studies). The per-component breakdown is recorded **per instance**
so each instance's component times are visible individually. Reports (§15) show both the
per-instance breakdown and the totals aggregated across instances.

### 10.3 Inductor pre-compilation primer

vLLM v1 (v0.20+) uses `torch.inductor` to JIT-compile CUDA kernels for large prefill
sequences (> 512 tokens) **lazily** — on the first request that triggers the path. This
one-time compilation is the dominant cold-start cost on first request after a server start.

**Requirements:**

- The benchmarker must send a **priming request** (a large prompt — target ~20K tokens but
  **capped to `max_model_len`** so it never exceeds the served context — with `max_tokens=1`)
  to the engine's HTTP endpoint before the sweep begins, and wait up to 300 s for it to
  complete. An over-long primer prompt is rejected (http_400) and silently fails to warm the
  compile (observed at E1 with `max_model_len=16384`; follow-up tracked in `TODOs.md`).
- After the primer completes, the first measurement request should exhibit genuine
  steady-state TTFT (not the cold compile delay). If it does not, **warn the operator**
  that the primer missed its target.
- When the backend supports persisting compilation artifacts across restarts, the path is
  exposed as a `BackendConfig` field (§16.2); when it does not, the primer simply runs on
  every fresh start. Exploring this loading-time optimisation is tracked in `TODOs.md`.

---


## 11. Prompt Generation

### 11.1 Location, ownership, persistence

Prompts are produced on the **Benchmarker** SLURM allocation by its **dataset-generator**
subcomponent, from `dataset_config` in the benchmark YAML, sequentially before the
load-generator phase starts (per §1's separation of concerns). Generating on the cluster
sidesteps the FirecREST 5 MB direct-upload limit and allows arbitrarily large prompt
pools — the coordinator never ships prompt data to the cluster.

**Persistence.** Generated artefacts live on the Benchmarker's capstor scratch directory
for the duration of the run, reused across every deployment and every rate-level sweep
step, and reclaimed by the §7 teardown. They are **not** copied into `experiments/<run>/`:
the manifest (persisted on `experiments.scenario_manifest`), the master seed (in
`dataset_config`), and the scenario-registry revision (recorded alongside) are sufficient
to regenerate. The full reproducibility contract lives in §11.8.

**Source-failure semantics.** A failure of any dataset source (LongBench download error,
HuggingFace unreachable, reasoning-trace dataset unavailable, etc.) **aborts the run**
with a clear error. There is no silent fallback to synthetic data — this avoids the trap
where, e.g., a speculative-decoding experiment silently degrades to filler text and
reports ~0% acceptance as if it were a property of the model.

### 11.2 Key concepts

Three artefacts cooperate to produce a benchmark dataset:

- **Scenario registry** (`tools/scenarios/<slug>.yaml`) — the canonical declaration of
  what each scenario is: its **source** (§11.5), **length distributions** (§11.6),
  **multi-turn structure** and **session mode** (§11.7), and the **`modelled` /
  `not_modelled`** lists — human-authored statements describing what the scenario
  explicitly represents and what it deliberately does not, so any reader of a report
  knows in what context the numbers should be interpreted; copied verbatim into the
  scenario manifest (§11.8, §14.7).
- **`dataset_config`** in the benchmark YAML — the per-run knobs: the `scenario_mix`
  (workload classes and their weights), the master seed, `num_prompts`, and any
  per-class overrides to registry defaults.
- **Dataset generator** — reads both, materializes the prompt pool on capstor scratch,
  and emits the scenario manifest as a structured side-effect for the experiment row
  (§14.1).

The registry is data, not code: adding a new scenario does not require editing the
generator. Registry entries are versioned with the repo; changes to a scenario's
`modelled` / `not_modelled` lists are reviewable in PRs.

### 11.3 Scenario registry

Each registry entry is a YAML file with the schema below. Fields not relevant to a given
scenario are omitted (e.g. `session.think_time_ms` does not apply to single-turn
scenarios).

| Field | Notes |
|---|---|
| `name` | Slug, matches the filename. Used as the class slug in `experiments.scenario_mix`, `requests.scenario`, and the manifest (§14.7). |
| `summary` | One-line human description. |
| `maturity` | `established` \| `emerging` \| `exploratory`. |
| `source` | `kind` (§11.5) + per-source `config`. |
| `input_length` | `distribution` (`lognormal` \| `normal` \| `fixed`) + `params` (§11.6). |
| `output_length` | Same shape as `input_length`. Widened when `thinking: true` (§11.6). |
| `thinking` | Optional boolean; widens output sampling per §11.6. Default `false`. |
| `session.mode` | `open_loop` \| `sequential` (§11.7). |
| `session.turns_per_session` | Distribution (same shapes as `input_length`). |
| `session.followup_input_length` | Optional distribution for the input length of follow-up turns (`turn_idx ≥ 1`); defaults to `input_length`. Lets returning-user scenarios pair a large turn-0 context with short follow-ups (§11.6). |
| `session.prefix_strategy` | `append_delta` (only supported value; §11.7). |
| `session.think_time_ms` | Distribution; paces follow-up turns in **both** session modes — anchored at the previous turn's send time (`open_loop`) or response time (`sequential`); see §11.7. Required for multi-turn scenarios. |
| `manifest.modelled` | Human-authored list of what the scenario explicitly represents. |
| `manifest.not_modelled` | Human-authored list of what the scenario deliberately omits. |

See `tools/scenarios/agentic-coding.yaml` for a worked example. `assumptions` is **not**
stored in the registry — it is computed at runtime from the actual `dataset_config`
consumed (§11.8).

Registry entries are always **single-scenario**. Workload blending happens exclusively
in `dataset_config.scenario_mix` (§11.4): a mix references N registry entries; the
registry itself never encodes a mix.

### 11.4 dataset_config schema

The benchmark YAML's `dataset_config` block:

| Field | Type | Required | Notes |
|---|---|---|---|
| `scenario_mix` | list | yes | One entry per workload class. A single-scenario run is the degenerate mix with one entry at `weight: 1.0`. Weights must sum to 1.0; the Coordinator aborts otherwise (§14.7). |
| `scenario_mix[].scenario` | string | yes | Must match a registered scenario name (§11.3). |
| `scenario_mix[].weight` | float | yes | Fraction of **session starts** assigned to this class (§12.3). The per-request share then follows from each class's `turns_per_session` and is disclosed in the manifest (§11.8). |
| `scenario_mix[].input_length` | object | no | Per-class override of the registry's `input_length` distribution. |
| `scenario_mix[].output_length` | object | no | Per-class override of the registry's `output_length` distribution. |
| `scenario_mix[].session` | object | no | Per-class override of session fields. |
| `scenario_mix[].source_overrides` | object | no | Per-class source-specific overrides (e.g. LongBench task subset). |
| `num_prompts` | integer | yes | Total size of the generated prompt pool, split across classes proportionally to `weight × E[turns_per_session]` so no class exhausts its sub-pool early. Prompts are never recycled (the load generator raises `PoolExhaustedError`), so the pool must outlast the **whole sweep's** session-start budget — rule of thumb: `num_prompts ≥ Σ over sweep steps of λ × (warmup_s + measurement_s) × E[turns_per_session]` (summed over the mix). Undersizing aborts the run mid-sweep (observed at E1). This also keeps prompt uniqueness (§11.6) holding for every request actually issued. |
| `seed` | integer | yes | Master seed; per-class sub-seeds derived deterministically (§11.8). |
| `tokenizer_id` | string | no | Override the tokenizer (defaults to the target model's; see §11.6). Shared by every class in the mix — all lengths are measured with one tokenizer. |
| `output_length_mode` | string | no | `forced` (default) or `natural` — governs `ignore_eos` on sweep traffic; see §11.6. Disclosed in `run_assumptions`; results are not comparable across modes. |

Example — the canonical mixed workload (80% agentic coding, 20% chat):

```yaml
dataset_config:
  scenario_mix:
    - scenario: agentic-coding
      weight: 0.8
    - scenario: chat-short-turns
      weight: 0.2
  num_prompts: 20000
  seed: 1234
```

Any field absent from a mix entry is inherited from that class's scenario registry entry.

### 11.5 Dataset sources

The `source.kind` enum, with v1 scope:

| `kind` | What it is | `source.config` |
|---|---|---|
| `synthetic` | Filler text with unique `[prompt-NNNNNN]` headers. No network required. | — |
| `longbench` | LongBench code tasks downloaded from HuggingFace. | `tasks: [lcc, repobench-p, …]` |
| `reasoning_trace_replay` | Recorded reasoning traces (GSM8K-with-cot, MATH, AIME, R1-distill). Output length comes from the recorded target and **overrides** `output_length`. | `dataset: <name>` |
| `wildchat` | Real user↔assistant conversations from `allenai/WildChat-1M`. Multi-turn (median ~3, long tail), multilingual. Conversation turn boundaries drive the session structure; per-turn lengths are clamped to the scenario's distributions. | `languages: [en, …]`, `min_turns: N` |

Per-source suitability for the optimisations being measured (speculative-decoding
acceptance, prefix-cache hit-rates) is summarised in §11.9. Licenses: LongBench (MIT),
WildChat (ODC-BY), reasoning-trace datasets (per-dataset, all permissive for
benchmarking).

**v1 is text-only.** Multimodality (image first, then audio and video) is the next
feature on the dataset-generator roadmap — see `TODOs.md`. A scenario whose registry
entry declares any `modalities` other than `[text]` is rejected at registry-load time
until support lands.

### 11.6 Per-request mechanics

**Prompt uniqueness.** Every prompt must start with a distinct token block so the
engine's prefix cache does not serve synthetic cache hits — which would collapse TTFT to
~100 ms regardless of load, an artefact rather than real performance. Single-turn prompts
begin with a unique `[prompt-NNNNNN]` header; multi-turn sessions begin with a unique
`[session-NNNNNN]` header reused across the session's turns, so the prefix cache *does*
hit on the shared session prefix — the locality the benchmark is meant to expose (§11.7).
Header counters are allocated globally across all classes of the mix, so uniqueness
holds pool-wide, not merely within one class.

**Length distributions.** `input_length` shape is per-scenario, declared in the registry
(§11.3). Supported: `lognormal` (truncated), `normal` (truncated), `fixed`. Heavy-tailed
`lognormal` matches observed LLM-workload distributions and is the recommended default;
`fixed` is for isolation studies. `input_length` governs turn 0 (and single-turn
prompts); follow-up turns sample from `session.followup_input_length` when the scenario
declares it, falling back to `input_length` otherwise — so returning-user scenarios pair
a heavy initial context with short follow-ups instead of repeating turn-0-sized prompts.

**Output length control.** Each prompt carries a target `max_tokens` sampled from
`output_length`. Behaviour is governed by `output_length_mode` (§11.4):

- **`forced` (default)**: the load generator sends `max_tokens=<sampled>` **and**
  `ignore_eos=True`, forcing the model to emit exactly that many decode tokens. Decode
  cost becomes reproducible across runs, configs, and models — measured TPOT and
  `output_tokens` no longer depend on per-model stopping behaviour. The price: responses
  are truncated / padded arbitrarily, so **sweep traffic is ungradeable for quality by
  construction** — response quality is measured by the separate evaluation phase (§13.5).
- **`natural`**: `ignore_eos=False`; the sampled value acts as a cap only. Output
  lengths — and therefore decode cost — become model- and config-dependent. Use for
  token-efficiency studies and stopping-behaviour analysis. **λ\*, latency, and
  throughput results are not comparable across modes** — reports must refuse to overlay
  them — and the active mode is disclosed in the manifest's `run_assumptions` (§14.7).

Sources that carry ground-truth output lengths (`reasoning_trace_replay`) override the
sampled value with the recorded target.

**Thinking models (v1 approximation).** When a scenario sets top-level `thinking: true`,
the generator widens `output_length` sampling to approximate the combined
think-trace-plus-answer length: `params.mean × 2.5`, `params.sigma` (or `stdev`) `× 1.5`;
`fixed` values multiplied by 2.5. `params.min` / `params.max` are preserved as clamps
(not rescaled). A precise bimodal sampler (tiny direct answers vs long deep-thinking
outputs) is deferred — see TODOs.md *Bimodal output distribution as first-class*. The
flag's effect is recorded in the class's `modelled` list and the simplification
disclosed in its `not_modelled` (§14.7).

**Tokenization.** Length filtering, length-distribution sampling, and the `input_tokens`
field all use the **target model's tokenizer**, loaded by HuggingFace ID on the
Benchmarker at dataset-generation time. Changing the target model invalidates the dataset
and triggers regeneration. For same-family draft/target pairs (e.g. Apertus-8B +
Apertus-70B — the in-scope pairing for v1, see §9.2) tokenizers are identical and
only one is loaded. For cross-family pairs the **target's** tokenizer is authoritative;
any draft-tokenizer mismatch is logged but does not block the run.

### 11.7 Sessions and agentic approximation

**Multi-turn structure.** Multi-turn scenarios produce N turns per session, with N
sampled from `session.turns_per_session` (§11.3). Each session is assigned a stable
integer `session_idx ∈ [0, M-1]`; every turn carries the same `session_idx` (exposed to
the load generator for session-affinity routing — §12.4), and the `[session-NNNNNN]`
header from §11.6 encodes the same identifier in the prompt text. Single-turn scenarios
are the degenerate case M = `num_prompts`, `session_idx = prompt_idx`.

**Prefix strategy** is always `append_delta`: turn K+1's prompt = full prior transcript +
new user turn. The engine's prefix cache reuses the shared prefix naturally — exactly as
real chat / agentic clients do. (A `regenerate` strategy was considered but rejected: it
defeats the prefix cache and is better expressed as a separate ablation by disabling
prefix caching at the backend.)

**Class membership.** In a mixed run (§11.4) every session belongs to exactly one
workload class — its `scenario_mix` entry, assigned at session start (§12.3). The class
determines the session's source, length distributions, turn structure, and session mode;
every request of the session carries the class slug in `requests.scenario` (§14.3),
which is the key report-time per-class group-bys operate on (§13.2, §13.4, §15.1).

**Session mode** governs how follow-up turns interact with the load generator's open-loop
arrival process (§12.3):

| `session.mode` | Follow-up behaviour | Use for |
|---|---|---|
| `open_loop` (default) | Turn K+1 is sent at turn K's **send time** plus a `think_time_ms` sample — independent of when (or whether) turn K completed. Preserves open-loop queueing semantics within the session: a saturated server does not slow the turn stream. | RAG-style queries against a shared long-lived prefix; reasoning workloads; any scenario where turn ordering is incidental. |
| `sequential` | Turn K+1 is sent at turn K's **response time** plus a `think_time_ms` sample. Closed-loop coupling *within a session*; cross-session arrivals remain open-loop (session starts keep arriving per §12.3 regardless of server state, so backlog growth is preserved). | Conversational chat; agentic-coding follow-ups; any scenario where a follow-up cannot meaningfully precede its predecessor's response. |

The two modes are exact mirrors — both pace follow-ups with `think_time_ms`, anchored at
the previous turn's send time (`open_loop`) or response time (`sequential`).

**Agentic approximation (v1).** Agentic workloads — a user prompt fanning out to many
model invocations (think → tool call → tool result → …) — are approximated as **multi-turn
sessions with bursty fan-out**: each session = one agentic task; each turn = one model
invocation; tool results synthesised as injected text in the next turn's prompt. No tool
catalog, no fan-out DSL, no per-tool JSON schemas. This is enough to derive supportable
agentic-user count from the SLO-attained rate λ* (§13.4) via the report notebook's
supportable-users estimate (§15.1). The precise mechanism (distinct `think` / `tool_call`
/ `tool_result` roles, per-tool schemas, schema-constrained-decoding validity, a
dedicated `agent_tasks` table, first-class bimodal output, per-tool result-content
synthesis) is deferred to TODOs.md *Precise agentic / tool-calling measurement*. Routing:
`session_affinity` (§12.4) is the natural choice — every turn of a task lands on the same
instance so the prefix cache exposes the locality the workload depends on.

### 11.8 Reproducibility surface

**Seeding.** `dataset_config.seed` is a single integer. Per-class, per-axis sub-seeds
derived as `blake2b(f"{seed}:{scenario}:{axis}", digest_size=8)` over axes: `header`,
`length_input`, `length_output`, `selection`, `turns`, `thinktime`. One run-level axis,
`mix` (`blake2b(f"{seed}:mix", digest_size=8)`), seeds the categorical assignment of
session starts to classes (§12.3).

**Contract.** Same `dataset_config` + same scenario-registry revision + same target
tokenizer → identical prompt pool (byte-for-byte). Changing the target model triggers
regeneration (different tokenizer → different length filtering and different sampled
lengths).

**Manifest.** The dataset generator emits the `scenario_manifest` (schema in §14.7) as a
side-effect of running:

| Manifest field(s) | Source |
|---|---|
| `mix` | The `scenario_mix` actually consumed: `[{scenario, weight}, …]`, plus the resulting expected per-request share per class (derived from `weight × E[turns_per_session]`). |
| Per-class `name`, `summary`, `maturity`, `modelled`, `not_modelled` | Copied verbatim from each class's scenario registry entry. |
| Per-class `assumptions` | Auto-filled from the per-class config actually consumed: input / output (and follow-up, when distinct) length distributions; turns-per-session distribution; session mode; prefix strategy; source `kind` + relevant source config. |
| `run_assumptions` | Auto-filled run-level facts: arrival process + parameters (§12.3); routing strategy (§12.4); `output_length_mode` (§11.6); master seed; tokenizer ID. |

Together these are sufficient to reconstruct what the run measured without re-reading the
registry at a specific revision.

### 11.9 Notes on dataset suitability

- **Synthetic prompts**: acceptable for latency and throughput benchmarking but produce
  near-zero speculative-decoding acceptance rates (random text is unpredictable).
- **LongBench / real code**: required for meaningful speculative-decoding acceptance.
  Same-family draft/target pairings (e.g. Apertus-8B as draft for Apertus-70B)
  typically yield 0.5–0.7 acceptance on real code.
- **Reasoning-trace replay**: required when measuring speculative-acceptance for
  reasoning workloads — synthetic prompts produce the same near-zero rate as for code.
- **WildChat (real chat)**: required for meaningful speculative-decoding acceptance and
  realistic prefix-cache hit-rates on chat workloads — synthetic chat would otherwise
  collapse both signals. Real conversational structure (system-prompt prefixes,
  recurring opening phrases, language priors, follow-up patterns) is what the prefix
  cache actually has to exploit in production.

---


## 12. Load Generation

### 12.1 Server readiness and model-loading tracking

Before the sweep starts, the load generator must, for **each** deployed instance:

- Wait for `/health 200`. Per-instance wait bounded by `server_ready_timeout_s` (default 3600 s;
  see §10.1). If any instance fails to come ready within the timeout, the experiment aborts.
- Parse the per-instance model-loading breakdown from the backend's structured logs / runtime
  API and persist one row per instance into the `instances` table (§14.2) with the
  `model_load_*` fields populated (§10.2).
- Run the inductor pre-compilation primer (§10.3).

The sweep begins only once **all** instances are ready, profiled, primed, and — unless
`skip_quality_gate: true` — quality-gated (§13.5 Stage A).

### 12.2 Sweep structure

- **Warmup phase**: requests sent but metrics excluded. Long enough for:
  - Inductor JIT compilation to complete after primer (≥ 1 full round of compilation per model)
  - KV cache and queue to reach steady state
  - The **session population** to reach steady state: at step start only first turns
    arrive, and the request rate ramps from λ toward λ × E[turns_per_session] (§12.3
    *What λ counts*) as sessions accumulate — warmup must cover at least a mean session
    wall-time (turns × (latency + think time)) so the measured window sees the full
    follow-up load.
- **Measurement phase**: TTFT, ITL, E2E recorded per request.
- **Drain phase**: **no new session starts** after measurement ends; in-flight sessions
  and requests are allowed to complete up to `drain_timeout_s`. Sessions still
  unfinished at drain end are truncated: their issued requests are kept in `requests`,
  but the session is **incomplete** for session-level metrics (§13.2). Sessions never
  span sweep steps — every request inherits the `rate_lambda` of the step its session
  started in.
- `request_timeout_s`: client-side TTFT hard cutoff; exceeded requests recorded as `success=0`.
- After the final rate level's drain, the **quality comparison** (§13.5 Stage B) runs
  against the still-running deployment(s), before teardown.

### 12.3 Open-loop stochastic arrivals

The load generator supports **configurable arrival processes**, selected per sweep step via
`arrival_process` in the benchmark YAML. The chosen process and its parameters are serialized
into `experiments.scenario_manifest.run_assumptions` (§14.7) so the conditions a result
was measured under are always recoverable.

| `arrival_process` | Description and intuition |
|---|---|
| `poisson` (default) | **Mathematical**: memoryless Poisson at rate λ; inter-arrivals drawn from `Exp(1/λ)`. **Intuition**: a sterile lab baseline where every request is independent of every other — picture users hitting "send" at random moments with no clustering, no batching, no feedback between them. Useful as the cleanest characterisation of pure queue behaviour, but produces optimistic tail latencies because real traffic always clusters more than this. |
| `burst_mmpp` | **Mathematical**: two-state on/off Markov-Modulated Poisson Process at mean rate λ, with configurable burst factor (peak-to-mean ratio) and mean burst / idle durations. **Intuition**: traffic alternates between a high-arrival burst phase and a quiet phase, modelling production patterns where a tool-calling cycle fans out several requests at once, batch-API submissions flush queues at intervals, or cohorts of users arrive in waves. Reveals tail-latency amplification and queueing collapse that Poisson hides because real traffic's coefficient of variation is well above Poisson's 1. |

**What λ counts.** λ is the **session-start rate** (sessions/s). For single-turn
scenarios a session is one request, so λ reduces to the familiar request rate. For
multi-turn classes the arrival process schedules **session starts**; the session's
follow-up turns are then paced by the class's session mode and `think_time_ms` (§11.7)
— `open_loop` turns anchored at the previous turn's send time, `sequential` turns at
its response time. Follow-up arrivals are therefore deterministic offspring of the
session-start process, not a separate stream: the steady-state request rate observed by
the server is **λ × Σ weight × E[turns_per_session]**, which the dataset generator
derives per class and disclosed in the manifest's `run_assumptions` (§14.7). This gives
one coherent λ definition across a mixed run regardless of each class's mode, aligned
with the session-start semantics of `scenario_mix` weights (§11.4). Under saturation,
`sequential` classes self-throttle their in-flight turns (realistic user behaviour)
while session starts keep arriving — so queue growth remains observable.

A heavy-tailed (Pareto) arrival process is intentionally out of scope for v1 — tracked in
`TODOs.md`.

Each session start is assigned a workload class by a seeded categorical draw over the
`scenario_mix` weights (axis `mix`, §11.8); all turns of the session inherit the class.
Each arriving request is then routed to one of N server instances per `routing_strategy` (§12.4).

### 12.4 Routing strategies

- `random` (default): uniformly random instance selection per request.
- `session_affinity`: `session_idx % N` — same-session prompts always route to the same
  instance. Enables meaningful prefix-cache benefit across multi-turn sessions. Useful to
  measure the effect of session affinity vs random routing in multi-instance deployments.

---

## 13. Measurement

### 13.1 Request error tracking

Every failed request (`success=0`) is **kept** in the `requests` table (§14.3) — never dropped —
with its `error` column populated. The classification is used both for diagnosis and for
reporting error rates per λ level (see §15.1).

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

### 13.2 Per-session derived metrics

Every multi-turn scenario — conversational chat, long-context follow-ups, the v1 agentic
approximation, … — produces sessions whose turns share a `session_idx`. Session-level
metrics are derived at report time by the report notebook grouping the `requests` table
(§14.3) on that key — in mixed runs (§11.4) grouped per class first
(`GROUP BY scenario, session_idx`) and only then aggregated, so one class's sessions
never dilute another's statistics (§15.1):

| Per-session metric | Derivation |
|---|---|
| `session_e2e_ms` | Time from the first turn's send to the last turn's last token: `MAX(issued_at_ms + e2e_ms) − MIN(issued_at_ms)` within the session (§14.3) |
| `session_turns` | Number of requests with the same `session_idx` |
| `session_input_tokens` | `SUM(input_tokens)` per `session_idx` |
| `session_output_tokens` | `SUM(output_tokens)` per `session_idx` |
| `session_success` | `MIN(success)` per `session_idx` — the session succeeds only if every turn within it succeeded |

Single-turn scenarios are the degenerate case (session = single request, §11.7), where
these metrics collapse to the underlying per-request values.

**Boundary rule.** Session-level metrics are computed only over sessions **fully
contained in the sweep step** — started during warmup or measurement and completed by
drain end (§12.2); completion is detected by the presence of the session's `final_turn`
row (§14.3). Truncated sessions are excluded from the session-level aggregates
(their per-request rows still count toward request-level metrics), and the report
discloses the excluded count per λ level; a high truncation share at a given λ is
itself a saturation signal.

For **agentic scenarios** under the v1 approximation (per §11.7, where one task ≡ one
session), these session metrics also serve as task-level metrics — `session_e2e_ms` is
the task latency, `session_turns` is the fan-out depth, etc. A dedicated `agent_tasks`
table that carries truly task-specific signals (tool calls emitted, schema validity per
tool call, task identity distinct from session identity, possibly multiple tasks per
session) is deferred — see `TODOs.md` *Precise agentic / tool-calling measurement*.

### 13.3 Hardware utilization sampling

To detect untapped hardware headroom (per §1 *Observed execution*), the benchmarker
samples host-side telemetry on every inference-server node for the full duration of every
sweep step (warmup + measurement + drain). Samples are stored in the `hardware_stats`
table (§14.5).

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

Signals a platform cannot expose are stored `NULL`. Reports (§15) overlay these signals
against λ so that the "p95 TTFT meets SLO but GPU SM-active is 35%" case (untapped headroom)
is immediately visible to the reader.

### 13.4 Service-level objectives (SLOs)

The benchmark YAML's `slos` block declares the latency / reliability objectives the
experiment's results are judged against. Objectives are declared **per workload class**
(§11.4) because classes differ in what their users feel: chat is TTFT-sensitive, while
agentic-coding sessions are dominated by decode pace (`tpot_ms`) and total task time
(`session_e2e_ms`). The block is persisted verbatim to `experiments.slos` (§14.1) and
**evaluated at report time only** (§15.1) — the load generator neither enforces
admission control nor sheds load based on it.

| Field | Type | Notes |
|---|---|---|
| `slos[].scenario` | string | Class slug from `scenario_mix`, or `all` to apply the objective to every class. |
| `slos[].metric` | string | `ttft_ms` \| `tpot_ms` \| `e2e_ms` \| `session_e2e_ms` \| `error_rate_pct`. Session-level metrics per §13.2. |
| `slos[].percentile` | string | `p50` \| `p90` \| `p95` \| `p99`. Omitted for `error_rate_pct` (evaluated as the per-class failure fraction over the measurement phase). |
| `slos[].threshold` | number | Upper bound — milliseconds for latency metrics, percent for `error_rate_pct`. |

Example:

```yaml
slos:
  - { scenario: chat-short-turns, metric: ttft_ms,        percentile: p95, threshold: 800 }
  - { scenario: chat-short-turns, metric: tpot_ms,        percentile: p95, threshold: 80 }
  - { scenario: agentic-coding,   metric: session_e2e_ms, percentile: p90, threshold: 600000 }
  - { scenario: all,              metric: error_rate_pct,                  threshold: 1.0 }
```

**SLO-attained rate (λ\*).** For each sweep step, every objective is evaluated over the
measurement-phase requests of its class (warmup and drain excluded, §12.2). **λ\*** is
the highest swept λ at which **all** declared objectives hold simultaneously — the
experiment's goodput operating point, and the anchor for the supportable-users estimate
(§15.1). If no swept λ satisfies all objectives, λ\* is undefined and the report flags
it prominently (the sweep should be extended toward lower rates). λ\* is a **derived,
report-time quantity** — nothing cluster-side computes or persists it.

### 13.5 Response-quality evaluation

Sweep traffic is **ungradeable for quality by construction**: in the default
`output_length_mode: forced` (§11.6) every response is truncated or padded to a sampled
length. Response quality is therefore measured by a **separate evaluation phase** that
runs graded QnA suites — via the
[EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
— against the experiment's already-deployed OpenAI-compatible endpoint(s), with natural
decoding and suite-defined sampling parameters. Two stages, configured by the benchmark
YAML's `quality_eval` block:

**Stage A — pre-sweep sanity gate.** Runs after the inductor primer (§10.3), before the
first sweep step. A small graded subset checked against a **blunt absolute floor** whose
only job is detecting rubbish — corrupted weights, a broken chat template, a
mis-quantized checkpoint, a kernel-level correctness bug — **before** GPU-hours are
committed to the sweep. The floor is deliberately coarse; fine regressions are Stage B's
job. The gate runs in **every experiment** unless explicitly disabled
(`skip_quality_gate: true`; use sparingly, mirroring §8.5).

**Stage B — post-sweep quality comparison.** Runs after the final sweep step's drain, on
the same still-running deployment(s): the full configured suites, each at one or more
**eval-concurrency** levels (the eval traffic itself is the load, exposing
concurrency-dependent numerics such as batch-variant kernels). Stage B is **measurement,
not a gate**: one score set per deployment config, persisted to `quality_evals` (§14.9).
There is **no standing quality reference**: deltas are **experiment-internal** — when
the deployment sweep varies a quality-impacting knob (weight quantization,
`kv_cache_dtype`, a speculative-decoding implementation, …), the report (§15.1) pairs
each config's capacity (λ\*, supportable users) with its measured quality, so *"the
quantized config serves N× more users"* and *"the quantized config costs M pts on
GPQA"* are two columns of the same table in the same report. An experiment with a
single deployment config reports absolute scores, informational.

| `quality_eval` field | Default | Notes |
|---|---|---|
| `gate.suite` | `gsm8k` | Stage-A suite. |
| `gate.sample_size` | `100` | Subset size — minutes of wall-clock, not hours. |
| `gate.floor` | `0.5` | Blunt rubbish detector; per-model tuning tracked in TODOs. |
| `gate.on_fail` | `abort` | `abort` \| `continue`. A failed-but-continued gate marks the run's results **quality-flagged** (§15.1). |
| `compare.suites` | `[gsm8k, gpqa_diamond]` | Stage-B suites. GPQA-Diamond is HF-gated (license + auth token on the Benchmarker). |
| `compare.eval_concurrency` | `[1, 32]` | lm-eval parallel request counts; each level produces its own score rows. |
| `skip_quality_gate` | `false` | Disables Stage A only. |
| `skip_quality_compare` | `false` | Disables Stage B only (e.g. when no quality-impacting knob is swept). |

The consumed `quality_eval` block is persisted on `experiments.quality_eval` (§14.1) for
provenance.

---


## 14. Results

Per-run results live in a SQLite database file (`run_<id>.db`) with seven tables:
`experiments` (one row per sweep), `instances` (one row per deployed server instance),
`requests` (one row per issued request), `server_stats` (periodic samples of
server-side counters), `hardware_stats` (periodic samples of host hardware telemetry),
`system_prechecks` (one row per pre-check metric), and `quality_evals` (one row per
quality-eval measurement, §14.9). A first-class `agent_tasks`
table is deferred (see `TODOs.md` *Precise agentic / tool-calling measurement*); v1
derives per-task agentic metrics by grouping `requests` on `session_idx` (§13.2).

### 14.1 `experiments` table

One row per sweep — the configuration and overall outcome of the run.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, PK | Unique identifier (`timestamp + model_slug + backend + deployment + 4-hex random`; see §7.2). |
| `model` | TEXT | Model identifier (HuggingFace ID or path). |
| `backend` | TEXT | Inference engine (`vllm`, `sglang`, `dynamo`). |
| `backend_config` | TEXT (JSON) | Serialized `BackendConfig` — all fields from §16.2. |
| `dataset_config` | TEXT (JSON) | Serialized dataset configuration (§11). |
| `scenario_mix` | TEXT (JSON) | The workload mix: `[{scenario, weight}, …]` (§11.4). Single-scenario runs carry one entry with `weight = 1.0`. |
| `scenario_manifest` | TEXT (JSON) | Structured disclosure of what each class models, omits, and assumes, plus run-level assumptions. See §14.7. |
| `slos` | TEXT (JSON) | Serialized `slos` block (§13.4); `NULL` when the experiment declares none. |
| `quality_eval` | TEXT (JSON) | Serialized `quality_eval` block as consumed (§13.5); `NULL` when both stages are disabled. |
| `rate_levels` | TEXT (JSON) | List of λ values (session starts/s; §12.3 *What λ counts*) swept in this run. |
| `warmup_s` | INTEGER | Warmup phase duration in seconds (metrics excluded; see §12.2). |
| `measurement_s` | INTEGER | Measurement phase duration in seconds. |
| `created_at` | TEXT (ISO 8601) | Experiment start timestamp. |

### 14.2 `instances` table

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

### 14.3 `requests` table

One row per issued request — the per-request latency record.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `rate_lambda` | REAL | λ value (session starts/s; §12.3) of the sweep step this request belongs to. |
| `request_id` | INTEGER | Per-rate-level request index (monotonic). |
| `session_idx` | INTEGER | Session this request belongs to (§11.7). Shared by every turn of the session; enables grouping per-session for session-affinity routing analysis (§12.4). For single-turn scenarios equals the request's underlying prompt index. |
| `instance_id` | TEXT | Instance that served this request (§14.2). Required by routing-strategy analysis (§12.4): per-instance load and prefix-cache locality under `session_affinity` vs `random` are not reconstructible without it. |
| `scenario` | TEXT | Workload-class slug of the session this request belongs to (§11.4, §11.7). Constant across a session's turns; the key for per-class group-bys (§13.2, §13.4, §15.1). |
| `turn_idx` | INTEGER | 0-based position of this request within its session (§11.7). `0` for the first turn (and for every request in single-turn scenarios); `1` for the first follow-up; etc. Lets reports plot per-turn metrics directly (e.g. "TTFT vs turn index" to visualise prefix-cache benefit on follow-up turns) without reconstructing the order from timestamps. |
| `issued_at_ms` | REAL | Milliseconds from the sweep-step start at which the request was sent. Lets reports derive measurement-window membership (§12.2) and `session_e2e_ms` (§13.2) without extra state. |
| `final_turn` | INTEGER | `1` if this request is its session's last planned turn. A session is complete iff its `final_turn` row exists and every turn succeeded; sessions truncated at drain end lack it (§12.2, §13.2). |
| `ttft_ms` | REAL | Time to first token, milliseconds — authoritative SLO metric. |
| `tpot_ms` | REAL | Inter-token latency, mean across the request's output tokens. |
| `e2e_ms` | REAL | End-to-end request time, milliseconds. |
| `input_tokens` | INTEGER | Number of input tokens. |
| `output_tokens` | INTEGER | Number of generated output tokens. |
| `success` | INTEGER | `1` if completed within timeouts; `0` if client-side `request_timeout_s` exceeded or the server returned an error. |
| `error` | TEXT | Error message or class when `success=0`; `NULL` otherwise. |

### 14.4 `server_stats` table

Periodic samples of server-side counters during a sweep step. Sampling cadence is
backend-dependent. Samples are scraped **per instance** — for multi-instance deployments
(routing-strategy or replica-sizing experiments, §12.4 + §14.2), each instance produces
its own row stream so saturation on one instance is distinguishable from idleness on
another.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Instance the sample was scraped from. Composite key with `run_id` + `ts` + `rate_lambda`. Matches §14.2 / §14.5 / §14.6. |
| `rate_lambda` | REAL | λ value (session starts/s; §12.3) of the sweep step being sampled. |
| `ts` | TEXT (ISO 8601) | Sample timestamp. |
| `requests_running` | INTEGER | Requests currently executing on the server. |
| `requests_waiting` | INTEGER | Requests queued on the server. |
| `gpu_cache_pct` | REAL | KV cache utilization, percent. |
| `spec_accept_rate` | REAL | Speculative-decoding token acceptance rate; `NULL` if speculative decoding disabled. |

### 14.5 `hardware_stats` table

Periodic samples of host-side hardware telemetry on the inference-server node(s) during a
sweep step. See §13.3 for sampling cadence and per-signal meaning. GPU-scoped rows carry
a non-`NULL` `gpu_index`; node-scoped rows carry `gpu_index = NULL`.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Instance whose host this sample belongs to. |
| `rate_lambda` | REAL | λ value (session starts/s; §12.3) of the sweep step being sampled. |
| `ts` | TEXT (ISO 8601) | Sample timestamp. |
| `gpu_index` | INTEGER | GPU device index for GPU rows; `NULL` for node-wide rows. |
| `gpu_util_pct` | REAL | Coarse GPU activity (§13.3). |
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

### 14.6 `system_prechecks` table

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

### 14.7 Scenario manifest

Every result carries a structured **scenario manifest** that discloses what the benchmarked
workload mix models, what it explicitly does *not* model, and the numeric assumptions baked
in — per class and run-wide. Without this, a reader looking at a plot has no principled way
to know whether the result applies to *their* workload — e.g. a Pareto frontier built from
text-only large-prompt agentic-coding traffic may be wildly off for a chat workload
dominated by short turns and image inputs.

The manifest is stored in `experiments.scenario_manifest` as a JSON object with the
following required keys:

| Field | Type | Semantic |
|---|---|---|
| `mix` | list[object] | `[{scenario, weight, expected_request_share}, …]` — the workload classes, their session-start weights (§11.4), and the per-request share that follows from each class's turn structure. |
| `classes` | list[object] | One object per mix entry, carrying the per-class disclosure fields below. |
| `classes[].name` | string | Class slug; matches the corresponding `mix` entry. |
| `classes[].summary` | string | One- to two-sentence human description of the class. |
| `classes[].maturity` | string | One of `established` (validated against real-workload telemetry), `emerging` (early-signal, partially validated), `exploratory` (anticipated future pattern with no validation yet). The tag describes the maturity of the **workload pattern**; the fidelity of this scenario's *approximation* of that pattern is disclosed separately via `not_modelled` (e.g. `agentic-coding` is an `established` pattern modelled through the v1 fan-out approximation). Reports must visually flag `emerging` and `exploratory` Pareto frontiers as forward-looking so procurement readers can distinguish validated patterns from early signals. |
| `classes[].modelled` | list[string] | Aspects of real workload that the class *does* exercise — e.g. `"large multi-turn prompts (16K–32K input tokens)"`, `"follow-up turns reusing the initial context"`. |
| `classes[].not_modelled` | list[string] | Aspects the class explicitly does *not* cover — e.g. `"no image inputs"`, `"no audio inputs"`, `"no reasoning / thinking traces"`, `"no tool-call interleaving"`. |
| `classes[].assumptions` | list[string] | Numeric or structural assumptions baked into the class — e.g. `"follow-up turn probability = 0.4"`, `"max output tokens = 4096"`, `"input length distribution: lognormal, mean=20K, σ=0.3"`, `"system prompt length: 1.2K tokens, identical across sessions"`. |
| `run_assumptions` | list[string] | Run-level assumptions shared by all classes: arrival process + parameters (§12.3), routing strategy (§12.4), `output_length_mode` (§11.6), master seed, tokenizer ID. |

Validation: the Coordinator aborts **before submission** if the benchmark YAML's
`scenario_mix` is missing or empty, names an unregistered scenario (§11.3), or carries
weights that do not sum to 1.0. The dataset generator on the Benchmarker aborts **before
load-generation begins** if any required field of the emitted `scenario_manifest` is
missing or fails schema validation (matching §11.8). There is no implicit default —
every benchmark must declare what it is and is not.

### 14.8 Experiment directories

Each completed sweep produces an `experiments/YYYY-MM-DD_description/` folder containing:

- `benchmark_config.yaml` (copy of the input config for provenance)
- the run's SQLite DB file (`run_<id>.db`)
- deployment artifacts used for the run (sbatch scripts, Kubernetes YAML, Dockerfile)
- the executed report notebook and its rendered outputs (see §15)

### 14.9 `quality_evals` table

One row per quality-eval measurement (§13.5) — per stage, suite, and eval-concurrency
level.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id` (one run per deployment config; cross-config comparisons join across runs in the report). |
| `instance_id` | TEXT | Instance/endpoint the eval targeted; `NULL` when routed across instances. |
| `stage` | TEXT | `gate` (Stage A) or `compare` (Stage B). |
| `suite` | TEXT | Suite identifier (e.g. `gsm8k`, `gpqa_diamond`). |
| `eval_concurrency` | INTEGER | lm-eval parallel request count used for this measurement. |
| `sample_size` | INTEGER | Number of items graded. |
| `metric` | TEXT | Suite metric name (e.g. `exact_match`). |
| `score` | REAL | Measured score. |
| `floor` | REAL | Stage-A floor in effect; `NULL` for `compare` rows. |
| `status` | TEXT | `pass` / `fail` for `gate` rows; `NULL` for `compare` rows (measurement, not a gate). |
| `sampling_params` | TEXT (JSON) | Sampling parameters actually used (temperature, top_p, …). |
| `harness_version` | TEXT | lm-eval version + task version, for provenance. |
| `ts` | TEXT (ISO 8601) | Time the measurement completed. |

---

## 15. Reporting

The Reports generator produces a Jupyter notebook from the centralized results database
and writes it back into the experiment directory.

### 15.1 Report notebook (`experiments/template_report.ipynb`)

Every experiment report must include:

- Experiment title and description
- **Scenario & assumptions panel** (from `experiments.scenario_mix` and
  `experiments.scenario_manifest`, §14.7): the mix table (class, session-start weight,
  expected request share), then per class: name, one-line summary, the `modelled`
  list, the `not_modelled` list, and the `assumptions` list, plus the run-level
  `run_assumptions` — surfaced near the top of the report, before any plot, so every
  downstream chart is read in the context of what each class actually does and does
  not cover. Items in `not_modelled` must be visually distinguished (e.g.
  struck-through or in a warning-coloured panel) so a reader cannot miss them.
- Configuration summary table (model, TP, KV dtype, spec dec, SLO, etc.)
- **System pre-checks** (from `system_prechecks`, §14.6): table of pre-check metrics with
  measured / expected / status — warns and fails flagged prominently at the top of the
  report so a degraded foundation is impossible to overlook when interpreting downstream
  numbers.
- **Model loading times**: per instance (from the `instances` table, §14.2),
  `model_load_total_s` plus the per-component breakdown (`model_load_weights_s`,
  `model_load_engine_init_s`, `model_load_cuda_graph_capture_s`,
  `model_load_inductor_compile_s`) — see §10.2.
- TTFT p50/p95/p99 vs λ plot (log scale) with per-class SLO lines (§13.4)
- ITL p50/p95/p99 vs λ plot
- Failure rate bar chart (bottom panel of each plot)
- **Per-class breakdowns** (mixed runs, §11.4): every latency and failure-rate panel is
  rendered both aggregate and grouped by `requests.scenario`, and the §13.2
  session-level metrics are derived per class (`GROUP BY scenario, session_idx`) — so
  cross-class interference (e.g. long agentic prefills inflating chat TTFT on the
  shared instance) is directly visible rather than averaged away.
- **SLO attainment** (from `experiments.slos`, §13.4): a per-objective pass/fail table
  per λ level, the derived λ\* highlighted, and each class's SLO thresholds drawn on
  its respective panels.
- **Supportable-users estimate** — the λ→users translation, computed **only here** (and
  in curated reports built on top, §15.3); nothing cluster-side computes it. The
  notebook exposes editable per-class parameters (`sessions_per_user_per_hour`) and
  combines them with quantities measured at λ\*: per-class session throughput (sessions
  started per second during the measurement phase) and mean session wall-time (§13.2).
  It reports, per class, (a) the **supportable user population** = session throughput ÷
  per-user session rate, and (b) **concurrent active sessions** via Little's law =
  session throughput × mean session wall-time. Always presented as an estimate, with
  the parameters disclosed alongside the result. Undefined when λ\* is undefined.
- **Response-quality panel** (from `quality_evals`, §14.9 + §13.5): the Stage-A gate
  outcome — with an unmissable **quality-flagged** banner if the gate failed under
  `on_fail: continue` — and the Stage-B per-suite scores per eval-concurrency level.
  When the experiment's deployment sweep varies a quality-impacting knob, a
  **capacity-vs-quality table**: per deployment config, users at λ\* alongside quality
  scores and the deltas between configs — the *"N× more users, −M pts"* pairing in one
  view, in the same report as the capacity claim.
- **Hardware utilization** (from `hardware_stats`, §14.5), per λ level, overlaid against
  TTFT/ITL so untapped headroom is visible at a glance:
  - GPU SM-active and tensor-active vs λ (the key headroom indicator — SLO met with these
    well below 100% means the system can take more traffic on the same allocation)
  - GPU power, HBM bandwidth, and memory occupancy vs λ
  - NVLink and PCIe throughput vs λ
  - Node CPU, RAM, storage-read, network rx/tx vs λ
- Raw per-rate-level data table

### 15.2 Notebook output

The executed notebook (`report.ipynb`) and its rendered plots (`ttft.png`, `itl.png`,
`hardware.png`, `prechecks.png`) are written into the corresponding
`experiments/YYYY-MM-DD_description/` folder (§14.8).

### 15.3 Curated reports (`reports/`)

The per-experiment notebooks in §15.1–§15.2 exist for **reproducibility**: every plot the
template renders, the raw per-rate-level table, full provenance back to the SQLite DB. An
`experiments/<run>/` folder is self-contained: everything about that one experiment —
config, raw results, notebook, rendered plots — lives there.

A separate top-level `reports/` directory holds **curated, audience-facing Markdown
reports** intended for stakeholders to read **directly on GitHub** in their browser —
procurement decision-makers, capacity planners, leadership. Each curated report draws
plots and numbers from one or more `experiments/<run>/` folders, framed by a narrative
the operator authors **interactively with an AI coding assistant** (such as Claude Code,
per the README's operating model). The operator supplies the intent (*"compare
speculative-decoding effect across Apertus-70B and Kimi-K2.6 for procurement"*) plus
iterative adjustments (*"focus on cost-per-task; drop the storage panel; SLO line at 250 ms"*);
the assistant assembles the Markdown, references the source experiment artefacts, and
iterates with the operator until the report lands. One per-experiment notebook may feed
several curated reports targeting different audiences; one curated report may span
several experiments.

#### Folder structure

```
reports/
  STYLE.md              # reusable styling decisions Claude applies when drafting any curated report
  <topic-slug>/
    <report-name>.md    # the Markdown report — what GitHub renders for stakeholders
    figures/            # optional: PNGs re-rendered with audience-tuned styling
```

Plots are embedded with standard Markdown image syntax. The simplest case points at the
per-experiment notebook output via relative path (`../../experiments/<run_id>/<plot>.png`),
which is a verbatim reuse — no risk of contradicting the source. When a re-render is
needed (different percentile selection, different scale, audience-tuned fonts), the
re-rendered PNG lives under `figures/` and the narrative explicitly flags the styling
change.

#### Reusable styling decisions (`reports/STYLE.md`)

To prevent the same styling adjustments from being re-derived for every new curated
report, `reports/STYLE.md` captures styling decisions the operator has already made — per
audience and globally. When drafting a curated report, the AI coding assistant **reads
STYLE.md first** and honours its directives by default; the operator only intervenes when
STYLE.md does not yet cover the case at hand. When a new rule emerges from a session of
iterative adjustments — e.g. *"I keep asking to hide the NVLink panel for procurement
audiences"* — the operator adds it to STYLE.md and every subsequent report inherits it
without re-asking.

STYLE.md is operator-curated and grows over time. Typical entry shapes:

- **Per-audience sections**: *"For procurement, prefer linear x-axis on TTFT-vs-λ plots,
  show p50 and p95 only, hide the storage and NVLink panels."*
- **Global decisions**: *"SLO line is always dashed red (#cc0000)"*, *"Per-model colour
  is fixed across all reports — Apertus-70B → blue, Kimi-K2.6 → green."*
- **Audience-aware narrative defaults**: *"Procurement reports lead with the
  cost/throughput claim, not with TTFT."*

STYLE.md is the single source of truth for "how curated reports look". It does **not**
override the data-integrity contract below: no styling rule may hide a SLO breach, a
failure-rate spike, or any other signal a stakeholder needs to see — only how visible
material is presented.

#### Required content

Each curated report must include:

- A **headline** capturing the audience-relevant claim (e.g. *"Speculative decoding cuts
  TTFT p95 by 35 % at the SLO-met λ on Apertus-70B."*).
- A **narrative** framing the plots — what was measured, in what context, what to read
  from each figure.
- The **selected plots and tables**, embedded inline.
- A **provenance block** at the end of the Markdown: the source `run_id`s, the model +
  BackendConfig of each, the date of curation, and the assistant-session reference if
  available — so any figure can be traced back to its underlying experiment.

#### Contract

Nothing in `reports/<topic>/` may contradict the underlying experiment data — same
percentiles, same SLO line, same λ range. If a re-render changes any of these (e.g.
dropping p99 for a procurement audience to keep the chart clean), the change is called
out explicitly in the narrative, with a pointer to the per-experiment notebook where the
full picture lives.

---

## 16. Experiment Plans

Each experiment is composed from the features being measured (§16.1), the **deployment
target** (SLURM vs Kubernetes — frequently a sweep dimension in its own right, §6), the
**backend and its version / variant** under test (§9.1; e.g. one vLLM release vs
another, or upstream vs a CSCS-patched build — comparing two versions of the same backend is a first-class
experiment shape), the BackendConfig knobs that vary across deployments within the
experiment (§16.2), and the model(s) under test (§9.2). The benchmark YAML specifies
all five, plus the workload mix the deployment is loaded with (`scenario_mix`, §11.4),
the SLOs the results are judged against (`slos`, §13.4), and the quality-evaluation
configuration (`quality_eval`, §13.5). An experiment thus has two nested sweeps: the **deployment sweep** over
deployment-target × backend-version × BackendConfig × model combinations (one engine
launch per combination), and inside each deployment the **rate-level sweep** over λ
values (each λ being one "sweep step" in the sense used by the `requests` /
`server_stats` / `hardware_stats` tables).

The lists in §16.1 and §16.2 are **deliberately non-exhaustive** — they capture the v1
priorities so the implementation phase has concrete context to build against, but
experiments routinely introduce additional features, modes of usage, or backend knobs on
a per-experiment basis. Claude adapts the framework to support each addition as the
operator requests it: extending the BackendConfig surface, the planner templates, the
manifest disclosure surface, and the report panels in lock-step.

### 16.1 Features under test

The framework benchmarks the inference-serving features listed below. The list is
**non-exhaustive** — it captures the v1 priorities to give the implementation phase
concrete context, not a closed set. Experiments routinely require additional features or
modes of usage; new entries are added by extending this list, defining any new config
knobs in §16.2, binding to at least one scenario, and surfacing the marginal effect in
reports (§15). Each feature is exercised by one or more scenarios (see the scenario
taxonomy in the README).

| Feature | Why it matters | Where configured | Procurement implication |
|---|---|---|---|
| **Automatic prefix caching** | Reduces TTFT for sessions sharing prompt prefixes; critical for chat and AI-assisted coding. | `enable_prefix_caching` (§16.2) | Cache-friendly KV memory hierarchy; cache hit-rate as a procurement metric. |
| **KV-cache offloading** | Extends effective KV capacity by spilling to host DRAM / unified memory; trades per-request latency for concurrency. | `kv_offloading_size`, `kv_offloading_backend` (§16.2) | **Memory-layer sizing decisions** — HBM vs Grace-DRAM vs CXL. Offloading bandwidth profiles drive host-DRAM-per-GPU sizing and the choice of unified-memory / CXL fabrics for next-generation systems. |
| **KV-cache reuse across requests** | Identical or partially-overlapping prefixes from different requests reuse already-computed KV; effectiveness depends on routing. | `enable_prefix_caching` (§16.2) + `routing_strategy` (§12.4) | KV memory pressure under realistic locality; informs replica-pool sizing. |
| **Speculative decoding** | Improves decode throughput when a smaller draft model proposes tokens accepted by the target. | `speculative_decoding.*` (§16.2) | Compute headroom for draft model; memory budget for two-model deployments. |
| **Continuous batching** | Schedules new requests into running batches without waiting for current ones to finish — the dominant throughput optimization for online serving. | Backend default; not directly exposed | Scheduler responsiveness characterisation; admission-control budget. |
| **MoE expert routing** | Token-to-expert dispatch and load balance govern memory pressure and inter-GPU traffic. | Observed via §13.3 telemetry (NVLink / PCIe all-to-all signals) and §8 NVSHMEM perftest; not configured at the framework layer. | Interconnect sizing for all-to-all expert traffic; hot-expert memory pressure. |
| **Quantization (weights / KV / activation)** | Trades model fidelity and memory footprint against throughput. | `kv_cache_dtype` (§16.2); weight quantization via backend | Memory hierarchy: lower-precision math support vs higher-precision storage. |
| **Disaggregated prefill / decode** | Splits compute-heavy prefill from memory-bandwidth-heavy decode across different accelerator classes. | Per-component `nodeSelector` (§6.2); KV-transfer mechanism deferred (TODOs.md *NIXL disaggregated prefill/decode*). | Heterogeneous accelerator procurement; interconnect bandwidth between roles. |
| **Multi-replica routing and session affinity** | Distributes load across replicas; `session_affinity` preserves prefix-cache hits at the cost of fairness. | `routing_strategy` (§12.4) | Ingress / load-balancer requirements; cache-locality vs replica-fairness trade-off. |

Each feature's contribution to latency, throughput, error rate, and hardware utilisation
(§13.3) is recorded per sweep — and, for quality-impacting knobs (weight quantization,
`kv_cache_dtype`, speculative-decoding implementations, …), response quality (§13.5
Stage B), so reports pair each feature's capacity gain with its measured quality change
(§15.1). Reports plot the marginal effect of enabling / disabling individual features so
procurement evidence can isolate the value of each.

**Platform comparison (SLURM ↔ Kubernetes).** Independent of any specific feature,
comparing the same workload across **SLURM** (clariden / bristen / beverin) and
**Kubernetes** (breithorn) is a frequent first-class sweep dimension. The benchmark
YAML's `deployment_target` selects the platform; the engine, BackendConfig, model, and
scenario are otherwise held constant so the comparison isolates the platform's own
contribution — ingress, scheduling, lifecycle, networking, container-engine startup,
storage path. The platform-comparison sweep composes orthogonally with any feature
comparison above, so a single experiment can answer "does feature X behave differently
on SLURM vs Kubernetes?" by sweeping both axes together. Reports plot per-platform
overlays for any latency / throughput / utilisation panel.

---


### 16.2 Sweepable backend configuration

All fields are optional (sensible defaults apply); the experiment's **deployment sweep**
iterates over combinations of them, instantiating one engine deployment per combination
(λ is then swept inside each deployment — see §16 intro). Values flow from `BackendConfig`
through Coordinator → Planner (§5) → backend-specific Jinja2 templates (§10) into the
engine launch command.

**Backend-by-backend, version-by-version.** Each backend has its own knob surface —
same conceptual configurations (TP, PP, KV dtype, prefix caching, speculative decoding,
…) expressed through that backend's flags — and the knob surface itself shifts across
versions (flags get removed, renamed, or made mandatory; see the flag-compatibility
block under vLLM). The subsections below are **non-exhaustive** and **per-version**:
they capture the v1 knob surface for the active backend-and-version pairs so the
implementation phase can wire planner templates and Jinja2 renderers; new versions of
the same backend get their own subsection when introduced, and additional knobs are
added per-experiment as the operator requests them. Claude extends the surface
(BackendConfig field + Jinja template branch + the relevant table here) when a new knob
or version comes into scope.

#### vLLM — v0.22.x

| Field | vLLM flag | Notes |
|---|---|---|
| `tensor_parallel_size` | `--tensor-parallel-size` | Default 1. Must not exceed `gpus_per_node` (= 4 on Alps) — per-layer all-reduce is bandwidth-heavy; see §6.1. |
| `pipeline_parallel_size` | `--pipeline-parallel-size` | Default 1. Use for cross-node scale-out — PP traffic is much lighter than TP. |
| `data_parallel_size` | `--data-parallel-size` | Default 1. Each DP replica is an independent `instances` row (§14.2). |
| `expert_parallel_size` | `--expert-parallel-size` | Default 1. MoE engines only. |
| `max_model_len` | `--max-model-len` | |
| `max_num_batched_tokens` | `--max-num-batched-tokens` | Must equal `max_model_len` for long-context (avoids chunked-prefill rejection) |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | |
| `kv_cache_dtype` | `--kv-cache-dtype` | e.g. `"fp8"`. Doubles KV capacity but worsens per-request latency due to higher batch concurrency. |
| `enable_prefix_caching` | `--enable-prefix-caching` | Default True. Set False to isolate TTFT from cache artefacts (but prefer unique prompts instead). |
| `safetensors_load_strategy` | `--safetensors-load-strategy` | `"prefetch"` recommended on Lustre (capstor / iopsstor) — see §6.1. |
| `kv_offloading_size` | `--kv-offloading-size` | Total GiB across all TP ranks (e.g. `400` = 100 GiB/GPU for TP=4). Uses GH200 Grace DRAM at 900 GB/s via NVLink-C2C. |
| `kv_offloading_backend` | `--kv-offloading-backend` | `"native"` (default). |
| `speculative_decoding.draft_model` | part of `--speculative-config` JSON | Draft model identifier (HuggingFace ID or path). See the vLLM compatibility notes below. |
| `speculative_decoding.num_speculative_tokens` | part of `--speculative-config` JSON | |
| `speculative_decoding.draft_tensor_parallel_size` | part of `--speculative-config` JSON | Draft tensor-parallel size. Shared- vs dedicated-GPU guidance in §17.2. |

**Flag compatibility.** The vLLM flag names above are stable on the current vLLM pin.
The table below records flags that were removed, renamed, or made mandatory in this
release, so that planner templates don't regress to deprecated syntax when the pin moves.

| Flag | Status | Notes |
|---|---|---|
| `--speculative-model` / `--num-speculative-tokens` | ❌ Removed | Use `--speculative-config '{"model":..., "num_speculative_tokens":N, "draft_tensor_parallel_size":M}'` |
| `--swap-space` | ❌ Removed | Use `--kv-offloading-size` |
| `--kv-offloading-size` | ✅ Available since v0.20.0 | Total GiB across all TP ranks |
| `--kv-cache-dtype fp8` | ✅ Works | Native FP8 hardware on SM90 (GH200) |
| `--enable-prefix-caching` | ✅ Works | |
| `--safetensors-load-strategy` | ✅ Works | |
| `--speculative-config` | ✅ Works | JSON string |
| `VLLM_ENABLE_CUDA_COMPATIBILITY=1` (env) | ❌ Must not be set on current GH200 drivers | Causes Error 803. |

#### SGLang

*Populated as SGLang experiments come into scope.* The same conceptual configuration
surface as vLLM — TP / PP / DP, KV dtype, prefix caching, speculative decoding, KV
offloading, MoE expert parallelism — expressed through SGLang's flags. Planner template:
`tools/templates/slurm/sglang.edf.j2` (TBD when the first SGLang experiment is wired).

| Field | SGLang flag | Notes |
|---|---|---|
| *TBD* | *TBD* | First SGLang experiment will seed the table; Claude adds rows as knobs are exercised. |

#### Dynamo

*Populated as Nvidia Dynamo experiments come into scope.* The same conceptual
configuration surface as vLLM, expressed through Dynamo's flags. Planner template:
`tools/templates/slurm/dynamo.edf.j2` (TBD when the first Dynamo experiment is wired).

| Field | Dynamo flag | Notes |
|---|---|---|
| *TBD* | *TBD* | First Dynamo experiment will seed the table; Claude adds rows as knobs are exercised. |


---

## 17. Findings Records

Computed properties and operational findings recorded from running experiments. New
entries are appended as findings emerge.

### 17.1 KV cache capacity (GH200, 70B, TP=4)

```
Available KV per GPU  = (96 GiB × gpu_memory_utilization) − (140 GiB / 4 GPUs)
                      ≈ 51 GiB at 0.90

KV per 25K-token request per GPU  ≈ 80 KB/tok × 25,000 = 1.91 GiB
Max concurrent requests            ≈ 51 / 1.91  ≈ 28

With --kv-offloading-size 400 (100 GiB/GPU via Grace DRAM):
Additional KV capacity  = 100 / 1.91 ≈ 52 additional slots
Total concurrent        ≈ 80 slots
```

### 17.2 Speculative decoding on shared GPUs

- Running draft at TP=4 on the **same** 4 GPUs as the 70B target is counterproductive:
  NCCL allreduce overhead for 5 draft passes per cycle (320 extra syncs) erases the
  memory bandwidth benefit.
- Optimal configuration when GPUs are shared: **draft TP=1**, accept reduced GPU-0 KV capacity.
- Optimal configuration when dedicated GPUs are available: **draft on separate node/GPUs**.

---

## 18. Known Issues & Workarounds

Issues discovered during image builds and experiment runs — and the workarounds that
resolved them — are tracked here as they arise.

### 18.1 Engine image cache is stale across rebuilds under the same tag

The container engine (enroot/pyxis) caches an imported image keyed by reference, so
rebuilding and re-pushing under the **same tag** leaves the cached squashfs pointing at
the old digest — a sanity or experiment run then silently uses the *previous* image.
**Workaround:** pin the launch EDF by the **registry manifest digest**
(`image = "<host>#<repo>@sha256:…"`), captured from `podman push --digestfile`. Do **not**
use `podman image inspect {{.Digest}}` — that is the *local* manifest digest, which a
format-converting push can change, and the registry returns `404` for it.
`tools/images/build.sh` writes the digest-pinned EDF that `sanity.sbatch` consumes.

### 18.2 Slim Ubuntu/CUDA bases need extra netstack steps vs NGC

NGC bases ship a complete CUDA toolkit, an HPC-X stack, and `/bin/sh`→bash; slim engine
bases (e.g. `vllm/vllm-openai`, Ubuntu 24.04 / CUDA 13) do not. Building the Alps network
stack on them therefore requires, as a principle (the exact steps live in the netstack
phase scripts and Containerfile):

- installing the CUDA dev components the stack links against but the slim toolkit omits —
  `cuda-nvml-dev` (libfabric `nvml.h`), `cuda-nvrtc-dev` (NVSHMEM `CUDA::nvrtc`), and a
  `libnvJitLink` runtime (present only in a pip wheel) symlinked into the toolkit;
- a `python`→`python3` alias (the slim base ships only `python3`);
- `ENV BASH_ENV=/etc/bash.bashrc` so the baked env reaches non-login `bash -c` (NGC bases
  set it; the slim base leaves it unset);
- a **POSIX-sh-safe** runtime env file — the container init sources it under **dash**,
  which rejects bash-only `${!x}` / `printf -v` / `[[ ]]` with "Bad substitution" and
  aborts container start.

The §9.1 post-push acceptance gate is what catches a base that silently lacks one of
these (e.g. collectives that fall back off the Slingshot path).
