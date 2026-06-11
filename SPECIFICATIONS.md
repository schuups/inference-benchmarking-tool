# Inference Benchmarking — Specification

This document enumerates all requirements captured so far.

## Table of Contents

1. [Guiding Principles](#1-guiding-principles)
2. [Project folder structure](#2-project-folder-structure)
3. [Pre-flight checks](#3-pre-flight-checks)
4. [Planner](#4-planner)
5. [Deployment Targets](#5-deployment-targets)
6. [Resources Lifecycle and Cleanup](#6-resources-lifecycle-and-cleanup)
7. [System Performance Pre-checks](#7-system-performance-pre-checks)
8. [Backends and models under test](#8-backends-and-models-under-test)
9. [Inference Engine Bring-up](#9-inference-engine-bring-up)
10. [Prompt Generation](#10-prompt-generation)
11. [Load Generation](#11-load-generation)
12. [Measurement](#12-measurement)
13. [Results](#13-results)
14. [Reporting](#14-reporting)
15. [Experiment Plans](#15-experiment-plans)
16. [Findings Records](#16-findings-records)
17. [Known Issues & Workarounds](#17-known-issues--workarounds)

---

## 1. Guiding Principles

- **Laptop-orchestrated**: the Coordinator runs on the operator's laptop — submits the
  Benchmarker job, monitors execution, collects per-run results into the centralised DB,
  tears down. Cluster-side sequencing (engine spawn, dataset gen → load gen) is the
  Benchmarker's job. The laptop carries orchestration and the results DB; the cluster
  carries the work.
- **Backend-agnostic**: vLLM, sglang, and NVIDIA Dynamo are first-class backends. Others
  can be added later by providing blueprint / example files.
- **Open-loop stochastic load generation**: requests arrive at mean rate λ via Poisson or
  burst-aware processes (§11.3), independent of server completions. This captures the
  **queuing dimension** of real load — backlog, saturation, latency amplification. It is
  one of two axes of realism. The other — semantic realism: prompt content, multi-turn
  structure, fan-out, modality mix — lives in the scenario registry (§10) and is equally
  necessary.
- **Reproducible by config**: a single YAML file fully specifies the experiment and its
  parameter sweep. Re-running the same file must produce comparable results.
- **Scenario-disclosed results**: every plot is published with its scenario manifest,
  which describes the experimental context — including the assumptions made — so the
  reader can readily interpret the results (§13.7, §14.1).
- **Separation of concerns**: the **Benchmarker** runs as its own SLURM allocation,
  sequencing three phases — **dataset generation**, then spawning the **inference
  deployment** on a separate GPU allocation, then **load generation**. GPUs are not
  occupied until the dataset is ready (no idle during prompt prep), and the Benchmarker's
  separate allocation keeps its CPU work from competing with engine GPU inference.
- **Validated foundation**: every experiment is preceded by micro-benchmarks
  (NCCL / RCCL collectives, NVSHMEM, storage — §7) checked against per-system references.
  They run in the engine's container session — same libfabric / CUDA / NCCL / mounts /
  NUMA — so the measured foundation is the one the engine sits on. A degraded foundation
  pauses the sweep and offers the operator an abort.
- **Observed execution**: GPU, CPU, memory, storage, network, and power telemetry is
  sampled per inference-server node throughout every sweep (§12.3), so untapped headroom
  is distinguishable from saturation.
- **Clean cluster state**: all deployed resources must be cleaned up after every run — on both
  success and failure paths. No orphaned jobs, pods, services, secrets, or scratch directories.
- **Flexible process**: tool development and experiment design + execution follow a soft
  sequential process — the canonical phase sequence and the skip / rewind rules live in
  `CLAUDE.md` *How we work together*.

---

## 2. Project folder structure

### 2.1 Laptop (repository root)

| Folder | Purpose |
|---|---|
| `.venv/` | uv-managed Python 3.14 virtualenv. Activation and invocation details in *Constants* below. |
| `examples/` | Reference deployments and build scripts (Docker image builds, K8s and SLURM deployments, NCCL pre-checks). Claude consults these when building the benchmarking tool itself. |
| `firecrest-mcp/` | FirecREST MCP server. **Started manually by the operator** before a session — not auto-managed. |
| `tools/` | Implementation of the laptop-side components (Coordinator, Planner, Pre-flight checker, Cleaner, Reports generator). |
| `experiments/` | Per-experiment folders (`YYYY-MM-DD_description/`) with config, deployment artifacts, raw results. See §6.2 for the run-ID format and §13.8 for the directory contents. |
| `reports/` | Curated, audience-facing reports synthesised from one or many `experiments/` runs (§14.3). |

The repository root also carries `SPECIFICATIONS.md`, `CLAUDE.md`, `TODOs.md`, and `README.md` as the four authoritative documents.

### 2.2 Remote (cluster scratch)

On SLURM clusters, the operator's scratch base is `/capstor/scratch/cscs/$USER/` (Lustre, HDD; see §5.1). Each experiment creates a run-specific subdirectory holding the Benchmarker's working files (sbatch + EDF copies), the dataset generator's prompt pool, and the inference deployment's container working directory. A shared `nccl-tests-cache/` lives at the same level — one entry per stack fingerprint (§7.2).

On Kubernetes (`breithorn`), the equivalent layout lives under Ceph-backed PVCs scoped per experiment.

Remote scratch is **transient** for a given run: per-run subdirectories are reclaimed by the cleanup phases in §6.

---

## 3. Pre-flight checks

A laptop-side **Pre-flight checker** runs at the start of every new session to verify
that the target systems are ready to accept work. It catches operator-environment problems
(missing credentials, unreachable APIs, exhausted K8s capacity) early.

Distinct from the *System Performance Pre-checks* in §7: those run **inside the engine
container on the cluster** and validate hardware; the Pre-flight checks run **on the
laptop** and validate access to the targets.

Required checks (fail-fast; the operator sees a single error message naming the first
failing check):

| Plane | Check | Validates |
|---|---|---|
| Auth — ML Platform | FirecREST MCP "ML Platform" server responds | Credentials for `clariden` and `bristen` (both SLURM clusters under MLP — one credential set covers both, per §5.1) |
| Auth — HPC Platform | FirecREST MCP "HPC Platform" server responds | Credentials for `beverin` |
| Auth — K8s | `kubectl get nodes` against `breithorn` succeeds | kubeconfig present; Rancher cluster reachable |
| K8s capacity (`breithorn` only) | At least one node has all GPUs free | Schedulability. If free GPUs are fragmented (scattered across nodes with none aggregated per-node), the operator defrags via external K8s tools before retrying. |
| Filesystem | The capstor scratch dir (`/capstor/scratch/cscs/$USER/`) exists and is writable | The dataset generator can write the prompt pool to capstor scratch. |
| Podman storage config | `~/.config/containers/storage.conf` is present (or `$XDG_CONFIG_HOME/containers/storage.conf` if `XDG_CONFIG_HOME` is set) | Required for podman container operations on Alps (image build, EDF import). Contents per the [CSCS container docs](https://docs.cscs.ch/build-install/containers/). |

Implementation: `tools/pre-flight-checks.py`.

---

## 4. Planner

A laptop-side **Planner** prepares the artifacts the rest of the workflow consumes — it
takes the operator's intent and produces a self-contained experiment directory:

- The backend container EDF (SLURM `--environment=` TOML) or K8s deployment / service /
  ingress manifests, with §15.2 server-config knobs rendered into engine flags.
- The Benchmarker sbatch (SLURM) or pod spec (K8s), with the concatenated pre-check →
  dataset prep → engine spawn → load gen chain (§7.2) wired up.
- The benchmark YAML's `dataset_config` block (§10.4), including the resolved scenario
  reference.

The Planner runs entirely on the laptop, against Jinja2 templates checked into the repo,
and does not touch the cluster — its output is config; nothing is submitted. The operator
drives the Planner through Claude Code (`/plan` or freeform conversation) or via a direct
CLI entry point; both paths produce the same artifacts. Planner output is then handed off
to the Coordinator at submission time.

The Planner does **not** persist any state of its own — every artifact it produces lives
in the experiment directory (§13.8), so a re-render is fully reproducible from the
benchmark YAML alone.

---

## 5. Deployment Targets

### 5.1 SLURM

Applies to all three SLURM clusters (`clariden`, `bristen`, `beverin`).

- All jobs (inference deployment, Benchmarker, image builds) submit to the cluster's
  default partition: **`normal`** on `clariden` and `bristen`, **`mi300`** on `beverin`.
  NCCL/RCCL benchmarks are **not** a separate job — they run inside the engine's container
  instance as part of the System Performance Pre-checks (§7).
- Account: `csstaff` (or `a-csstaff`). Never use other accounts.
- **Time-limit alignment**: every SLURM job in a single experiment — inference deployment
  and Benchmarker — must be configured with the **same** time limit, set conservatively
  enough to cover (chronologically): dataset generation + model load + CUDA graph capture +
  inductor compilation primer + full sweep + results finalisation (writing the per-run DB
  and staging outputs into the experiment directory). Coordinator-driven cleanup runs
  *after* the SLURM job exits and is outside the time limit (§6).
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
guidance (§16.1, §16.2, §9.3) as not portable to `bristen`.

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

### 5.2 Kubernetes (`breithorn`)

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
  `benchmarker_time_limit` of the same experiment (see §5.1).

---

## 6. Resources Lifecycle and Cleanup

Resources created during a run are labelled at creation (§6.1) and assigned a unique
run ID (§6.2). Per-run teardown (§6.3 → §6.6) runs on both success and failure paths at
end-of-run. Periodic operator-driven cleanup (§6.7) reclaims state that escaped per-run
teardown.

### 6.1 Resource identification

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

### 6.2 Run ID uniqueness

- Run IDs include: timestamp + model slug + backend + deployment + 4-hex random suffix.
- The random suffix prevents collision when multiple Coordinators start within the same
  second.

### 6.3 Benchmarker teardown

- Cancel the Benchmarker SLURM job (`scancel <job_id>`).
- Delete the Benchmarker's capstor scratch run directory
  (`/capstor/scratch/cscs/$USER/<run_id>/`), which holds the dataset, working files, and
  load-gen state.

The Benchmarker spawned the inference deployment(s) but its cancellation does **not**
automatically cancel them — the per-target sections below handle that explicitly.

### 6.4 Inference deployment teardown — SLURM

- Cancel all inference-deployment SLURM jobs (`scancel`).

### 6.5 Inference deployment teardown — Kubernetes

- Delete: Deployment, Service, Ingress, TLS Secret (`<name>-cert`).

### 6.6 K8s PVC retention

Model-cache PVCs (`model-cache-<model-slug>`) are **intentionally kept** across runs to
avoid repeated 20–30 min weight downloads. The Coordinator's per-run teardown leaves
them in place.

### 6.7 Periodic cleanup (Cleaner)

State that escapes the Coordinator's per-run teardown — Coordinator killed mid-run,
network failure during teardown, older runs predating a teardown fix — is reclaimed by
a separate **Cleaner** that is **run manually by the operator** — never by Claude, never
by the Coordinator automatically.

Each Cleaner invocation has **two stages**:

1. **Identification** (always executed, read-only). Lists candidate resources discovered
   via the §6.1 labels / tags and a configurable age threshold (default 24 h). Output is
   a candidate report shown to the operator; nothing is modified.
2. **Cleanup / pruning execution** (requires **manual operator approval**). Once the
   operator confirms the candidate list, the script deletes the resources.

Implementation: `tools/cleaner.py` (script invoked from the laptop). Claude's role is to
**periodically remind the operator to run the cleanup** (e.g. via a scheduled wake-up
that surfaces the reminder); Claude does **not** execute the cleanup itself.

Resource classes the Cleaner identifies (and, on approval, prunes):

| Class | Discovery | Notes |
|---|---|---|
| K8s objects (Deployment, Service, Ingress, TLS Secret) | `kubectl get ... -l app.kubernetes.io/managed-by=inference-benchmarking` | Skip Model-cache PVCs (§6.6 keeps them intentionally). |
| SLURM scratch dirs under `/capstor/scratch/cscs/$USER/` | Match the run-ID pattern (§6.2); skip dirs owned by an active job. | |
| JFrog images tagged for benchmark runs | JFrog API filtered by the benchmark tag prefix (§6.1). | Skip the most recent N tags per repository. |

Cleaner actions are logged on the laptop but are not persisted to the per-run results DB.

---

## 7. System Performance Pre-checks

Synthetic micro-benchmarks executed in the engine's own container instance immediately
before the engine binary starts (mechanism in §7.2) — so the foundation measured is the
one the engine actually sits on. Without this gate, a degraded NCCL fabric or slow
weight storage would silently bias benchmark results: good throughput / latency numbers
measured on top of a degraded foundation are misleading.

### 7.1 Scope

Pre-checks cover the three planes whose performance directly bounds LLM serving:

| Plane | Benchmark | Validates |
|---|---|---|
| Collective communication | NCCL / RCCL `all_reduce`, `all_gather`, and `alltoall` (see `examples/nccl-tests/`), run at the engine's rank topology | Intra-node (NVLink / NVLink-C2C on GH200, Infinity Fabric on MI300A) and inter-node (Slingshot 11 on Alps) bandwidth in one pass. The three-collective set covers TP all-reduce, sequence-parallel / weight-gather, and MoE expert dispatch; add `reduce_scatter` / `sendrecv` / `broadcast` for PP. |
| GPU-initiated one-sided RMA (vendor SHMEM) | SHMEM perftest binaries shipped with the engine image — **NVSHMEM** on NVIDIA targets, **ROCm SHMEM** on AMD targets (e.g. `device/coll/alltoall_latency`, `device/pt-to-pt/shmem_put_bw`) — run at the engine's rank topology | Put / get bandwidth and one-sided all-to-all latency. Used by MoE engines that bypass NCCL / RCCL for expert dispatch (DeepEP and equivalents). Skipped with a warning if the engine image lacks the relevant SHMEM library; `shmem_required: true` to enforce. |
| Storage | Sequential read against the engine's model-weights mount (`capstor` / `iopsstor` on SLURM, Ceph PVC on K8s) | Read throughput as seen by the engine — **contextualises the model-loading times collected later** (§9.2), so a slow `model_load_weights_s` can be attributed to either storage or engine overhead. On Lustre also validates that `safetensors_load_strategy=prefetch` is taking effect (see §5.1). |

GPU memory bandwidth (`bandwidthTest`) and host DRAM bandwidth (STREAM) are intentionally
**not** in the pre-check suite — they are stable per-SKU characteristics that rarely
degrade in isolation without also degrading the NCCL / RCCL bandwidth above, so a separate
test adds maintenance without adding signal.

### 7.2 Execution

- Pre-checks run in the **same container instance** the LLM engine will run in — not a
  sibling container, not an init container. Pre-check + engine commands are **concatenated
  into one container invocation** on both targets, so the dynamic linker has resolved the
  exact same libfabric / CUDA / NCCL / OpenMPI build for the checks and the engine:
  - **SLURM**: `srun --environment=<engine-edf> bash -c "run_system_prechecks && exec <engine>"`.
    One container session per task, shared between the pre-check and the engine. Same
    `--mpi`, `--container-mounts`, `--env`, and NUMA / CPU bindings apply throughout.
  - **Kubernetes**: pre-launch command in the engine's own pod and container —
    `command: ["bash", "-c", "run_system_prechecks && exec <engine>"]`. Same engine container,
    same env vars, mounts, `securityContext`, and resource requests in effect during
    the checks.
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
    `/capstor/scratch/cscs/$USER/collective-tests-cache` on SLURM; a PVC mount on K8s).
    Shared across experiments; safe to delete to force a rebuild.

  The pre-check script **installs missing build tools** (`make`, `g++`, OpenMPI dev, `curl`,
  `tar`) inside the engine container via `apt-get` / `dnf` / `yum` rather than aborting —
  the engine image is not required to ship them. Only **rank 0** performs the install
  (other ranks wait on a sentinel), so apt is not hammered by `N` ranks concurrently.
- The NVSHMEM benchmark uses the perftest binaries that ship with the engine image's
  NVSHMEM SDK (no separate build). If NVSHMEM is absent the row is skipped with a warning;
  set `nvshmem_required: true` in the benchmark YAML to make absence a failure.
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
  against `tools/system_prechecks_reference.yaml` (§7.3), and **storing** one row per
  metric in the `system_prechecks` table (§13.6). Streaming a verbose log alongside is
  fine; the table is the structured source of truth.

### 7.3 Reference values

The grading rule in §7.4 (pass / warn / fail) compares each measurement against an entry
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
runs on each system. Until populated the gate (§7.4) is **unenforceable** and
measurements log as informational only.

### 7.4 Outcome and abort flow

For each pre-check metric:

- **pass** — measured within tolerance band. Recorded; sweep proceeds.
- **warn** — measured below tolerance. A warning is emitted; the benchmarker pauses and
  surfaces the discrepancy to the coordinator on the laptop. The operator chooses to abort
  the experiment or proceed. Non-interactive runs default to the value of
  `system_prechecks_on_warn` in the benchmark YAML (`abort` or `continue`; default `abort`).
- **fail** — measured well below tolerance (e.g. < 50% of expected) or the benchmark binary
  itself errored. The experiment is aborted by default; override with `system_prechecks_on_fail:
  continue` in the benchmark YAML.

All measurements (pass, warn, fail) are persisted in `system_prechecks` (§13.6) so that
later analysis can correlate a degraded foundation with anomalous LLM benchmark results.

### 7.5 Skipping pre-checks

Pre-checks add ~120 s per experiment. They can be disabled by setting
`skip_system_prechecks: true` in the benchmark YAML.

Use sparingly — only when the exact same hardware + image + runtime environment + topology
combination was validated within the same session. Any change to the image, env vars,
mounts, NUMA pinning, NCCL/RCCL settings, driver, or node set invalidates a prior pass
and the checks should be re-run.

---

## 8. Backends and models under test

This section enumerates the inference engines and the LLM models the benchmarker covers.
These are **stable system-under-test facts** — the operational reference for what the
framework can deploy and measure. The experiment-design surfaces (which feature to
compare, which BackendConfig knobs to sweep, which scenarios to use) live in §15.

### 8.1 Backends

The benchmark YAML's `backend:` field selects the inference engine. Each backend is
wired via its own Jinja2 EDF / K8s template (§9 *Inference Engine Bring-up*); its
sweepable configuration knobs live in §15.2.

| Backend | Planner template | Status |
|---|---|---|
| **vLLM** | `tools/templates/vllm.edf.j2` (TBD) | active |
| **SGLang** | `tools/templates/sglang.edf.j2` (TBD) | planned |
| **Nvidia Dynamo** | `tools/templates/dynamo.edf.j2` (TBD) | planned |

The specific backend version and variant (e.g. `vllm-cxi` v0.20.x vs v0.21.x, CSCS-fork
vs upstream vLLM) is declared **per experiment** in the benchmark YAML and sweeps as a
deployment-sweep dimension — comparing versions of one backend, or comparing across
backends, are both first-class experiment shapes.

#### Image lifecycle and registry

Every inference-engine image the framework tests is **built from sources checked into
this repository**. For each backend:

- The **Dockerfile** lives at `tools/images/<backend>/Dockerfile`.
- Any source-level **patches** applied at build time (e.g. the CXI / Slingshot-11
  integration patches for `vllm-cxi`) live under `tools/images/<backend>/patches/`,
  tracked in the repo with the same review discipline as the rest of the code.
- A **build-args metadata file** captures the upstream version pin, the patch list, the
  base image, and the canonical image tag the build produces.

The image **registry** is the **CSCS JFrog Artifactory**. Built images are pushed there
and referenced from the engine EDF / K8s manifest via their canonical tag.

**Build-when-needed.** Most experiments deploy a pre-built image straight from JFrog.
When an experiment requires changes to the image — a new patch, a new upstream pin, a
new backend variant — Claude carries the build through as part of the experiment-
preparation phase:

1. Updates the Dockerfile / patches under `tools/images/<backend>/`.
2. Submits the build (SLURM-based Docker build workflow per TODOs.md *Support building
   Docker images via SLURM jobs*, or the operator's local Docker).
3. Pushes the resulting image to JFrog with a canonical tag.
4. Updates the planner template's image reference to the new tag.

The full build provenance — Dockerfile commit SHA, patch revisions, base image, build
date, JFrog tag — is recorded per experiment alongside the BackendConfig so the exact
stack any experiment ran on is recoverable. The canonical JFrog path and other
shared cluster-side constants are deferred to a global configuration location — see
TODOs.md *Establish a global configuration location for shared values* and *Define and
configure JFrog folder/path for publishing built images*.

### 8.2 Models

The first-pass model set the benchmarker covers. Each entry pins the operational
information the Planner needs to render an engine launch (HuggingFace ID, tokenizer,
context length) plus the experiment-design information the operator needs to pick
scenarios and BackendConfig combinations (role, thinking mode, MoE structure).

The forward-looking model taxonomy (additional families to add later — DeepSeek-V4 Pro,
GLM-5.1, etc.) lives in the README; this section is the **operational subset** under
active measurement.

| Model | Role | HuggingFace ID | Tokenizer | Context | Thinking mode | MoE | Scenarios |
|---|---|---|---|---|---|---|---|
| **Apertus-70B** | target | `swiss-ai/Apertus-70B-Instruct-2509` | Apertus family (multilingual, 1000+ languages) | 65,536 tokens | No dedicated thinking mode (base model) | No | `long-context-followup`, `chat-short-turns` (with the caveat below on `thinking: true`). Excluded from `agentic-coding` — Apertus is not used by operators as a coding model. |
| **Apertus-8B**     | draft (same-family with Apertus-70B) | `swiss-ai/Apertus-8B-Instruct-2509` | Apertus family — **identical to the 70B**, tokenizer loaded once (§10.6) | 65,536 tokens | No | No | Always paired with Apertus-70B as the draft for speculative-decoding experiments |
| **Kimi-K2.6**      | target | `moonshotai/Kimi-K2.6` | Kimi family | 262,144 tokens (256K) | Yes — deeper reasoning and planning; strong on agentic, multi-step workflows | Yes (MoE; expert routing exercised by §15.1 *MoE expert routing* row) | `agentic-coding`, `chat-short-turns`, `long-context-followup`. `thinking: true` scenarios are most representative on Kimi-K2.6 because the widened output distribution matches the model's actual think+answer behaviour. |
| **DeepSeek-V4-Pro** | target | `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek custom (`encoding_dsv4`; `<think>` / `</think>` reasoning delimiters) | 1,048,576 tokens (1M) | Yes — three modes: *Non-think* / *Think High* / *Think Max*, toggled via the `thinking_mode` runtime parameter (to be exposed as a BackendConfig knob when DeepSeek experiments are wired) | Yes (MoE; 1.6 T total parameters / 49 B activated; expert routing exercised by §15.1 *MoE expert routing* row) | `agentic-coding`, `chat-short-turns`, `long-context-followup`. `thinking: true` scenarios match the model's native thinking modes directly. *Think Max* requires context ≥ 384 K — align `max_model_len` accordingly when sweeping it. License: MIT. Recommended sampling: `temperature=1.0`, `top_p=1.0`. Precision: FP4 (MoE expert params) + FP8 (other params) mixed. |

**Notes on scenario / model pairing:**

- A scenario's `thinking: true` flag is a **workload declaration**, not a model-capability
  query. The dataset generator widens output sampling regardless of whether the model
  has a dedicated thinking mode (§10.6). When the model under test is **not** a thinking
  model (e.g. Apertus-70B), `thinking: true` simulates "what happens if this model
  were forced to emit thinking-length outputs" — a useful stress-test of decode capacity,
  but the manifest's `not_modelled` should disclose that the model's own emissions would
  ordinarily be shorter.
- Speculative-decoding experiments require a draft/target pair from this table. Only
  the **Apertus-8B → Apertus-70B** pairing is in scope for v1 (same-family, identical
  tokenizer). Kimi-K2.6 and DeepSeek-V4-Pro have no in-scope draft; cross-family
  pairings (e.g. a smaller open model as draft for an MoE target) are deferred until
  acceptance-rate baselines are characterised.
- Tokenizer-loading consequences for these pairings are documented in §10.6.
- Adding a new model to this set is a planner-template + benchmark-YAML change; no spec
  edit is required unless the model introduces a new capability dimension (e.g. native
  thinking mode toggle, new MoE topology) that the framework should sweep over.

---

## 9. Inference Engine Bring-up

Once the System Performance Pre-checks (§7) pass, the Benchmarker hands control to the
inference engine in the same container session. This section covers the full bring-up:
how the launch command is constructed, how readiness is tracked, how model-loading time
is decomposed for diagnostics, and how the Inductor JIT-compile cost is paid up front
via a priming request — so that the sweep that follows measures steady-state behaviour.

The engine launch command is rendered by the Planner (§4) from:

- the **backend choice** in the benchmark YAML (`backend: vllm | sglang | dynamo`),
- the **BackendConfig** knobs (§15.2) — varied across deployments within an experiment (one deployment per combination; λ is then swept inside each deployment, see §11),
- backend-specific Jinja2 templates checked into `tools/` (one EDF template + one
  K8s manifest template per backend, per §1's *Backend-agnostic* principle).

The rendered command is concatenated with `run_system_prechecks && exec <engine>` (§7.2),
so pre-checks and the engine launch share the same container session with identical
libfabric / CUDA / NCCL / mounts / NUMA.

Backend-version compatibility of individual flags (which were removed, renamed, or made
mandatory in which release) is captured alongside the configuration surface in §15.2.

### 9.1 Health-check timeout

The benchmarker health check must wait at least as long as `server_ready_timeout_s`
(default 3600 s) before giving up — long enough to cover dual model load + CUDA graph
capture for speculative-decoding configurations, which can exceed 15 min.

### 9.2 Model loading time tracking

The Benchmarker records both the **total** time-to-ready and its **individual components**
(weight load, graph capture, compilation, …) so that optimisation efforts have a
per-component baseline to compare against and measure progress over. Components that a
given backend cannot expose are stored as `NULL` rather than collapsed into another bucket.

The schema lives in §13.2 (the `instances` table). Each `model_load_*` column is parsed
from the backend's structured logs or runtime API (per backend; for vLLM specifics see
§15.2).

A single experiment may deploy **multiple instances** of the same configuration (routing
tests, replica-sizing studies). The per-component breakdown is recorded **per instance**
so each instance's component times are visible individually. Reports (§14) show both the
per-instance breakdown and the totals aggregated across instances.

### 9.3 Inductor pre-compilation primer

vLLM v1 (vllm-cxi v0.20+) uses `torch.inductor` to JIT-compile CUDA kernels for large prefill
sequences (> 512 tokens) **lazily** — on the first request that triggers the path. This
one-time compilation is the dominant cold-start cost on first request after a server start.

**Requirements:**

- The benchmarker must send a **priming request** (20K-token prompt, `max_tokens=1`) to the
  engine's HTTP endpoint before the sweep begins, and wait up to 300 s for it to complete.
- After the primer completes, the first measurement request should exhibit genuine
  steady-state TTFT (not the cold compile delay). If it does not, **warn the operator**
  that the primer missed its target.
- When the backend supports persisting compilation artifacts across restarts, the path is
  exposed as a `BackendConfig` field (§15.2); when it does not, the primer simply runs on
  every fresh start. Exploring this loading-time optimisation is tracked in `TODOs.md`.

---


## 10. Prompt Generation

### 10.1 Location, ownership, persistence

Prompts are produced on the **Benchmarker** SLURM allocation by its **dataset-generator**
subcomponent, from `dataset_config` in the benchmark YAML, sequentially before the
load-generator phase starts (per §1's separation of concerns). Generating on the cluster
sidesteps the FirecREST 5 MB direct-upload limit and allows arbitrarily large prompt
pools — the coordinator never ships prompt data to the cluster.

**Persistence.** Generated artefacts live on the Benchmarker's capstor scratch directory
for the duration of the run, reused across every deployment and every rate-level sweep
step, and reclaimed by the §6 teardown. They are **not** copied into `experiments/<run>/`:
the manifest (persisted on `experiments.scenario_manifest`), the master seed (in
`dataset_config`), and the scenario-registry revision (recorded alongside) are sufficient
to regenerate. The full reproducibility contract lives in §10.8.

**Source-failure semantics.** A failure of any dataset source (LongBench download error,
HuggingFace unreachable, reasoning-trace dataset unavailable, etc.) **aborts the run**
with a clear error. There is no silent fallback to synthetic data — this avoids the trap
where, e.g., a speculative-decoding experiment silently degrades to filler text and
reports ~0% acceptance as if it were a property of the model.

### 10.2 Key concepts

Three artefacts cooperate to produce a benchmark dataset:

- **Scenario registry** (`tools/scenarios/<slug>.yaml`) — the canonical declaration of
  what each scenario is: its **source** (§10.5), **length distributions** (§10.6),
  **multi-turn structure** and **session mode** (§10.7), and the **`modelled` /
  `not_modelled`** lists — human-authored statements describing what the scenario
  explicitly represents and what it deliberately does not, so any reader of a report
  knows in what context the numbers should be interpreted; copied verbatim into the
  scenario manifest (§10.8, §13.7).
- **`dataset_config`** in the benchmark YAML — the per-run knobs: which scenario to run,
  the master seed, `num_prompts`, and any per-run overrides to registry defaults.
- **Dataset generator** — reads both, materializes the prompt pool on capstor scratch,
  and emits the scenario manifest as a structured side-effect for the experiment row
  (§13.1).

The registry is data, not code: adding a new scenario does not require editing the
generator. Registry entries are versioned with the repo; changes to a scenario's
`modelled` / `not_modelled` lists are reviewable in PRs.

### 10.3 Scenario registry

Each registry entry is a YAML file with the schema below. Fields not relevant to a given
scenario are omitted (e.g. `session.think_time_ms` only applies in `sequential` mode).

| Field | Notes |
|---|---|
| `name` | Slug, matches the filename. Used as `experiments.scenario`. |
| `summary` | One-line human description. |
| `maturity` | `established` \| `emerging` \| `exploratory`. |
| `source` | `kind` (§10.5) + per-source `config`. |
| `input_length` | `distribution` (`lognormal` \| `normal` \| `fixed`) + `params` (§10.6). |
| `output_length` | Same shape as `input_length`. Widened when `thinking: true` (§10.6). |
| `thinking` | Optional boolean; widens output sampling per §10.6. Default `false`. |
| `session.mode` | `open_loop` \| `sequential` (§10.7). |
| `session.turns_per_session` | Distribution (same shapes as `input_length`). |
| `session.prefix_strategy` | `append_delta` (only supported value; §10.7). |
| `session.think_time_ms` | Distribution; `sequential` mode only (§10.7). |
| `manifest.modelled` | Human-authored list of what the scenario explicitly represents. |
| `manifest.not_modelled` | Human-authored list of what the scenario deliberately omits. |

See `tools/scenarios/agentic-coding.yaml` for a worked example. `assumptions` is **not**
stored in the registry — it is computed at runtime from the actual `dataset_config`
consumed (§10.8).

### 10.4 dataset_config schema

The benchmark YAML's `dataset_config` block:

| Field | Type | Required | Notes |
|---|---|---|---|
| `scenario` | string | yes | Must match a registered scenario name. |
| `num_prompts` | integer | yes | Size of the generated prompt pool. The load generator's request stream draws from this pool — must be set sufficiently large that prompts are not exhausted before the experiment's request budget is consumed (accounting for `turns_per_session` in multi-turn scenarios), so that prompt uniqueness (§10.6) holds for every request actually issued. |
| `seed` | integer | yes | Master seed; sub-seeds derived deterministically (§10.8). |
| `input_length` | object | no | Per-run override of the registry's `input_length` distribution. |
| `output_length` | object | no | Per-run override of the registry's `output_length` distribution. |
| `session` | object | no | Per-run override of session fields. |
| `tokenizer_id` | string | no | Override the tokenizer (defaults to the target model's; see §10.6). |
| `source_overrides` | object | no | Source-specific overrides (e.g. LongBench task subset). |

Any field absent from `dataset_config` is inherited from the scenario registry.

### 10.5 Dataset sources

The `source.kind` enum, with v1 scope:

| `kind` | What it is | `source.config` |
|---|---|---|
| `synthetic` | Filler text with unique `[prompt-NNNNNN]` headers. No network required. | — |
| `longbench` | LongBench code tasks downloaded from HuggingFace. | `tasks: [lcc, repobench-p, …]` |
| `reasoning_trace_replay` | Recorded reasoning traces (GSM8K-with-cot, MATH, AIME, R1-distill). Output length comes from the recorded target and **overrides** `output_length`. | `dataset: <name>` |
| `wildchat` | Real user↔assistant conversations from `allenai/WildChat-1M`. Multi-turn (median ~3, long tail), multilingual. Conversation turn boundaries drive the session structure; per-turn lengths are clamped to the scenario's distributions. | `languages: [en, …]`, `min_turns: N` |

Per-source suitability for the optimisations being measured (speculative-decoding
acceptance, prefix-cache hit-rates) is summarised in §10.9. Licenses: LongBench (MIT),
WildChat (ODC-BY), reasoning-trace datasets (per-dataset, all permissive for
benchmarking).

**v1 is text-only.** Multimodality (image first, then audio and video) is the next
feature on the dataset-generator roadmap — see `TODOs.md`. A scenario whose registry
entry declares any `modalities` other than `[text]` is rejected at registry-load time
until support lands.

### 10.6 Per-request mechanics

**Prompt uniqueness.** Every prompt must start with a distinct token block so the
engine's prefix cache does not serve synthetic cache hits — which would collapse TTFT to
~100 ms regardless of load, an artefact rather than real performance. Single-turn prompts
begin with a unique `[prompt-NNNNNN]` header; multi-turn sessions begin with a unique
`[session-NNNNNN]` header reused across the session's turns, so the prefix cache *does*
hit on the shared session prefix — the locality the benchmark is meant to expose (§10.7).

**Length distributions.** `input_length` shape is per-scenario, declared in the registry
(§10.3). Supported: `lognormal` (truncated), `normal` (truncated), `fixed`. Heavy-tailed
`lognormal` matches observed LLM-workload distributions and is the recommended default;
`fixed` is for isolation studies.

**Output length control.** Each prompt carries a target `max_tokens` sampled from
`output_length`. The load generator sends `max_tokens=<sampled>` **and** `ignore_eos=True`,
forcing the model to emit exactly that many decode tokens. This makes decode cost
reproducible across runs and across models — measured TPOT and `output_tokens` no longer
depend on per-model stopping behaviour. Sources that carry ground-truth output lengths
(`reasoning_trace_replay`) override the sampled value with the recorded target.

**Thinking models (v1 approximation).** When a scenario sets top-level `thinking: true`,
the generator widens `output_length` sampling to approximate the combined
think-trace-plus-answer length: `params.mean × 2.5`, `params.sigma` (or `stdev`) `× 1.5`;
`fixed` values multiplied by 2.5. `params.min` / `params.max` are preserved as clamps
(not rescaled). A precise bimodal sampler (tiny direct answers vs long deep-thinking
outputs) is deferred — see TODOs.md *Bimodal output distribution as first-class*. The
flag's effect is recorded in the manifest's `modelled` list and the simplification
disclosed in `not_modelled`.

**Tokenization.** Length filtering, length-distribution sampling, and the `input_tokens`
field all use the **target model's tokenizer**, loaded by HuggingFace ID on the
Benchmarker at dataset-generation time. Changing the target model invalidates the dataset
and triggers regeneration. For same-family draft/target pairs (e.g. Apertus-8B +
Apertus-70B — the in-scope pairing for v1, see §8.2) tokenizers are identical and
only one is loaded. For cross-family pairs the **target's** tokenizer is authoritative;
any draft-tokenizer mismatch is logged but does not block the run.

### 10.7 Sessions and agentic approximation

**Multi-turn structure.** Multi-turn scenarios produce N turns per session, with N
sampled from `session.turns_per_session` (§10.3). Each session is assigned a stable
integer `session_idx ∈ [0, M-1]`; every turn carries the same `session_idx` (exposed to
the load generator for session-affinity routing — §11.4), and the `[session-NNNNNN]`
header from §10.6 encodes the same identifier in the prompt text. Single-turn scenarios
are the degenerate case M = `num_prompts`, `session_idx = prompt_idx`.

**Prefix strategy** is always `append_delta`: turn K+1's prompt = full prior transcript +
new user turn. The engine's prefix cache reuses the shared prefix naturally — exactly as
real chat / agentic clients do. (A `regenerate` strategy was considered but rejected: it
defeats the prefix cache and is better expressed as a separate ablation by disabling
prefix caching at the backend.)

**Session mode** governs how follow-up turns interact with the load generator's open-loop
arrival process (§11.3):

| `session.mode` | Follow-up behaviour | Use for |
|---|---|---|
| `open_loop` (default) | All turns scheduled by the arrival process; turn K+1 fires on its own schedule regardless of when turn K completed. Preserves open-loop queueing semantics throughout. | RAG-style queries against a shared long-lived prefix; reasoning workloads; any scenario where turn ordering is incidental. |
| `sequential` | Turn K+1 is sent only after turn K's response, plus a `think_time_ms` delay. Closed-loop coupling *within a session*; cross-session arrivals remain open-loop (session starts are still Poisson per §11.3). | Conversational chat; agentic-coding follow-ups; any scenario where a follow-up cannot meaningfully precede its predecessor's response. |

**Agentic approximation (v1).** Agentic workloads — a user prompt fanning out to many
model invocations (think → tool call → tool result → …) — are approximated as **multi-turn
sessions with bursty fan-out**: each session = one agentic task; each turn = one model
invocation; tool results synthesised as injected text in the next turn's prompt. No tool
catalog, no fan-out DSL, no per-tool JSON schemas. This is enough to derive supportable
agentic-user count from λ + an SLO. The precise mechanism (distinct `think` / `tool_call`
/ `tool_result` roles, per-tool schemas, schema-constrained-decoding validity, a
dedicated `agent_tasks` table, first-class bimodal output, per-tool result-content
synthesis) is deferred to TODOs.md *Precise agentic / tool-calling measurement*. Routing:
`session_affinity` (§11.4) is the natural choice — every turn of a task lands on the same
instance so the prefix cache exposes the locality the workload depends on.

### 10.8 Reproducibility surface

**Seeding.** `dataset_config.seed` is a single integer. Per-axis sub-seeds derived as
`blake2b(f"{seed}:{axis}", digest_size=8)` over axes: `header`, `length_input`,
`length_output`, `selection`, `turns`, `thinktime`.

**Contract.** Same `dataset_config` + same scenario-registry revision + same target
tokenizer → identical prompt pool (byte-for-byte). Changing the target model triggers
regeneration (different tokenizer → different length filtering and different sampled
lengths).

**Manifest.** The dataset generator emits the `scenario_manifest` (schema in §13.7) as a
side-effect of running:

| Manifest field(s) | Source |
|---|---|
| `name`, `summary`, `maturity`, `modelled`, `not_modelled` | Copied verbatim from the scenario registry entry. |
| `assumptions` | Auto-filled from the `dataset_config` actually consumed: input / output length distributions; turns-per-session distribution; session mode; prefix strategy; source `kind` + relevant source config; master seed; tokenizer ID. |

Together these are sufficient to reconstruct what the run measured without re-reading the
registry at a specific revision.

### 10.9 Notes on dataset suitability

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


## 11. Load Generation

### 11.1 Server readiness and model-loading tracking

Before the sweep starts, the load generator must, for **each** deployed instance:

- Wait for `/health 200`. Per-instance wait bounded by `server_ready_timeout_s` (default 3600 s;
  see §9.1). If any instance fails to come ready within the timeout, the experiment aborts.
- Parse the per-instance model-loading breakdown from the backend's structured logs / runtime
  API and persist one row per instance into the `instances` table (§13.2) with the
  `model_load_*` fields populated (§9.2).
- Run the inductor pre-compilation primer (§9.3).

The sweep begins only once **all** instances are ready, profiled, and primed.

### 11.2 Sweep structure

- **Warmup phase**: requests sent but metrics excluded. Long enough for:
  - Inductor JIT compilation to complete after primer (≥ 1 full round of compilation per model)
  - KV cache and queue to reach steady state
- **Measurement phase**: TTFT, ITL, E2E recorded per request.
- **Drain phase**: in-flight requests after measurement window are allowed to complete up to
  `drain_timeout_s`.
- `request_timeout_s`: client-side TTFT hard cutoff; exceeded requests recorded as `success=0`.

### 11.3 Open-loop stochastic arrivals

The load generator supports **configurable arrival processes**, selected per sweep step via
`arrival_process` in the benchmark YAML. The chosen process and its parameters are serialized
into `experiments.scenario_manifest.assumptions` (§13.7) so the conditions a result was
measured under are always recoverable.

| `arrival_process` | Description and intuition |
|---|---|
| `poisson` (default) | **Mathematical**: memoryless Poisson at rate λ; inter-arrivals drawn from `Exp(1/λ)`. **Intuition**: a sterile lab baseline where every request is independent of every other — picture users hitting "send" at random moments with no clustering, no batching, no feedback between them. Useful as the cleanest characterisation of pure queue behaviour, but produces optimistic tail latencies because real traffic always clusters more than this. |
| `burst_mmpp` | **Mathematical**: two-state on/off Markov-Modulated Poisson Process at mean rate λ, with configurable burst factor (peak-to-mean ratio) and mean burst / idle durations. **Intuition**: traffic alternates between a high-arrival burst phase and a quiet phase, modelling production patterns where a tool-calling cycle fans out several requests at once, batch-API submissions flush queues at intervals, or cohorts of users arrive in waves. Reveals tail-latency amplification and queueing collapse that Poisson hides because real traffic's coefficient of variation is well above Poisson's 1. |

A heavy-tailed (Pareto) arrival process is intentionally out of scope for v1 — tracked in
`TODOs.md`.

Each arriving request is routed to one of N server instances per `routing_strategy` (§11.4).

### 11.4 Routing strategies

- `random` (default): uniformly random instance selection per request.
- `session_affinity`: `session_idx % N` — same-session prompts always route to the same
  instance. Enables meaningful prefix-cache benefit across multi-turn sessions. Useful to
  measure the effect of session affinity vs random routing in multi-instance deployments.

---

## 12. Measurement

### 12.1 Request error tracking

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

### 12.2 Per-session derived metrics

Every multi-turn scenario — conversational chat, long-context follow-ups, the v1 agentic
approximation, … — produces sessions whose turns share a `session_idx`. Session-level
metrics are derived at report time by grouping the `requests` table (§13.3) on that key:

| Per-session metric | Derivation |
|---|---|
| `session_e2e_ms` | Time from the first turn's `send` to the last turn's last token (max of per-request E2E within the session) |
| `session_turns` | Number of requests with the same `session_idx` |
| `session_input_tokens` | `SUM(input_tokens)` per `session_idx` |
| `session_output_tokens` | `SUM(output_tokens)` per `session_idx` |
| `session_success` | `MIN(success)` per `session_idx` — the session succeeds only if every turn within it succeeded |

Single-turn scenarios are the degenerate case (session = single request, §10.7), where
these metrics collapse to the underlying per-request values.

For **agentic scenarios** under the v1 approximation (per §10.7, where one task ≡ one
session), these session metrics also serve as task-level metrics — `session_e2e_ms` is
the task latency, `session_turns` is the fan-out depth, etc. A dedicated `agent_tasks`
table that carries truly task-specific signals (tool calls emitted, schema validity per
tool call, task identity distinct from session identity, possibly multiple tasks per
session) is deferred — see `TODOs.md` *Precise agentic / tool-calling measurement*.

### 12.3 Hardware utilization sampling

To detect untapped hardware headroom (per §1 *Observed execution*), the benchmarker
samples host-side telemetry on every inference-server node for the full duration of every
sweep step (warmup + measurement + drain). Samples are stored in the `hardware_stats`
table (§13.5).

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


## 13. Results

Per-run results live in a SQLite database file (`run_<id>.db`) with six tables:
`experiments` (one row per sweep), `instances` (one row per deployed server instance),
`requests` (one row per issued request), `server_stats` (periodic samples of
server-side counters), `hardware_stats` (periodic samples of host hardware telemetry),
and `system_prechecks` (one row per pre-check metric). A first-class `agent_tasks`
table is deferred (see `TODOs.md` *Precise agentic / tool-calling measurement*); v1
derives per-task agentic metrics by grouping `requests` on `session_idx` (§12.2).

### 13.1 `experiments` table

One row per sweep — the configuration and overall outcome of the run.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, PK | Unique identifier (`timestamp + model_slug + backend + deployment + 4-hex random`; see §6.2). |
| `model` | TEXT | Model identifier (HuggingFace ID or path). |
| `backend` | TEXT | Inference engine (`vllm`, `sglang`, `dynamo`). |
| `backend_config` | TEXT (JSON) | Serialized `BackendConfig` — all fields from §15.2. |
| `dataset_config` | TEXT (JSON) | Serialized dataset configuration (§10). |
| `scenario` | TEXT | Scenario slug (e.g. `agentic-coding`, `chat-short-turns`, `long-context-followup`). See §13.7. |
| `scenario_manifest` | TEXT (JSON) | Structured disclosure of what the scenario models, what it omits, and the numeric assumptions baked in. See §13.7. |
| `rate_levels` | TEXT (JSON) | List of λ values (req/s) swept in this run. |
| `warmup_s` | INTEGER | Warmup phase duration in seconds (metrics excluded; see §11.2). |
| `measurement_s` | INTEGER | Measurement phase duration in seconds. |
| `created_at` | TEXT (ISO 8601) | Experiment start timestamp. |

### 13.2 `instances` table

One row per deployed server instance for the experiment. A single experiment may deploy
multiple instances of the same configuration (routing tests, disaggregation studies, multi-
replica deployments); each instance has its own load profile (§9.2).

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Per-experiment instance identifier (stable across the run). Composite PK with `run_id`. |
| `endpoint` | TEXT | URL the load generator targets for this instance (`host:port`). |
| `node` | TEXT | Hosting node — SLURM node name or K8s pod / node-type. `NULL` if not applicable. |
| `model_load_total_s` | REAL | Total time-to-ready for this instance (§9.2). |
| `model_load_weights_s` | REAL | Weights load subcomponent (§9.2). |
| `model_load_engine_init_s` | REAL | Engine/runtime startup subcomponent (§9.2). |
| `model_load_cuda_graph_capture_s` | REAL | CUDA graph capture subcomponent (§9.2). |
| `model_load_inductor_compile_s` | REAL | Inductor compilation primer subcomponent (§9.2). |

Loading-time components a backend cannot expose are stored `NULL` (see §9.2).

### 13.3 `requests` table

One row per issued request — the per-request latency record.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `rate_lambda` | REAL | λ value (req/s) of the sweep step this request belongs to. |
| `request_id` | INTEGER | Per-rate-level request index (monotonic). |
| `session_idx` | INTEGER | Session this request belongs to (§10.7). Shared by every turn of the session; enables grouping per-session for session-affinity routing analysis (§11.4). For single-turn scenarios equals the request's underlying prompt index. |
| `turn_idx` | INTEGER | 0-based position of this request within its session (§10.7). `0` for the first turn (and for every request in single-turn scenarios); `1` for the first follow-up; etc. Lets reports plot per-turn metrics directly (e.g. "TTFT vs turn index" to visualise prefix-cache benefit on follow-up turns) without reconstructing the order from timestamps. |
| `ttft_ms` | REAL | Time to first token, milliseconds — authoritative SLO metric. |
| `tpot_ms` | REAL | Inter-token latency, mean across the request's output tokens. |
| `e2e_ms` | REAL | End-to-end request time, milliseconds. |
| `input_tokens` | INTEGER | Number of input tokens. |
| `output_tokens` | INTEGER | Number of generated output tokens. |
| `success` | INTEGER | `1` if completed within timeouts; `0` if client-side `request_timeout_s` exceeded or the server returned an error. |
| `error` | TEXT | Error message or class when `success=0`; `NULL` otherwise. |

### 13.4 `server_stats` table

Periodic samples of server-side counters during a sweep step. Sampling cadence is
backend-dependent. Samples are scraped **per instance** — for multi-instance deployments
(routing-strategy or replica-sizing experiments, §11.4 + §13.2), each instance produces
its own row stream so saturation on one instance is distinguishable from idleness on
another.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Instance the sample was scraped from. Composite key with `run_id` + `ts` + `rate_lambda`. Matches §13.2 / §13.5 / §13.6. |
| `rate_lambda` | REAL | λ value (req/s) of the sweep step being sampled. |
| `ts` | TEXT (ISO 8601) | Sample timestamp. |
| `requests_running` | INTEGER | Requests currently executing on the server. |
| `requests_waiting` | INTEGER | Requests queued on the server. |
| `gpu_cache_pct` | REAL | KV cache utilization, percent. |
| `spec_accept_rate` | REAL | Speculative-decoding token acceptance rate; `NULL` if speculative decoding disabled. |

### 13.5 `hardware_stats` table

Periodic samples of host-side hardware telemetry on the inference-server node(s) during a
sweep step. See §12.3 for sampling cadence and per-signal meaning. GPU-scoped rows carry
a non-`NULL` `gpu_index`; node-scoped rows carry `gpu_index = NULL`.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Instance whose host this sample belongs to. |
| `rate_lambda` | REAL | λ value (req/s) of the sweep step being sampled. |
| `ts` | TEXT (ISO 8601) | Sample timestamp. |
| `gpu_index` | INTEGER | GPU device index for GPU rows; `NULL` for node-wide rows. |
| `gpu_util_pct` | REAL | Coarse GPU activity (§12.3). |
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

One row per pre-check metric per instance (see §7). Used both for warning the operator about
a degraded foundation and for later correlation with anomalous sweep results.

| Column | Type | Semantic |
|---|---|---|
| `run_id` | TEXT, FK | Foreign key to `experiments.run_id`. |
| `instance_id` | TEXT | Instance the check ran against. |
| `metric` | TEXT | Metric identifier (e.g. `nccl_allreduce_16MiB_GBs`). |
| `measured` | REAL | Measured value. |
| `expected` | REAL | Expected value from reference table (§7.3); `NULL` if no reference. |
| `tolerance_pct` | REAL | Negative deviation tolerance (e.g. `-10`); `NULL` if no reference. |
| `status` | TEXT | `pass`, `warn`, or `fail` (§7.4). |
| `ts` | TEXT (ISO 8601) | Time the check completed. |

### 13.7 Scenario manifest

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

Validation: the Coordinator aborts **before submission** if the benchmark YAML's `scenario`
field is missing or names an unregistered scenario (§10.3). The dataset generator on the
Benchmarker aborts **before load-generation begins** if any required field of the emitted
`scenario_manifest` is missing or fails schema validation (matching §10.8). There is no
implicit default — every benchmark must declare what it is and is not.

### 13.8 Experiment directories

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
  `experiments.scenario_manifest`, §13.7): scenario name, one-line summary, the
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
  `model_load_inductor_compile_s`) — see §9.2.
- TTFT p50/p95/p99 vs λ plot (log scale) with SLO line
- ITL p50/p95/p99 vs λ plot
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
`experiments/YYYY-MM-DD_description/` folder (§13.8).

### 14.3 Curated reports (`reports/`)

The per-experiment notebooks in §14.1–§14.2 exist for **reproducibility**: every plot the
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

## 15. Experiment Plans

Each experiment is composed from the features being measured (§15.1), the **deployment
target** (SLURM vs Kubernetes — frequently a sweep dimension in its own right, §5), the
**backend and its version / variant** under test (§8.1; v0.20.x vs v0.21.x, or
CSCS-fork vs upstream — comparing two versions of the same backend is a first-class
experiment shape), the BackendConfig knobs that vary across deployments within the
experiment (§15.2), and the model(s) under test (§8.2). The benchmark YAML specifies
all five. An experiment thus has two nested sweeps: the **deployment sweep** over
deployment-target × backend-version × BackendConfig × model combinations (one engine
launch per combination), and inside each deployment the **rate-level sweep** over λ
values (each λ being one "sweep step" in the sense used by the `requests` /
`server_stats` / `hardware_stats` tables).

The lists in §15.1 and §15.2 are **deliberately non-exhaustive** — they capture the v1
priorities so the implementation phase has concrete context to build against, but
experiments routinely introduce additional features, modes of usage, or backend knobs on
a per-experiment basis. Claude adapts the framework to support each addition as the
operator requests it: extending the BackendConfig surface, the planner templates, the
manifest disclosure surface, and the report panels in lock-step.

### 15.1 Features under test

The framework benchmarks the inference-serving features listed below. The list is
**non-exhaustive** — it captures the v1 priorities to give the implementation phase
concrete context, not a closed set. Experiments routinely require additional features or
modes of usage; new entries are added by extending this list, defining any new config
knobs in §15.2, binding to at least one scenario, and surfacing the marginal effect in
reports (§14). Each feature is exercised by one or more scenarios (see the scenario
taxonomy in the README).

| Feature | Why it matters | Where configured | Procurement implication |
|---|---|---|---|
| **Automatic prefix caching** | Reduces TTFT for sessions sharing prompt prefixes; critical for chat and AI-assisted coding. | `enable_prefix_caching` (§15.2) | Cache-friendly KV memory hierarchy; cache hit-rate as a procurement metric. |
| **KV-cache offloading** | Extends effective KV capacity by spilling to host DRAM / unified memory; trades per-request latency for concurrency. | `kv_offloading_size`, `kv_offloading_backend` (§15.2) | **Memory-layer sizing decisions** — HBM vs Grace-DRAM vs CXL. Offloading bandwidth profiles drive host-DRAM-per-GPU sizing and the choice of unified-memory / CXL fabrics for next-generation systems. |
| **KV-cache reuse across requests** | Identical or partially-overlapping prefixes from different requests reuse already-computed KV; effectiveness depends on routing. | `enable_prefix_caching` (§15.2) + `routing_strategy` (§11.4) | KV memory pressure under realistic locality; informs replica-pool sizing. |
| **Speculative decoding** | Improves decode throughput when a smaller draft model proposes tokens accepted by the target. | `speculative_decoding.*` (§15.2) | Compute headroom for draft model; memory budget for two-model deployments. |
| **Continuous batching** | Schedules new requests into running batches without waiting for current ones to finish — the dominant throughput optimization for online serving. | Backend default; not directly exposed | Scheduler responsiveness characterisation; admission-control budget. |
| **MoE expert routing** | Token-to-expert dispatch and load balance govern memory pressure and inter-GPU traffic. | Observed via §12.3 telemetry (NVLink / PCIe all-to-all signals) and §7 NVSHMEM perftest; not configured at the framework layer. | Interconnect sizing for all-to-all expert traffic; hot-expert memory pressure. |
| **Quantization (weights / KV / activation)** | Trades model fidelity and memory footprint against throughput. | `kv_cache_dtype` (§15.2); weight quantization via backend | Memory hierarchy: lower-precision math support vs higher-precision storage. |
| **Disaggregated prefill / decode** | Splits compute-heavy prefill from memory-bandwidth-heavy decode across different accelerator classes. | Per-component `nodeSelector` (§5.2); KV-transfer mechanism deferred (TODOs.md *NIXL disaggregated prefill/decode*). | Heterogeneous accelerator procurement; interconnect bandwidth between roles. |
| **Multi-replica routing and session affinity** | Distributes load across replicas; `session_affinity` preserves prefix-cache hits at the cost of fairness. | `routing_strategy` (§11.4) | Ingress / load-balancer requirements; cache-locality vs replica-fairness trade-off. |

Each feature's contribution to latency, throughput, error rate, and hardware utilisation
(§12.3) is recorded per sweep. Reports plot the marginal effect of enabling / disabling
individual features so procurement evidence can isolate the value of each.

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


### 15.2 Sweepable backend configuration

All fields are optional (sensible defaults apply); the experiment's **deployment sweep**
iterates over combinations of them, instantiating one engine deployment per combination
(λ is then swept inside each deployment — see §15 intro). Values flow from `BackendConfig`
through Coordinator → Planner (§4) → backend-specific Jinja2 templates (§9) into the
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

#### vLLM — `vllm-cxi` v0.20.x

| Field | vLLM flag | Notes |
|---|---|---|
| `tensor_parallel_size` | `--tensor-parallel-size` | Default 1. Must not exceed `gpus_per_node` (= 4 on Alps) — per-layer all-reduce is bandwidth-heavy; see §5.1. |
| `pipeline_parallel_size` | `--pipeline-parallel-size` | Default 1. Use for cross-node scale-out — PP traffic is much lighter than TP. |
| `data_parallel_size` | `--data-parallel-size` | Default 1. Each DP replica is an independent `instances` row (§13.2). |
| `expert_parallel_size` | `--expert-parallel-size` | Default 1. MoE engines only. |
| `max_model_len` | `--max-model-len` | |
| `max_num_batched_tokens` | `--max-num-batched-tokens` | Must equal `max_model_len` for long-context (avoids chunked-prefill rejection) |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | |
| `kv_cache_dtype` | `--kv-cache-dtype` | e.g. `"fp8"`. Doubles KV capacity but worsens per-request latency due to higher batch concurrency. |
| `enable_prefix_caching` | `--enable-prefix-caching` | Default True. Set False to isolate TTFT from cache artefacts (but prefer unique prompts instead). |
| `safetensors_load_strategy` | `--safetensors-load-strategy` | `"prefetch"` recommended on Lustre (capstor / iopsstor) — see §5.1. |
| `kv_offloading_size` | `--kv-offloading-size` | Total GiB across all TP ranks (e.g. `400` = 100 GiB/GPU for TP=4). Uses GH200 Grace DRAM at 900 GB/s via NVLink-C2C. |
| `kv_offloading_backend` | `--kv-offloading-backend` | `"native"` (default). |
| `speculative_decoding.draft_model` | part of `--speculative-config` JSON | Draft model identifier (HuggingFace ID or path). See the vLLM compatibility notes below. |
| `speculative_decoding.num_speculative_tokens` | part of `--speculative-config` JSON | |
| `speculative_decoding.draft_tensor_parallel_size` | part of `--speculative-config` JSON | Draft tensor-parallel size. Shared- vs dedicated-GPU guidance in §16.2. |

**Flag compatibility.** The vLLM flag names above are stable on the current vllm-cxi pin.
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
`tools/templates/sglang.edf.j2` (TBD when the first SGLang experiment is wired).

| Field | SGLang flag | Notes |
|---|---|---|
| *TBD* | *TBD* | First SGLang experiment will seed the table; Claude adds rows as knobs are exercised. |

#### Dynamo

*Populated as Nvidia Dynamo experiments come into scope.* The same conceptual
configuration surface as vLLM, expressed through Dynamo's flags. Planner template:
`tools/templates/dynamo.edf.j2` (TBD when the first Dynamo experiment is wired).

| Field | Dynamo flag | Notes |
|---|---|---|
| *TBD* | *TBD* | First Dynamo experiment will seed the table; Claude adds rows as knobs are exercised. |


---

## 16. Findings Records

Computed properties and operational findings recorded from running experiments. New
entries are appended as findings emerge.

### 16.1 KV cache capacity (GH200, 70B, TP=4)

```
Available KV per GPU  = (96 GiB × gpu_memory_utilization) − (140 GiB / 4 GPUs)
                      ≈ 51 GiB at 0.90

KV per 25K-token request per GPU  ≈ 80 KB/tok × 25,000 = 1.91 GiB
Max concurrent requests            ≈ 51 / 1.91  ≈ 28

With --kv-offloading-size 400 (100 GiB/GPU via Grace DRAM):
Additional KV capacity  = 100 / 1.91 ≈ 52 additional slots
Total concurrent        ≈ 80 slots
```

### 16.2 Speculative decoding on shared GPUs

- Running draft at TP=4 on the **same** 4 GPUs as the 70B target is counterproductive:
  NCCL allreduce overhead for 5 draft passes per cycle (320 extra syncs) erases the
  memory bandwidth benefit.
- Optimal configuration when GPUs are shared: **draft TP=1**, accept reduced GPU-0 KV capacity.
- Optimal configuration when dedicated GPUs are available: **draft on separate node/GPUs**.

---

## 17. Known Issues & Workarounds

Issues discovered during experiment runs — and the workarounds that resolved them — will
be tracked here as they arise. Empty for now.
