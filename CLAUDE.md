# Inference Benchmarking Tool

Systematic measurement of LLM inference deployments across multiple dimensions:

- **Latency and throughput envelope**: full latency profile (TTFT, TPOT, end-to-end; p50/p90/p95/p99, tail p99.9) across a sweep of request rates. Covers scaling efficiency, engine comparison (vLLM / SGLang / Dynamo), quantization, speculative decoding, KV / prefix caching, cold starts.
- **Pareto frontier**: sweep config options (TP, replicas, KV dtype, offloading) to find optimal operating points.
- **Token efficiency**: tokens consumed per task for a given model + config (system prompts, templates).
- **Cost efficiency**: cost per completed task at a given SLO — for buy-vs-scale decisions.
- **Reliability**: request error rates across load levels.
- **Model loading times**: time-to-ready per configuration (SPECIFICATIONS.md §10.2) — supports auto-scaling decisions.
- **Hardware elasticity**: benefit of dynamically adding compute under load.

## Architecture

A LLM deployment can use different engines (vLLM, SGLang, Nvidia Dynamo), span a single GPU / many GPUs on one node / multiple nodes, run on SLURM (via FirecREST) or Kubernetes, and involve one or many replicas (to include ingress and routing overhead). The **Benchmarker** is a dedicated SLURM allocation (separate from the inference deployment's allocation) that both spawns the inference deployment(s) under test and hosts the dataset generator and the load generator.

### Components

Laptop-side (run by the operator, not allocated on any cluster):

- **Pre-flight checker** — verifies cluster accessibility and required configuration before running experiments.
- **Planner** — prepares experiments from Claude-based interactions (rendering Jinja2 templates, deployment scripts).
- **Coordinator** — drives a single experiment end-to-end: submits the Benchmarker job, monitors it (and the inference deployment(s) it spawns), downloads the per-run SQLite DB into the centralized results DB on the laptop, and tears down the resources **it created for that run** on both success and failure paths (cancel the Benchmarker job — which in turn cancels its inference deployment(s) — delete K8s objects, remove the run's capstor scratch dir; §4). SLURM via FirecREST MCP, K8s via kubectl. (§2, §4, §11)
- **Reports generator** — generates and executes a Jupyter notebook producing tables and plots from the centralized results DB. (§12)
- **Cleaner** — periodic, cluster-wide garbage collection for state that escaped the Coordinator's per-run teardown (e.g. coordinator killed mid-run, network failure during teardown, or older runs predating a teardown fix): orphaned JFrog images, leftover K8s Ingress/PV/Services, stale SLURM scratch dirs. Discovers resources via the labels in §3 — scoped to *all* benchmark-managed resources older than a threshold, not to one specific run. (§4)

Cluster-side (allocated on the cluster under test):

- **Benchmarker** — Coordinator-submitted SLURM allocation, one per experiment, separate from the inference deployment's. Runs three sequential phases — dataset prep → engine spawn → load generation. Spawns the inference deployment(s) under test **only after the dataset generator has completed** (so GPUs stay out of idle during CPU-bound prompt prep); polls readiness, records model-load times into `instances`, tears down at end-of-experiment. Hosts:
  - **Dataset generator** — produces the prompt dataset (synthetic with unique headers, or real-text e.g. LongBench). (§7)
  - **Load generator** — awaits LLM readiness, sends requests at Poisson rate λ, collects per-request metrics. (§7, §8, §9, §10)
- **Inference deployment(s) under test** — the actual subject of measurement: one or more LLM serving stacks (engine + replicas + ingress/auth/accounting where applicable), spawned by the Benchmarker as separate SLURM jobs (SLURM target) or K8s manifests (K8s target). A single experiment may deploy multiple instances of the same configuration for multi-replica / routing studies — each instance is recorded as a row in the `instances` table. (§2, §5, §13.2)

## Targeted Infrastructures

| Name | Type | Access | Notes |
|---|---|---|---|
| `clariden` | SLURM | FirecREST MCP (ML Platform) | Grace-Hopper (GH200) nodes |
| `beverin` | SLURM | FirecREST MCP (HPC Platform) | AMD MI300A nodes |
| `breithorn` | Kubernetes | kubectl | L1/L2 cluster |

Additional targets may be added (e.g. systems outside CSCS).

## Project files and folders structure

- `.venv` — uv-based Python 3.14 virtualenv (`source .venv/bin/activate`).
- `tool/` — implementation of the components above (includes `pre-flight-checks.py`).
- `experiments/` — per-experiment folders (`YYYY-MM-DD_description/`) with config, deployment artifacts (Dockerfiles, sbatch, K8s YAML), and raw results.
- `reports/` — generated notebooks and rendered outputs (tables, plots).
- `examples/` — image build via SLURM, vLLM deployment on K8s and SLURM.
- `firecrest-mcp/` — FirecREST MCP server registered in Claude Code. Do not modify; use via its registered tools.
- `SPECIFICATIONS.md` — authoritative reference for detailed requirements, schema, known constraints, and cluster-specific workarounds. Read it before making changes to the tool.
- `TODOs.md` — tracked future work; deferred activities per the working agreement (see "How we work together" below).

## Constants

Literal values to quote exactly when running project commands. Using anything else is the cause of the recurrent "python command not found / wrong python / pip not found" iterate-to-fix loop.

- **Python interpreter** — `.venv/bin/python` (absolute: `/Users/user/git/inference-benchmarking-tool/.venv/bin/python`). Python 3.14.4, uv-managed. Never invoke the macOS system `python3` for project commands — it does not see the venv's packages.
- **Venv activation** — `source .venv/bin/activate`, run from the repo root. After activation, bare `python` resolves to the venv binary.
- **Package installs** — `uv pip install <pkg>` (system `uv`, 0.8.22). The venv intentionally does **not** ship a `pip` binary; `pip install ...` from a non-activated shell will fail with "command not found", and even after activation `pip` is absent. Use `uv pip` instead.
- **Running a one-off Python command** — prefer `.venv/bin/python -c '...'` over activating the shell first; it is one fewer step and avoids leaking the activation into subsequent commands.

Cluster-side constants (capstor scratch base, JFrog publish path, default SLURM account, …) are deferred until populated; see `TODOs.md` ("Establish a global configuration location for shared values").

## How we work together

Behavioural constraints for assisting on this project. This list grows over time; treat each entry as binding.

- **No unverified ceilings.** Do not conclude that an extractable metric (throughput, latency headroom, memory utilisation, acceptance rate, …) has reached its ceiling unless you can ground that claim in analytical proof — hardware limits (peak HBM bandwidth, NVLink throughput, FLOPs roofline, …) or algorithmic ones (theoretical concurrency cap from KV-cache budget, scheduling lower bounds, …). On a plateau, the default hypothesis is *more is extractable* — degraded foundation, suboptimal config, or a missing optimisation — not "this is the limit".
- **Defer with a TODO, not silence.** When the verification or follow-up another point demands cannot be done in the current iteration (no time, out-of-scope, requires a separate run), record it in `TODOs.md` as a deferred activity rather than dropping it. The TODO entry is the trace that the question is still open, not closed.
