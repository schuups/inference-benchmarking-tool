# Inference Benchmarking Tool

Systematic measurement of LLM inference deployments across multiple dimensions:

- **Latency and throughput envelope**: map the full latency profile (TTFT, TPOT, end-to-end; p50/p90/p95/p99, including tail at p99.9) across a sweep of request rates. Characterizes the operating envelope and how many concurrent users can be served within a given SLO. Covers: scaling efficiency, engine comparison (vLLM vs SGLang vs Dynamo), quantization trade-offs, speculative decoding gain, KV cache / prefix caching impact, and cold start latency.
- **Pareto frontier**: sweep configuration options (tensor parallelism, replica count, KV dtype, offloading, etc.) to expose trade-offs and identify optimal operating points.
- **Token efficiency**: measure tokens consumed per task for a given model and configuration (system prompts, templates, etc.).
- **Cost efficiency**: cost per completed task at a given SLO, to support buy-vs-scale decisions.
- **Reliability**: request error rates across load levels to characterize failure modes under load.
- **Model loading times**: time-to-ready per configuration, to support auto-scaling decisions (§9.2 in SPECIFICATIONS.md).
- **Hardware elasticity**: measure the benefit of dynamically adding compute resources under load.

Requests are issued as open-loop load at a fixed Poisson rate λ, independent of server completions. This models realistic queuing behavior and produces latency-vs-throughput operating curves.

## Architecture

A LLM deployment:
- can be done using different engines (vLLM, SGLang, Nvidia Dynamo).
- can involve a single GPU, many GPUs on one node, or multiple nodes.
- can be deployed on SLURM (via FirecREST) or Kubernetes.
- can involve one or multiple instances (replicas) to include ingress and routing overhead.

The **Benchmarker** runs on a dedicated SLURM allocation, separate from the inference server, and hosts both the dataset generator and the load generator.

See SPECIFICATIONS.md for detailed requirements, constraints, and cluster-specific workarounds for each component.

### Components

**`tool/pre-flight-checks.py`** — verifies clusters accessibility and required configuration before running experiments.

**Planner** — supports experiment preparation via human interaction with Claude Code (rendering Jinja2 templates, preparing deployment scripts).

**Coordinator** (runs on laptop) — the laptop-side orchestrator. Submits experiments to remote clusters, monitors execution, collects results, appends them to the centralized database on the laptop, and performs cleanup on both success and failure. Interfaces with SLURM via FirecREST MCP and with Kubernetes via kubectl. (SPECIFICATIONS.md §2, §4, §10)

**Dataset generator** (runs on the Benchmarker) — produces the prompt dataset for an experiment (synthetic with unique headers, or real-text sources such as LongBench). (SPECIFICATIONS.md §8)

**Load generator** (runs on the Benchmarker) — the core benchmarking engine. Waits for LLM services to become available, sends requests at the configured Poisson rate λ, and collects per-request metrics. (SPECIFICATIONS.md §6, §7, §9)

**Reports generator** — generates and executes a Jupyter notebook producing tables and plots from the centralized results database. (SPECIFICATIONS.md §10)

**Cleaner** — periodic garbage-collection pass for state that escaped the Coordinator's per-run teardown (e.g. after a coordinator crash): orphaned Docker images in JFrog, leftover Kubernetes Ingress/PVs/Services, stale SLURM scratch directories. (SPECIFICATIONS.md §4)

## Working Conventions

- **Engine retrocompatibility — ask before changing the core**: existing experiment configs must remain re-runnable unchanged. The load generator and coordinator are the most stable components; changes to them risk breaking past experiments — always ask before modifying these, and prefer adding new capabilities as new modules rather than modifying existing interfaces. Templates, dataset generator, reports generator, and cleaner can be changed more freely.
- **SPECIFICATIONS.md is the authoritative reference** for detailed requirements, schema definitions, known constraints, and cluster-specific workarounds. Read it before making changes to the tool.

## Targeted Infrastructures

The tool is designed to target different infrastructures. Currently known targets:

| Name | Type | Access | Notes |
|---|---|---|---|
| `clariden` | SLURM | FirecREST MCP (ML Platform) | Grace-Hopper (GH200) nodes |
| `beverin` | SLURM | FirecREST MCP (HPC Platform) | AMD MI300A nodes |
| `breithorn` | Kubernetes | kubectl | L1/L2 cluster |

Additional targets may be added (e.g. systems outside CSCS).

## Models of Interest

Primary models targeted for benchmarking (see README.md for capability details):

- Apertus 70B 1.5
- Kimi-K2.6
- DeepSeek-V4 Pro
- GLM-5.1

## Project files and folders structure

- `.venv` is the uv-based python virtual environment to be used for everything. It is based on Python 3.14. It can be activated with `source .venv/bin/activate`
- `examples`: it contains examples for image building via SLURM, vLLM deployment on K8s and on SLURM
- `firecrest-mcp`: the FirecREST MCP server registered in Claude Code. Do not modify this code; use it as any other MCP server via its registered tools.
- `tool`: the actual tool implementing the components listed above. Includes `pre-flight-checks.py`, which verifies cluster accessibility and required configuration before running experiments.
- `experiments`: per-experiment folders (`YYYY-MM-DD_description/`) holding the full provenance of each run — config YAML, deployment artifacts (Dockerfiles, sbatch, K8s YAML), and raw results.
- `reports`: generated notebooks and rendered outputs (tables, plots) produced by the reports generator from the centralized results database.
