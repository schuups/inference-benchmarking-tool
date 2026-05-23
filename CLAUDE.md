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

A LLM deployment can use different engines (vLLM, SGLang, Nvidia Dynamo), span a single GPU / many GPUs on one node / multiple nodes, run on SLURM (via FirecREST) or Kubernetes, and involve one or many replicas (to include ingress and routing overhead). The **Benchmarker** is a dedicated SLURM allocation (separate from the inference server) hosting both the dataset generator and the load generator.

### Components

- **`tool/pre-flight-checks.py`** — verifies cluster accessibility and required configuration before running experiments.
- **Planner** — supports experiment preparation via interaction with Claude Code (rendering Jinja2 templates, deployment scripts).
- **Coordinator** (laptop) — submits experiments, monitors them, collects results into the centralized DB on the laptop, and cleans up on success and failure. SLURM via FirecREST MCP, K8s via kubectl. (§2, §4, §11)
- **Dataset generator** (on the Benchmarker) — produces the prompt dataset (synthetic with unique headers, or real-text e.g. LongBench). (§6)
- **Load generator** (on the Benchmarker) — core engine: awaits LLM readiness, sends requests at Poisson rate λ, collects per-request metrics. (§7, §8, §9, §10)
- **Reports generator** — generates and executes a Jupyter notebook producing tables and plots from the centralized results DB. (§12)
- **Cleaner** — periodic GC for state that escaped per-run teardown (orphaned JFrog images, leftover K8s Ingress/PV/Services, stale SLURM scratch dirs). (§4)

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
