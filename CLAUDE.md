# Inference Benchmarking Tool

Systematic measurement of LLM inference deployments across multiple dimensions:

- **Latency and throughput envelope**: full latency profile (TTFT, TPOT, end-to-end; p50/p90/p95/p99, tail p99.9) across a sweep of request rates. Covers scaling efficiency, engine comparison (vLLM / SGLang / Dynamo), quantization, speculative decoding, KV / prefix caching, cold starts.
- **Pareto frontier**: sweep config options (TP, replicas, KV dtype, offloading) to find optimal operating points.
- **Token efficiency**: tokens consumed per task for a given model + config (system prompts, templates).
- **Cost efficiency**: cost per completed task at a given SLO — for buy-vs-scale decisions.
- **Reliability**: request error rates across load levels.
- **Response quality**: quality cost of serving optimizations (weight quantization, KV dtype, …) measured via graded evals against the deployed endpoint (SPECIFICATIONS.md §13.5) — capacity gains and quality deltas disclosed in the same report; a pre-sweep sanity gate protects every sweep from measuring a corrupted deployment.
- **Model loading times**: time-to-ready per configuration (SPECIFICATIONS.md §10.2) — supports auto-scaling decisions.
- **Hardware elasticity**: benefit of dynamically adding compute under load.

## Architecture

A LLM deployment can use different engines (vLLM, SGLang, Nvidia Dynamo), span a single GPU / many GPUs on one node / multiple nodes, run on SLURM (via FirecREST) or Kubernetes, and involve one or many replicas (to include ingress and routing overhead). The **Benchmarker** is a dedicated SLURM allocation (separate from the inference deployment's allocation) that both spawns the inference deployment(s) under test and hosts the dataset generator and the load generator.

### Components

Laptop-side (run by the operator, not allocated on any cluster):

- **Pre-flight checker** — runs at the start of every session to verify credentials, cluster reachability, and K8s capacity. Distinct from the in-engine-container hardware gate in §8. (§4)
- **Planner** — renders Jinja2 templates into the experiment's deployment artifacts (EDF / K8s manifests / sbatch / benchmark YAML). Driven via Claude Code or CLI. (§5)
- **Coordinator** — drives one experiment end-to-end: submits the Benchmarker job, monitors progress, downloads the per-run SQLite DB into the centralized results DB, tears down on both success and failure paths. SLURM via FirecREST MCP, K8s via kubectl. (§7)
- **Reports generator** — generates and executes a Jupyter notebook producing tables, plots, per-class SLO attainment (λ*), and the supportable-users estimate from the centralized results DB. (§13.4, §15)
- **Cleaner** — manual, operator-approved cleanup of state that escaped the Coordinator's per-run teardown (orphaned JFrog images, leftover K8s objects, stale capstor dirs). Two stages: identification (always) + pruning (manual approval). Claude periodically reminds the operator to run it; never executes itself. (§7.7)

Cluster-side (allocated on the cluster under test):

- **Benchmarker** — Coordinator-submitted SLURM allocation, one per experiment. Runs three sequential phases (dataset prep → engine spawn → load generation); spawns the inference deployment **only after the dataset generator has completed** so GPUs do not sit idle during CPU-bound prompt prep. Hosts:
  - **Dataset generator** — produces the prompt dataset from the experiment's weighted `scenario_mix` (synthetic with unique headers, or real-text e.g. LongBench / WildChat). (§11)
  - **Load generator** — awaits LLM readiness, issues requests at rate λ via the configured arrival process (Poisson or burst-aware), collects per-request metrics. (§10.3, §12)
  - **Quality evaluator** — lm-eval-harness against the deployed endpoint(s): pre-sweep sanity gate (default-on, skippable) + post-sweep quality comparison across deployment configs. (§13.5)
- **Inference deployment(s) under test** — the LLM serving stack(s) being measured (engine + replicas + ingress/auth/accounting). Spawned by the Benchmarker as SLURM jobs or K8s manifests; multi-instance deployments map to rows in the `instances` table. (§16.2, §14.2)

## Targeted Infrastructures

| Name | Type | Access | Notes |
|---|---|---|---|
| `clariden` | SLURM | FirecREST MCP (ML Platform) | Grace-Hopper (GH200), 4 GPUs/node |
| `bristen` | SLURM | FirecREST MCP (ML Platform) | A100, 4 GPUs/node |
| `beverin` | SLURM | FirecREST MCP (HPC Platform) | AMD MI300A, 4 GPUs/node |
| `breithorn` | Kubernetes | kubectl | L1/L2 cluster |

Additional targets may be added (e.g. systems outside CSCS).

## Project files and folders structure

- `.venv` — uv-based Python 3.14 virtualenv (`source .venv/bin/activate`).
- `tools/` — implementation of the components above (includes `pre-flight-checks.py`).
- `experiments/` — per-experiment folders (`YYYY-MM-DD_description/`) with config, deployment artifacts (Dockerfiles, sbatch, K8s YAML), and raw results.
- `reports/` — curated, audience-facing reports synthesised from one or many `experiments/` runs. Distinct from the per-experiment notebook (which lives under `experiments/<run>/`). See §15.3.
- `examples/` — image build, communication-plane pre-checks (`nccl-tests`), benchmark-config examples, and vLLM deployment on K8s and SLURM.
- `firecrest-mcp/` — FirecREST MCP server registered in Claude Code. Do not modify; use via its registered tools.
- `SPECIFICATIONS.md` — authoritative reference for detailed requirements, schema, known constraints, and cluster-specific workarounds. Read it before making changes to the tool.
- `IMPLEMENTATION_PLAN.md` — build order: component milestones (M0–M11), experiments track (E1–E5), dependencies, definitions of done, open decisions, review log. Living document; consult and update it when starting or finishing implementation work.
- `TODOs.md` — tracked future work; deferred activities per the working agreement (see "How we work together" below).

## Constants

Literal values to quote exactly when running project commands. Using anything else is the cause of the recurrent "python command not found / wrong python / pip not found" iterate-to-fix loop.

- **Python interpreter** — `.venv/bin/python` (absolute: `/Users/user/git/inference-benchmarking-tool/.venv/bin/python`). Python 3.14.4, uv-managed. Never invoke the macOS system `python3` for project commands — it does not see the venv's packages.
- **Venv activation** — `source .venv/bin/activate`, run from the repo root. After activation, bare `python` resolves to the venv binary.
- **Package installs** — `uv pip install <pkg>` (system `uv`, 0.8.22). The venv intentionally does **not** ship a `pip` binary; `pip install ...` from a non-activated shell will fail with "command not found", and even after activation `pip` is absent. Use `uv pip` instead.
- **Running a one-off Python command** — prefer `.venv/bin/python -c '...'` over activating the shell first; it is one fewer step and avoids leaking the activation into subsequent commands.

Cluster-side constants (cluster catalogue, capstor scratch base, SLURM account, collective-tests cache dir, JFrog base) live in **`tools/common/global.yaml`** (SPECIFICATIONS.md §3.3), loaded by `tools/common/config.py` — read them from there instead of hardcoding literals. Per-experiment values stay in the benchmark YAML.

## How we work together

Behavioural constraints for assisting on this project. This list grows over time; treat each entry as binding.

- **No unverified ceilings.** Do not conclude that an extractable metric (throughput, latency headroom, memory utilisation, acceptance rate, …) has reached its ceiling unless you can ground that claim in analytical proof — hardware limits (peak HBM bandwidth, NVLink throughput, FLOPs roofline, …) or algorithmic ones (theoretical concurrency cap from KV-cache budget, scheduling lower bounds, …). On a plateau, the default hypothesis is *more is extractable* — degraded foundation, suboptimal config, or a missing optimisation — not "this is the limit".
- **Defer with a TODO, not silence.** When the verification or follow-up another point demands cannot be done in the current iteration (no time, out-of-scope, requires a separate run), record it in `TODOs.md` as a deferred activity rather than dropping it. The TODO entry is the trace that the question is still open, not closed.
- **Flexible sequential process.** Tool development and experiment design + execution follow a soft sequential process — the suggested path from intent to outcome:
  1. **Goal definition** (including the envisioned results report)
  2. **Literature and research**
  3. **Options analysis and decisions**
  4. **Strategy alignment** with the broader effort
  5. **Planning and design**
  6. **Adversarial review** (of the plan)
  7. **Implementation** of the changes
  8. **Adversarial review** (of the implementation)
  9. **Audit**
  10. **Pre-execution assessment** (go / no-go on readiness)
  11. **Execution**
  12. **Monitoring + error-recovery loop** during execution
  13. **Results collection**
  14. **Results review and statistical analysis**
  15. **Results assessment** (does the data answer the goal? validity, limitations)
  16. **Adversarial review** (of the results)
  17. **Conclusions write-up** (synthesise findings, link back to the goal)
  18. **Report generation**
  19. **User validation** of the results

  Each phase may draw on different roles, skills, and expertises relevant to the work and to the specific experiment. The process is **soft**: skips require explicit user approval, and the user may rewind to any earlier phase and restart from there.

  Across all phases, Claude is responsible for two cross-cutting duties:
  - **Surfacing ambiguity through clarifying questions** at every step — preventing assumptions, vetting every possibility, and pausing to ask when something is unclear rather than guessing.
  - **Calling in the right roles, skills, and expertises** as the situation and the specific experiment demand — not pretending generalist coverage when a specialist is required.
