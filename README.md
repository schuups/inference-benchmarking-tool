# Inference Benchmarking Tool

__This is a refactoring of the early experimentations made in [https://github.com/schuups/inference-benchmarking](https://github.com/schuups/inference-benchmarking).__

Systematic measurement of LLM inference deployments across latency/throughput envelopes, Pareto frontiers, token and cost efficiency, reliability, model loading time, and hardware elasticity.

Requests are issued as open-loop load at a fixed Poisson rate λ, independent of server completions — this models realistic queuing behavior and produces latency-vs-throughput operating curves.

## Architecture

- **Coordinator** (laptop) — submits experiments, monitors execution, collects results into a centralized database on the laptop, and cleans up on success and failure.
- **Benchmarker** (SLURM allocation, separate from the inference server) — hosts the **dataset generator** and the **load generator**.
- **Reports generator** — produces Jupyter notebooks (tables, plots) from the centralized results database.
- **Cleaner** — periodic garbage-collection pass for state that escaped the Coordinator's per-run teardown.

A LLM deployment can use different engines (vLLM, SGLang, Nvidia Dynamo), span one or many GPUs / nodes, be deployed on SLURM (via FirecREST) or Kubernetes, and involve one or multiple replicas to exercise ingress and routing overhead.

See `SPECIFICATIONS.md` for detailed requirements, schema definitions, and cluster-specific constraints.

## Targeted Infrastructures

| Name | Type | Access | Notes |
|---|---|---|---|
| `clariden` | SLURM | FirecREST MCP (ML Platform) | Grace-Hopper (GH200) nodes |
| `beverin` | SLURM | FirecREST MCP (HPC Platform) | AMD MI300A nodes |
| `breithorn` | Kubernetes | kubectl | L1/L2 cluster |

## Models of Interest

Capability columns indicate whether the model natively supports each modality/feature.

| Model | Text | Reasoning / Thinking | Multimodal | Tools |
|---|---|---|---|---|
| **Apertus 70B 1.5** (Swiss AI / EPFL / ETHZ / CSCS) | Yes — multilingual (1000+ languages, incl. Swiss German, Romansh) | No dedicated thinking mode (base model) | Yes — image input (new in 1.5) | Yes — improved tool-use (new in 1.5) |
| **Kimi-K2.6** (Moonshot AI) | Yes | Yes — deeper reasoning and planning; strong on agentic, multi-step workflows | Yes — text + image + video (MoonViT encoder; multimodal performance comparatively weak) | Yes — strong tool-use reliability; leads open weights on HLE-with-tools |
| **DeepSeek-V4 Pro** | Yes — 1M-token context | Yes — Think High (bounded) / Think Max (unbounded) modes; interleaved thinking preserved across tool calls | Yes — text + image + video + audio | Yes — XML-based schema with a dedicated `\|DSML\|` separator token for structured params |
| **GLM-5.1** (Zhipu AI) | Yes — 202K context | Yes — "rumination" multi-iteration self-revision; long autonomous loops (up to 8 h) | Yes — images + documents + text in unified pipeline | Yes — agentic planning + tool use (multi-step) |

Apertus 70B 1.5 is the incremental successor to Apertus 70B with better tool-use and image input added; public material is sparse.

## Layout

- `tool/` — implementation of the components above; includes `pre-flight-checks.py`
- `experiments/` — per-experiment folders (`YYYY-MM-DD_description/`) with config, deployment artifacts, raw results
- `reports/` — generated notebooks and rendered outputs
- `examples/` — reference Docker image builds, K8s and SLURM deployments
- `firecrest-mcp/` — FirecREST MCP server registered in Claude Code
- `SPECIFICATIONS.md` — authoritative reference for requirements and constraints
- `TODOs.md` — tracked future work
