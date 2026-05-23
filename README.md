# Inference Benchmarking Tool

__This is a refactoring of the early experimentations made in [https://github.com/schuups/inference-benchmarking](https://github.com/schuups/inference-benchmarking).__

**TL/DR:** Tool for systematic measurement of LLM inference deployment options — latency/throughput envelopes, token and cost efficiency, reliability, model loading time, and hardware elasticity — to map Pareto frontiers across systems.

**Goal:** Tune deployments for **maximum throughput subject to latency constraints** (rather than peak tokens/second), and feed evidence into next-generation system procurements.

**Operating model:** The repository is designed to be developed and operated through an AI coding assistant — such as Claude Code — driving the Coordinator on the laptop, dispatching SLURM jobs via the FirecREST MCP, and reasoning about results. That said, it is **not** required: to stay accessible to contributors without a high-tier AI subscription, every step the assistant takes is also a plain shell command or Python entry point that can be invoked manually. The assistant orchestrates and interprets; it does not hide functionality behind itself.

## Approach

The framework load-tests inference-serving deployments under realistic online conditions to characterise **latency-constrained throughput**, **tail-latency behaviour under varying concurrency**, and the effectiveness of semantics-sensitive serving optimizations. Measurements span deployment architectures, runtime schedulers, autoscaling policies, interconnect-aware inference strategies, and heterogeneous serving infrastructures (e.g. prefill on one accelerator class and decode on another) under production-like operating conditions.

### Background

Inference deployments are evaluated using **open-loop stochastic request generation**: requests arrive independently of server completions according to configurable arrival processes (Poisson and burst-aware variants), preserving the queueing dynamics that arise in practice — backlog formation, saturation behaviour, and latency amplification under load. This contrasts both with closed-loop measurement (which truncates queueing pathologies by construction) and with the simple fixed-rate open-loop scheme used in the previous version of this tool (see the archived repository linked above), which lacked burst-aware processes and a principled scenario framework.

The benchmark targets the inference-serving system end-to-end, exercising the optimizations whose effectiveness depends on workload semantics and temporal locality — automatic prefix caching, speculative decoding, continuous batching, Mixture-of-Experts (MoE) routing, KV-cache reuse, and scheduler behaviour. Synthetic filler does not approximate these mechanisms: speculative acceptance rates, expert-routing entropy (how evenly tokens are distributed across the model's experts, which governs MoE load balance and per-expert memory pressure), batching efficiency, and cache locality depend on actual token-stream semantics rather than on sequence length alone.

### Workloads and scenarios

Workloads are generated from **semantically realistic prompt and response distributions**. Scenarios include conversational chat, AI-assisted coding, multi-modal interactions (e.g. text + image), long-context follow-up sessions representative of returning users (typified by a short follow-up prompt issued after an initial large context, as in AI-assisted coding), and reasoning-intensive workloads containing realistic reasoning traces and token distributions rather than arbitrarily long filler.

**Agentic and tool-calling workloads** are first-class scenarios. They differ from conversational traffic in two ways that the load generator models explicitly:

- **Fan-out per user request** — a single user turn produces many model invocations (think → tool call → tool result → think …), so the unit of measurement is the agent task, not the individual request.
- **Bimodal token distributions** — outputs alternate between tiny structured tool calls and large injected tool results (RAG-style documents, code-execution output), creating mixed prefill/decode pressure that uniform synthetic prompts cannot reproduce.

Schema-constrained decoding (JSON / XML), structured-output validity, and end-to-end agent-task latency are recorded alongside the standard request-level metrics.

### Forward-looking scenario taxonomy

Procurement-grade evidence must anticipate workloads that will dominate in 3–5 years, not only those that dominate today. The scenario taxonomy is therefore reviewed on a regular cadence against leading indicators — frontier model releases, agent-framework launches, published production telemetry — and each scenario carries a **maturity tag** (`established`, `emerging`, `exploratory`) in its manifest, so reports can mark which Pareto frontiers come from validated workloads versus early signals.

| Scenario | Maturity | What it stresses |
|---|---|---|
| 💬 Conversational chat (short turns) | 🪨 `established` | Multi-turn prefix-cache hits; high request rate; small per-turn token counts. |
| 💻 AI-assisted coding (long context + follow-ups) | 🪨 `established` | KV-cache reuse, prefix caching, returning-user locality, long-context decoding. |
| 🖼️ Multimodal prompts (text + image) | 🪨 `established` | Mixed-modality token streams; vision encoder coupled to the LLM. |
| 🧠 Reasoning-intensive workloads (chain-of-thought) | 🪨 `established` | Decode-heavy traffic; speculative-acceptance rates over real reasoning traces. |
| 🛠️ Agentic tool-calling (think → tool → think) | 🪨 `established` | Multi-turn fan-out; bimodal token distributions; schema-constrained decoding. |
| 📦 Code-execution sandboxes interleaved with reasoning | 🪨 `established` | Large injected tool results; tight observe-think-act loops. |
| 📚 Retrieval-augmented + long-context decoding | 🪨 `established` | Large injected documents; mixed prefill pressure with long output. |
| 🤖 Long-horizon autonomous agents (multi-hour task graphs) | 🧪 `exploratory` | Long-lived sessions, persistent context, mixed compute and tool latency. |
| 🕸️ Multi-agent orchestration (shared context) | 🧪 `exploratory` | Inter-agent message passing; collective context updates. |

The list is reviewed on the same cadence and grows as new patterns emerge — new entries enter as `exploratory`, advance to `emerging` once partially validated, and to `established` once they match observed production telemetry.

### Key contributions

MLPerf Inference: Datacenter is a valuable, fair, reproducible cross-platform benchmark; this framework does not aim to replace it, but to cover workload and systems dimensions that MLPerf intentionally abstracts for comparability:

1. **Semantic workload realism.** Realistic prompt/response distributions, so semantics-dependent optimizations — speculative decoding, MoE routing, prefix caching, KV-cache reuse, continuous batching — are exercised rather than averaged out.
2. **Temporal locality.** Long-lived sessions and returning-user follow-ups over established contexts are first-class, capturing locality patterns typical of AI-assisted coding, agentic workflows, and RAG.
3. **Agentic and tool-calling realism.** Tool-calling fan-out and bimodal output distributions are modelled explicitly; metrics frame around the agent task rather than the individual request.
4. **Open-loop queueing dynamics.** Poisson and burst-aware arrivals expose backlog growth, tail-latency amplification, scheduler collapse, admission control, and autoscaling responsiveness — invisible to closed-loop measurement.
5. **Service-oriented evaluation.** Captures schedulers, batching, routing, distributed runtimes, heterogeneous accelerators, interconnects, autoscaling, and orchestration overhead — not just kernel execution.
6. **Modern LLM and multimodal workloads.** Long-context, reasoning-heavy, multimodal, and heterogeneous request mixes (mixed prefill/decode pressure, mixed latency sensitivity) — exposing memory-hierarchy, scheduling, and interconnect bottlenecks rather than peak FLOPs.
7. **Forward-looking scenario taxonomy.** Workload taxonomy reviewed on cadence against leading indicators; scenarios carry maturity tags so procurement evidence distinguishes validated patterns from emerging trends.

## Architecture

- **Inference deployment under test** — the complete LLM serving stack: the engine (vLLM, SGLang, or NVIDIA Dynamo) on its own GPU allocation (SLURM or Kubernetes, single- or multi-replica), *plus* the ingress, authentication, and accounting layers in front of it. The framework load-tests the entire request path as users experience it, not the model server in isolation.
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

## Current models of interest

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

## Future roadmap

- Support **geo-redundancy scenarios**: load-test multi-site deployments to characterise failover latency (time from primary-site failure to standby serving traffic), in-flight request loss during failover, and the steady-state cost of cross-site routing — making availability and disaster-recovery posture testable by this tool.
- Cover **resource elasticity**: auto-scaling latency (time from load-spike detection to an additional replica ready and serving), request-loss during scale-up / scale-down events, and pre-warmed-pool sizing trade-offs. Results feed directly into the requirements definition for the elasticity feature of CSCS vClusters.
- Extend this benchmarking tool beyond infrastructure-centric metrics (e.g. TTFT and ITL) toward task efficiency evaluation — measuring how effectively a deployment configuration completes real tasks under realistic agentic workflows. This includes studying how deployment-time controls such as system prompts, decoding policies, tool availability, and context-management strategies influence token efficiency, task completion quality, and overall operational cost.

## References

Resources consulted to develop the methodology in this tool.

- *MLPerf Inference Benchmark* — Reddi et al., 2019 — [arXiv:1911.02549](https://arxiv.org/abs/1911.02549). The MLPerf Inference: Datacenter scenario definitions (Server / Offline) and metric conventions.
- *InferenceMAX: open-source inference benchmarking* — SemiAnalysis newsletter — [newsletter post](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference) and [project site](https://inferencex.semianalysis.com/).
