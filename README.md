# Inference Benchmarking Tool

__This is a refactoring of the early experimentations made in [https://github.com/schuups/inference-benchmarking](https://github.com/schuups/inference-benchmarking).__

**TL/DR:** Tool for systematic measurement of LLM inference deployment options — latency/throughput envelopes, token and cost efficiency, reliability, model loading time, and hardware elasticity — to map Pareto frontiers across systems.

**Goal:** Tune deployments for **maximum throughput subject to latency constraints** (rather than peak tokens/second), and feed evidence into next-generation system procurements.

**Operating model:** Designed for operation through an AI coding assistant (such as Claude Code) — preparing artifacts via the **Planner**, driving the **Coordinator** on the laptop, dispatching jobs via FirecREST MCP / kubectl, and reasoning about results. The assistant is **not** required: every step is also a plain shell command or Python entry point. Component definitions live in [`SPECIFICATIONS.md`](SPECIFICATIONS.md).

## Approach

The framework load-tests inference-serving deployments under realistic online conditions to characterise **latency-constrained throughput**, **tail-latency behaviour under varying concurrency**, and the effectiveness of semantics-sensitive serving optimizations. Measurements span deployment architectures, runtime schedulers, autoscaling policies, interconnect-aware inference strategies, and heterogeneous serving infrastructures (e.g. prefill on one accelerator class and decode on another) under production-like operating conditions.

### Background

Inference deployments are evaluated using **open-loop stochastic request generation**: requests arrive independently of server completions according to configurable arrival processes (Poisson and burst-aware variants), preserving the queueing dynamics that arise in practice — backlog formation, saturation behaviour, and latency amplification under load. This contrasts both with closed-loop measurement (which truncates queueing pathologies by construction) and with the simple fixed-rate open-loop scheme used in the previous version of this tool (see the archived repository linked above), which lacked burst-aware processes and a principled scenario framework.

The benchmark targets the inference-serving system end-to-end, exercising the optimizations whose effectiveness depends on workload semantics and temporal locality — automatic prefix caching, speculative decoding, continuous batching, Mixture-of-Experts (MoE) routing, KV-cache reuse, and scheduler behaviour. Synthetic filler does not approximate these mechanisms: speculative acceptance rates, expert-routing entropy (how evenly tokens are distributed across the model's experts, which governs MoE load balance and per-expert memory pressure), batching efficiency, and cache locality depend on actual token-stream semantics rather than on sequence length alone.

### Workloads and scenarios

Workloads are generated from **semantically realistic prompt and response distributions**. Scenarios include conversational chat, AI-assisted coding, long-context follow-up sessions representative of returning users (typified by a short follow-up prompt issued after an initial large context, as in AI-assisted coding), and reasoning-intensive workloads containing realistic reasoning traces and token distributions rather than arbitrarily long filler. Scenarios are blended within a single experiment via a weighted **`scenario_mix`** (e.g. 80% agentic coding + 20% chat), so cross-class interference — long agentic prefills inflating chat TTFT on a shared instance — is part of every measurement (SPECIFICATIONS.md §10.4). v1 covers text only; multimodality (image, then audio and video) is the next feature on the dataset-generator roadmap — see `TODOs.md`.

**Agentic workloads** are first-class scenarios. v1 approximates them as **multi-turn sessions with bursty fan-out**: each session models one agentic task, each turn within the session models one model invocation, and tool results are synthesised as injected text in the next turn's prompt. This is enough to answer the framework's primary v1 question — *how many concurrent users can a model instance support under a realistic workload mix?* — by sweeping λ at the LLM endpoint, applying per-class SLOs to find the SLO-attained rate λ* (SPECIFICATIONS.md §12.4), and translating λ* to per-class user counts in the report notebook (§14.1).

The precise mechanics (per-tool JSON schemas, fan-out template DSL, schema-constrained decoding, a dedicated `agent_tasks` table, first-class bimodal output distributions) are deferred to a future iteration — see `TODOs.md` *Precise agentic / tool-calling measurement*.

### Forward-looking scenario taxonomy

Procurement-grade evidence must anticipate workloads that will dominate in 3–5 years, not only those that dominate today. The scenario taxonomy is therefore reviewed on a regular cadence against leading indicators — frontier model releases, agent-framework launches, published production telemetry — and each scenario carries a **maturity tag** (`established`, `emerging`, `exploratory`) in its manifest, so reports can mark which Pareto frontiers come from validated workloads versus early signals.

| Scenario | Maturity | What it stresses |
|---|---|---|
| 💬 Conversational chat (short turns) | 🪨 `established` | Multi-turn prefix-cache hits; high request rate; small per-turn token counts. |
| 💻 AI-assisted coding (long context + follow-ups) | 🪨 `established` | KV-cache reuse, prefix caching, returning-user locality, long-context decoding. |
| 🖼️ Multimodal prompts (text + image) | 🪨 `established` | Mixed-modality token streams; vision encoder coupled to the LLM. |
| 🧠 Reasoning-intensive workloads (chain-of-thought) | 🪨 `established` | Decode-heavy traffic; speculative-acceptance rates over real reasoning traces. |
| 🛠️ Agentic tool-calling (think → tool → think) | 🪨 `established` | Multi-turn fan-out as a session of model invocations (v1); precise per-tool modelling deferred. |
| 📦 Code-execution sandboxes interleaved with reasoning | 🪨 `established` | Large injected tool results; tight observe-think-act loops. |
| 📚 Retrieval-augmented + long-context decoding | 🪨 `established` | Large injected documents; mixed prefill pressure with long output. |
| 🤖 Long-horizon autonomous agents (multi-hour task graphs) | 🧪 `exploratory` | Long-lived sessions, persistent context, mixed compute and tool latency. |
| 🕸️ Multi-agent orchestration (shared context) | 🧪 `exploratory` | Inter-agent message passing; collective context updates. |

The list is reviewed on the same cadence and grows as new patterns emerge — new entries enter as `exploratory`, advance to `emerging` once partially validated, and to `established` once they match observed production telemetry.

### Key contributions

MLPerf Inference: Datacenter is a valuable, fair, reproducible cross-platform benchmark; this framework does not aim to replace it, but to cover workload and systems dimensions that MLPerf intentionally abstracts for comparability:

1. **Semantic workload realism.** Realistic prompt/response distributions, so semantics-dependent optimizations — speculative decoding, MoE routing, prefix caching, KV-cache reuse, continuous batching — are exercised rather than averaged out.
2. **Temporal locality.** Long-lived sessions and returning-user follow-ups over established contexts are first-class, capturing locality patterns typical of AI-assisted coding, agentic workflows, and RAG.
3. **Agentic-workload approximation.** Agentic workloads are modelled as multi-turn sessions with bursty fan-out — enough to derive supportable-user-count from the SLO-attained rate λ* (SPECIFICATIONS.md §12.4, §14.1); precise per-tool modelling deferred (see `TODOs.md`).
4. **Open-loop queueing dynamics.** Poisson and burst-aware arrivals expose backlog growth, tail-latency amplification, scheduler collapse, admission control, and autoscaling responsiveness — invisible to closed-loop measurement.
5. **Service-oriented evaluation.** Captures schedulers, batching, routing, distributed runtimes, heterogeneous accelerators, interconnects, autoscaling, and orchestration overhead — not just kernel execution.
6. **Modern LLM and multimodal workloads.** Long-context, reasoning-heavy, multimodal, and heterogeneous request mixes (mixed prefill/decode pressure, mixed latency sensitivity) — exposing memory-hierarchy, scheduling, and interconnect bottlenecks rather than peak FLOPs.
7. **Forward-looking scenario taxonomy.** Workload taxonomy reviewed on cadence against leading indicators; scenarios carry maturity tags so procurement evidence distinguishes validated patterns from emerging trends.

## Reference documentation

- [`CLAUDE.md`](CLAUDE.md) — architecture (components and their boundaries), targeted clusters, repo layout, environment constants, working agreement.
- [`SPECIFICATIONS.md`](SPECIFICATIONS.md) — authoritative requirements, schema definitions, cluster-specific constraints, and known workarounds.
- [`TODOs.md`](TODOs.md) — tracked future work.

## Current models of interest

Capability columns indicate whether the model natively supports each modality/feature.

| Model | Text | Reasoning / Thinking | Multimodal | Tools |
|---|---|---|---|---|
| **Apertus 70B** (Swiss AI / EPFL / ETHZ / CSCS) | Yes — multilingual (1000+ languages, incl. Swiss German, Romansh) | No dedicated thinking mode (base model) | Yes — image input | Yes — tool-use |
| **Kimi-K2.6** (Moonshot AI) | Yes | Yes — deeper reasoning and planning; strong on agentic, multi-step workflows | Yes — text + image + video (MoonViT encoder; multimodal performance comparatively weak) | Yes — strong tool-use reliability; leads open weights on HLE-with-tools |
| **DeepSeek-V4-Pro** | Yes — 1M-token context | Yes — three modes: Non-think / Think High / Think Max; toggled via `thinking_mode` runtime parameter; `<think>` / `</think>` delimiters in output | No — text only | Yes — tool-use supported via the model's custom encoding (`encoding_dsv4`); 1.6T MoE / 49B activated |
| **GLM-5.1** (Zhipu AI) | Yes — 202K context | Yes — "rumination" multi-iteration self-revision; long autonomous loops (up to 8 h) | Yes — images + documents + text in unified pipeline | Yes — agentic planning + tool use (multi-step) |

## Future roadmap

- Support **geo-redundancy scenarios**: load-test multi-site deployments to characterise failover latency (time from primary-site failure to standby serving traffic), in-flight request loss during failover, and the steady-state cost of cross-site routing — making availability and disaster-recovery posture testable by this tool.
- Cover **resource elasticity**: auto-scaling latency (time from load-spike detection to an additional replica ready and serving), request-loss during scale-up / scale-down events, and pre-warmed-pool sizing trade-offs. Results feed directly into the requirements definition for the elasticity feature of CSCS vClusters.
- Extend this benchmarking tool beyond infrastructure-centric metrics (e.g. TTFT and ITL) toward task efficiency evaluation — measuring how effectively a deployment configuration completes real tasks under realistic agentic workflows. This includes studying how deployment-time controls such as system prompts, decoding policies, tool availability, and context-management strategies influence token efficiency, task completion quality, and overall operational cost.
- Extend **modality coverage** beyond text: v1 of the dataset generator handles text-only scenarios; **image** is the next planned modality, followed by **audio** and **video** (paired corpora, per-second / per-clip token-cost accounting, registry-load-time acceptance of `modalities: [image]` / `[audio]` / `[video]`) — all tracked in [`TODOs.md`](TODOs.md).

## References

Resources consulted to develop the methodology in this tool.

- *MLPerf Inference Benchmark* — Reddi et al., 2019 — [arXiv:1911.02549](https://arxiv.org/abs/1911.02549). The MLPerf Inference: Datacenter scenario definitions (Server / Offline) and metric conventions.
- *InferenceMAX: open-source inference benchmarking* — SemiAnalysis newsletter — [newsletter post](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference) and [project site](https://inferencex.semianalysis.com/).
