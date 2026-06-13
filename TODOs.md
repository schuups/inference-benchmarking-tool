# TODOs

## Architecture & Engine

- [ ] Define which files and folders are immutable (cannot be modified when running experiments)
- [ ] Structure the tool modularly so the core engine is rarely (ideally never) modified; new capabilities added as modules

## Experiment Execution

- [ ] Run NCCL benchmarks (using the same Docker images) before inference benchmarks
- [ ] Support testing endpoints provided via URL (not only SLURM and Kubernetes deployments)

### Cold-start optimisation experiment groups

- [ ] **Model loading + Inductor compile-time optimisation** — characterise and reduce
  cold-start cost on each cluster (SPECIFICATIONS.md §9.3): collect measured per-component
  timings (weight load, graph capture, Inductor compile) and identify reductions via
  weight-load strategy, CUDA-graph capture sizes, eager-vs-compile trade-offs.
- [ ] **Inductor compilation cache persistence** — explore persisting Inductor compilation
  artifacts across restarts to skip the cold-compile cost on every fresh start (vLLM:
  `--compilation-config '{"local_cache_dir": "..."}'`; equivalents on other backends). When
  supported, expose the path as a `BackendConfig` field in SPECIFICATIONS.md §15.2.
  Validate that restored caches produce identical runtime behaviour to fresh compiles.
- [ ] **Trim CUDA graph capture sizes** to match actual concurrency ceiling (~28 slots
  for GH200 70B at 25K context); the current default capture range 1–512 is wasteful,
  ~32 suffices.

### Backend feature experiments

- [ ] **Test `--kv-offloading-size 400`** for GH200 KV extension via Grace DRAM
  (§15.2, §16.1). Implementation complete; not yet run.
- [ ] **Session affinity experiment** (§11.4, §13.3 `session_idx`) — compare random vs
  `session_affinity` routing for multi-instance deployments to quantify prefix-cache
  benefit in production. Now first-class: §13.3 carries `session_idx` and `turn_idx`.
- [ ] **NIXL disaggregated prefill/decode** — vLLM v1 startup logs show *"NIXL is
  available"*; configuration via `--kv-transfer-config` (BackendConfig field
  `kv_transfer_config`, JSON string) not yet implemented or run. When wired up,
  re-add the field to §15.2 as the configuration surface.
- [ ] **LMCache KV-offloading backend** — evaluate `--kv-offloading-backend=lmcache` as
  an alternative to the v1 default `native` (§15.2 `kv_offloading_backend`). Goal: quantify
  bandwidth + concurrency trade-offs vs `native` on Grace DRAM. When wired up, re-add
  `"lmcache"` to the §15.2 `kv_offloading_backend` notes.
- [ ] **DeepSeek `thinking_mode` as BackendConfig knob** — DeepSeek-V4-Pro exposes
  three reasoning-effort modes (Non-think / Think High / Think Max) via the
  `thinking_mode` runtime parameter. Wire it as a sweepable BackendConfig field
  (SPECIFICATIONS.md §15.2 vLLM and SGLang subsections — DeepSeek-V4-Pro runs on both)
  so an experiment can compare TTFT / decode / cost across modes on the same model.
  Note: *Think Max* requires `max_model_len ≥ 384K`; the sweep must clamp combinations
  accordingly. See §8.2 *Models under test*.
- [ ] **Hardware elasticity / auto-scaling experiments** — measure time-to-scale-up,
  in-flight request loss during scale events, and pre-warmed-pool sizing trade-offs
  ("Elasticity requirements for CSCS vClusters"). Needs: auto-scale-injection mechanism
  during a sweep, scale-event observation hooks, pre-warmed-pool semantics. When wired
  up, re-add the row to §15 *Features Under Test*.

### Characterisation

- [ ] **Characterise pre-check reference values** for `tools/system_prechecks_reference.yaml`
  on each cluster (clariden GH200, bristen A100, beverin MI300A, breithorn gh200) — the
  TBD placeholders in §7.3 must be replaced with measured medians plus tolerances before
  the foundation gate (§7.4) becomes enforceable.

## Candidate models

v1 focuses on **Apertus-70B** (with **Apertus-8B** as its speculative-decoding draft)
and **Kimi-K2.6** only. Other models stay here until they come under active measurement,
at which point they are promoted into SPECIFICATIONS.md §8.2 (planner-template +
benchmark-YAML change per §8.2).

- [ ] **DeepSeek-V4-Pro** (`deepseek-ai/DeepSeek-V4-Pro`) — target. 1,048,576-token (1M)
  context; MoE, 1.6 T total / 49 B activated (expert routing exercises §15.1 *MoE expert
  routing*). Three reasoning modes *Non-think* / *Think High* / *Think Max* via the
  `thinking_mode` runtime parameter (*Think Max* needs context ≥ 384 K — clamp
  `max_model_len` accordingly); DeepSeek custom encoding (`encoding_dsv4`; `<think>` /
  `</think>` delimiters). License MIT; recommended sampling `temperature=1.0`,
  `top_p=1.0`; precision FP4 (MoE expert params) + FP8 (other) mixed. Scenarios:
  `agentic-coding`, `chat-short-turns`, `long-context-followup`. See also the
  `thinking_mode` BackendConfig-knob item above.
- [ ] **GLM-5.1** (Zhipu AI) — 202K context, "rumination" multi-iteration self-revision,
  long autonomous loops (up to 8 h), unified multimodal pipeline, multi-step agentic
  tool use.

## Docker Image Builds

- [ ] Automate image builds — the image builds green via manual `sbatch` from a
  login node; automated submission is deferred. See IMPLEMENTATION_PLAN.md M5.
- [ ] Cross-rebuild layer cache — the per-phase Containerfile caches each phase,
  but with `graphroot` on `/dev/shm` (wiped at job end) the cache persists only
  within an allocation. Iterate interactively; cross-job caching would need a
  persistent (capstor) graphroot.
- [ ] **Embed a library-version manifest in built images** (§8.1) — replace the
  removed build-args provenance: record the resolved versions of CUDA, NCCL,
  libfabric, CXI, NVSHMEM, vLLM, etc. inside the image (e.g. a
  `/opt/alps/env/alps-versions.env` queryable at runtime and surfaced in the
  experiment provenance).
- [ ] **ROCm / RCCL engine image for `beverin` (MI300A)** — the AMD equivalent of the
  NVIDIA engine image: extend a ROCm vLLM base with the HPC network stack (RCCL +
  rccl-tests instead of NCCL/nccl-tests, ROCm instead of CUDA), still over Slingshot 11
  / CXI. The launch path (CXI hook disabled, `--network=disable_rdzv_get`, PMIx) is
  CXI-level and identical to the NVIDIA image — only the build phases differ. Add the
  ROCm image tree + the rccl-tests pre-check path (§7 already references
  `ROCm/rccl-tests`). Fits the image-folder restructuring (NVIDIA vs AMD).

## Prompt / Dataset Generation

- [ ] **Multimodality** — v1 is text-only (SPECIFICATIONS.md §10.5). Add modalities in
  order: **image** first (fixed corpus paired with text prompts, per-image token-cost
  accounting, `image_corpus` source kind), then **audio** and **video** (paired corpora,
  per-second / per-clip token-cost accounting, `audio_corpus` / `video_corpus` source
  kinds). Each step also requires per-modality length distributions in the dataset
  generator and registry-load-time acceptance of the corresponding `modalities: [...]`
  declaration. A per-model per-image token-cost fallback table (probably alongside the
  tokenizer ID in a model-info registry) needs a structured home before image lands.
- [ ] **Confirm scenario-registry replaces `dataset_source`** (SPECIFICATIONS.md §10) — the
  previous design had a top-level `dataset_source` knob; the current spec embeds source
  choice inside the scenario registry. Confirm this is the intended end state (no separate
  `dataset_source` at the benchmark-YAML level beyond `scenario_mix[].source_overrides`).
- [ ] **`source_overrides` schema per source kind** (§10.4) — enumerate the keys each source
  accepts (e.g. LongBench task subset, COCO split, reasoning-trace dataset name). The
  field is declared but its per-source schema is not specified.
- [ ] **Alternative chat corpora** (§10.5) — add `lmsys_chat` (lmsys/lmsys-chat-1m) and
  `oasst1` (OpenAssistant/oasst1) as additional real-chat source kinds alongside
  `wildchat`. LMSYS-Chat-1M brings per-message source-model identity (Vicuna / Llama /
  GPT family) — useful for splitting chat scenarios by source-model family. OASST1
  brings a smaller, fully-open (Apache-2.0) tree-structured conversation corpus.
- [ ] **Validate provisional `followup_input_length` values** (§10.3) —
  `long-context-followup` (~300-token follow-ups) and `agentic-coding` (~1.5K
  heavy-tailed tool results) carry PROVISIONAL follow-up length distributions;
  validate against telemetry alongside the think-time defaults below.
- [ ] **Think-time distribution defaults** (§10.3) — validate the provisional
  `think_time_ms` values now in the registry (`chat-short-turns` ~1.5 s reading delay,
  `long-context-followup` ~4 s, `agentic-coding` ~3 s heavy-tailed tool-execution gap —
  marked PROVISIONAL in the YAML) against real telemetry, and document the validated
  ranges per scenario.
- [ ] **Scenario-registry revision pinning for reproducibility** (§10.1) — define how the
  scenario-registry revision is "recorded alongside" the dataset: git SHA on the
  experiment row? content hash of the YAML file? Pick a storage location and mechanism.
- [ ] **Pin HF dataset revisions** (§10.8) — the WildChat / LongBench / gsm8k loaders
  read the dataset repos at HEAD; pin a `revision=` per experiment (recorded with the
  manifest) so pools regenerate identically across time and machines.
- [ ] **Additional reasoning-trace datasets** (§10.5) — `_REASONING_TRACE_DATASETS`
  supports gsm8k; add MATH, AIME, and R1-distill traces (field mapping + licence check
  each).
- [ ] **Precise agentic / tool-calling measurement** — v1 approximates agentic workloads
  as multi-turn sessions with bursty fan-out (high `turns_per_session`, mixed output
  sizes) so that the operator can derive supportable-user-count from the SLO-attained
  rate λ* (SPECIFICATIONS.md §12.4, §14.1). The
  precise approach was specified in earlier drafts of §10.7 / §12.4 / §13.3 / §13.7 and
  is kept here so the work can resume from the current thinking. When picked up,
  re-introduce:
  - **Tool registry** (`tools/tools/<tool-name>.yaml`) — per-tool JSON schema,
    `result_size` distribution, `result_content_source` for synthesis (`synthetic` /
    `longbench` / `static`).
  - **`tool_registry` source kind** in §10.5 — drives agentic scenarios from the
    catalog + task templates.
  - **Fan-out template DSL** — per-scenario state machine declaring turn types
    (`think`, `tool_call`, `tool_result`), allowed transitions, and cycle-count
    distribution. Open question on grammar: inline YAML state-machine in the scenario
    entry, a small Python DSL, or replay-only from recorded agent traces (SWE-Bench
    Verified, τ-bench, …).
  - **Schema-constrained decoding** — load generator requests structured output
    against the tool's JSON schema when the model emits a tool call; per-request
    `structured_output_valid` recorded.
  - **`agent_tasks` table** — first-class task-level metrics distinct from session
    metrics: `task_invocations`, `task_tool_calls_emitted`, `task_tool_calls_valid`,
    `task_input_tokens_total`, `task_output_tokens_total`, `task_success`,
    `task_e2e_ms`. Linked from `requests.agent_task_id`.
  - **Bimodal output distribution as first-class** — distinct tiny-tool-call vs
    large-think output sampling, per-tool result-content synthesis. Currently v1
    approximates by widening the output_length distribution.
  - **Per-tool `result_content_source` synthesis mechanics** — how a tool-result body
    of a sampled token length is produced from each source kind.
  - **`agent_task_id` + `structured_output_valid` columns** in `requests` (§13.3).
  - **§15 *Features Under Test*** — re-add the "Schema-constrained decoding (JSON / XML)"
    row.

## Metrics & Analysis

- [ ] **Per-class `sessions_per_user_per_hour` defaults** — the supportable-users
  estimate (SPECIFICATIONS.md §14.1) needs a defensible default per scenario class
  (agentic-coding tasks/dev/hour, chat sessions/user/hour); currently the notebook
  parameters have no recommended values.
- [ ] **Heavy-tailed (Pareto) arrival process** — v1 keeps only `poisson` + `burst_mmpp` in
  `arrival_process` (SPECIFICATIONS.md §11.3). The distinctive signal Pareto adds (prefix-
  cache aging across multi-minute idle gaps followed by a cold-cache burst) is approximated
  by `burst_mmpp` configured with a short on-phase and a long off-phase. Promote Pareto to
  a first-class process if a future scenario needs the heavy tail explicitly (or if MMPP
  approximation turns out to materially mis-rank deployments for cache-aging-sensitive
  workloads).

## Quality Evaluation

- [ ] **Harder eval suites** (SPECIFICATIONS.md §12.5) — extend Stage B beyond
  GSM8K / GPQA-Diamond (both approach saturation on frontier models): MATH-500, HLE,
  SWE-bench-style task suites. Converges with the README roadmap item on
  task-efficiency evaluation.
- [ ] **Quality under load at λ\*** — v1 Stage B uses eval-concurrency as the load
  proxy; replaying background sweep traffic at exactly λ\* while grading would measure
  quality at the SLO operating point proper.
- [ ] **Thinking-model answer parsing** (§12.5) — lm-eval needs reasoning-delimiter
  handling (e.g. DeepSeek `<think>` / `</think>`) before Stage B grades thinking modes
  correctly; InferenceX carries an equivalent patch.
- [ ] **Per-model Stage-A floor tuning** (§12.5) — the default 0.5 GSM8K floor is a
  blunt rubbish detector; tune per model once first gate runs land.

## Infrastructure Expansion

- [ ] Add `beverin` (AMD MI300A nodes) as a deployment target
- [ ] Support systems outside CSCS as deployment targets
- [ ] **Prefill disaggregation experiment on MI300A** — scientifically interesting, operationally risky (cross-vendor KV-transfer path over Slingshot is the dominant unknown). Use long prompts + moderate/long outputs (disaggregation adds little for short prompts due to KV-transfer overhead). Run in this order:
  1. GH prefill + GH decode, same node/fabric — best-case vLLM P/D baseline
  2. MI300A prefill + MI300A decode — ROCm/vLLM baseline
  3. GH prefill + MI300A decode over Slingshot — cross-vendor penalty
  4. MI300A prefill + GH decode — validate whether direction asymmetry matters
  5. Monolithic GH vs monolithic MI300A — confirm P/D beats simpler serving

  Metrics to capture:
  | Metric | Why it matters |
  |---|---|
  | TTFT p50/p95/p99 | Prefill-side win |
  | TPOT / ITL p50/p95/p99 | Decode smoothness |
  | KV transfer latency | Core tax of disaggregation |
  | Effective KV bandwidth | Whether Slingshot is limiting |
  | GPU utilization by phase | Whether resources are actually specialized |
  | Goodput under SLO | The only metric that really matters |
  | Failure/retry behavior | KV state is now distributed state |

## Reporting & Plots

- [ ] Show experiment duration (minutes) on each plot panel, not just request count N
- [ ] Show distributions in every plot
- [ ] Include all collected percentiles (p50/p75/p90/p95/p99) in reports; initially only p90 visible, others commented out for the user to enable
- [ ] Each experiment directory must contain all artifacts used to run it (Dockerfiles, sbatch scripts, Kubernetes YAML)
