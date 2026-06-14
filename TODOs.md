# TODOs

## Architecture & Engine

- [ ] Define which files and folders are immutable (cannot be modified when running experiments)
- [ ] Structure the tool modularly so the core engine is rarely (ideally never) modified; new capabilities added as modules
- [ ] **M7 launcher cluster validation (E1/E5)** — `SlurmEngineLauncher` / `K8sEngineLauncher`
  (`tools/benchmarker/launchers.py`) are exercised only against the mock in
  `test_orchestrator.py`. On-cluster, validate: nested `sbatch`/`squeue %N` node discovery
  (incl. the fast-aborting-job race where the node leaves the queue before discovery),
  `scancel`/`kubectl delete` teardown, engine-log path resolution, and `is_alive()`
  state mapping. Also: multi-instance readiness currently `asyncio.gather`s `_await_ready`
  without cancelling siblings on the first abort (harmless for the single-instance v1
  path, leaves pending tasks for DP/replica deployments), and `model_load_total_s` is the
  coarse submit→ready wait (scheduling + pre-check + load) with the precise breakdown in
  the parsed `model_load_*` components — revisit if a cleaner total is needed.
- [ ] **M8 Coordinator cluster validation (E1/E5)** — the `tools/coordinator/` logic is
  unit-tested vs `FakeClusterBackend` only. At E1 validate the assistant-driven SLURM path
  end-to-end via the FirecREST MCP (stage → submit → monitor → §8.4 gate surfacing →
  staged-download → merge → §7 teardown). **K8s engine target (E5):** the Benchmarker is
  always SLURM (§2) and deploys the K8s engine itself via `kubectl`; the open mechanics are
  (i) an engine endpoint reachable from the SLURM Benchmarker (NodePort / LB / Ingress, not
  the in-cluster `*.svc` DNS the launcher returns today), (ii) a `kubectl` context + which
  SLURM cluster hosts the Benchmarker, and (iii) results-PVC / `benchmark_config` staging.
  There is no Coordinator-side K8s backend (removed — the Benchmarker owns engine
  deploy/teardown; orphans are the Cleaner's job). Also: the
  monitor loop has no overall wall-clock timeout (relies on job status + resume); add one
  if a stuck-PENDING pod/job proves a problem.
- [ ] **M10 Cleaner — JFrog backend + cluster validation (E1/E5)** — `tools/cleaner.py`
  implements the `KubectlCleanerBackend` (K8s) and the pure `identify()` policy (tested over
  all three §7.7 resource classes incl. JFrog), but the **JFrog tag discovery/deletion
  backend** (`jf rt search`/`del` or the Artifactory REST API, filtered by the benchmark tag
  prefix) is not built — wire it alongside the M5 JFrog push work. SLURM scratch discovery is
  assistant-driven via the FirecREST MCP (`scratch_candidates()` from a `list_files` listing +
  `squeue` for active-job run-ids); validate identify→prune end-to-end on real orphaned
  resources at E1/E5. `keep_recent_jfrog` is currently global, not per-repository.

## Experiment Execution

- [ ] **E1-surfaced follow-ups (2026-06-14)** — found driving the first clariden run:
  - **Primer must cap its prompt to `max_model_len`** (§10.3, `load_gen/readiness.py`): the
    primer sends a fixed 20K-token prompt; with `max_model_len=16384` the engine returned
    http_400 and the primer warned (non-fatal, but it didn't warm the inductor compile).
    Clamp `prompt_tokens` to the served context (read from the engine, or pass it in).
  - **M7 re-run into a dirty run dir crashes** (`orchestrator._persist_experiment`): a
    pre-existing `run_<id>.db` makes the plain `INSERT` into `experiments` hit a UNIQUE
    constraint. Normal flow uses a fresh run_id per run, but make it robust — clear/replace
    the run's rows, or have the Coordinator ensure a clean run dir before submit.
  - **§8 pre-checks need an image that pre-ships the MPI/NCCL build toolchain** (§8.2,
    `prechecks/build-nccl-tests.sh`): the CSCS Container Engine runs the container non-root,
    so the `apt-get` fallback can't install missing build tools and `nccl-tests` aborts on
    `mpi.h: No such file or directory` (seen at E1 on stock `vllm-openai:0.22.1`). The
    worktree set `skip_system_prechecks: true` for stock vendor images; the repo-built Alps
    netstack image (§9.1) pre-ships the toolchain and runs §8 via the MPI-less single-node
    `nccl-tests` flavor. Exercise §8 against the Alps image (single-node now, multi-node at
    E2) and drop the skip once the run is on the netstack image.
- [ ] Run NCCL benchmarks (using the same Docker images) before inference benchmarks
- [ ] Support testing endpoints provided via URL (not only SLURM and Kubernetes deployments)

### Cold-start optimisation experiment groups

- [ ] **Model loading + Inductor compile-time optimisation** — characterise and reduce
  cold-start cost on each cluster (SPECIFICATIONS.md §10.3): collect measured per-component
  timings (weight load, graph capture, Inductor compile) and identify reductions via
  weight-load strategy, CUDA-graph capture sizes, eager-vs-compile trade-offs.
- [ ] **Inductor compilation cache persistence** — explore persisting Inductor compilation
  artifacts across restarts to skip the cold-compile cost on every fresh start (vLLM:
  `--compilation-config '{"local_cache_dir": "..."}'`; equivalents on other backends). When
  supported, expose the path as a `BackendConfig` field in SPECIFICATIONS.md §16.2.
  Validate that restored caches produce identical runtime behaviour to fresh compiles.
- [ ] **Trim CUDA graph capture sizes** to match actual concurrency ceiling (~28 slots
  for GH200 70B at 25K context); the current default capture range 1–512 is wasteful,
  ~32 suffices.

### Backend feature experiments

- [ ] **Test `--kv-offloading-size 400`** for GH200 KV extension via Grace DRAM
  (§16.2, §17.1). Implementation complete; not yet run.
- [ ] **Session affinity experiment** (§12.4, §14.3 `session_idx`) — compare random vs
  `session_affinity` routing for multi-instance deployments to quantify prefix-cache
  benefit in production. Now first-class: §14.3 carries `session_idx` and `turn_idx`.
- [ ] **NIXL disaggregated prefill/decode** — vLLM v1 startup logs show *"NIXL is
  available"*; configuration via `--kv-transfer-config` (BackendConfig field
  `kv_transfer_config`, JSON string) not yet implemented or run. When wired up,
  re-add the field to §16.2 as the configuration surface.
- [ ] **LMCache KV-offloading backend** — evaluate `--kv-offloading-backend=lmcache` as
  an alternative to the v1 default `native` (§16.2 `kv_offloading_backend`). Goal: quantify
  bandwidth + concurrency trade-offs vs `native` on Grace DRAM. When wired up, re-add
  `"lmcache"` to the §16.2 `kv_offloading_backend` notes.
- [ ] **DeepSeek `thinking_mode` as BackendConfig knob** — DeepSeek-V4-Pro exposes
  three reasoning-effort modes (Non-think / Think High / Think Max) via the
  `thinking_mode` runtime parameter. Wire it as a sweepable BackendConfig field
  (SPECIFICATIONS.md §16.2 vLLM and SGLang subsections — DeepSeek-V4-Pro runs on both)
  so an experiment can compare TTFT / decode / cost across modes on the same model.
  Note: *Think Max* requires `max_model_len ≥ 384K`; the sweep must clamp combinations
  accordingly. See §9.2 *Models under test*.
- [ ] **Hardware elasticity / auto-scaling experiments** — measure time-to-scale-up,
  in-flight request loss during scale events, and pre-warmed-pool sizing trade-offs
  ("Elasticity requirements for CSCS vClusters"). Needs: auto-scale-injection mechanism
  during a sweep, scale-event observation hooks, pre-warmed-pool semantics. When wired
  up, re-add the row to §16 *Features Under Test*.

### Characterisation

- [ ] **Characterise pre-check reference values** for `tools/system_prechecks_reference.yaml`
  on each cluster (clariden GH200, bristen A100, beverin MI300A, breithorn gh200) — the
  TBD placeholders in §8.3 must be replaced with measured medians plus tolerances before
  the foundation gate (§8.4) becomes enforceable. **Done so far:** clariden "4× GH200, 1 node"
  NCCL all_reduce/all_gather/alltoall (E2a, mean of 2 runs). **Re-characterise next phase:**
  the clariden capstor `Sequential read` (0.063 GB/s = 62.9 MB/s) was a single sample taken
  while capstor was under general slowness — replace with a healthy-mount median (a few
  samples; consider `lfs getstripe` / stripe-count sensitivity, and an iopsstor/flash sample).
  NVSHMEM rows stay TBD until the dedicated multi-task step lands (see below).
- [ ] **Dedicated NVSHMEM pre-check srun step** (§8.1) — NVSHMEM perftest is multi-PROCESS
  (1 PE per task, bootstrapped by SLURM's PMIx; CSCS guidance:
  docs.cscs.ch/software/communication/nvshmem), so it **cannot** run inside the engine's
  single-task `srun` session. Confirmed at E2a (2026-06-14): in-session it collapses to
  npes=1 — `alltoall` reports busbw≡0, `put_bw` aborts "requires exactly two processes".
  `run-nvshmem.sh` now skips cleanly in a single-task step and `grade.py` skips degenerate
  npes=1 results, so no bogus number is recorded — but NVSHMEM is therefore **uncharacterised**.
  To characterise it, add a pre-engine step to `engine.sbatch.j2`:
  `srun --ntasks-per-node=N --mpi=pmix --environment=engine.toml bash run-nvshmem.sh`
  (N PEs for `alltoall_latency`, exactly 2 for `shmem_put_bw` — two steps, set `NVSHMEM_TESTS`
  per step), writing captures to `PRECHECK_OUT`, with `grade.py` run after. Keep the host CXI /
  `aws_ofi_nccl` hooks **disabled** — the self-contained Alps net image (§9.1) provides its own
  libfabric/cxi; verify on-image that NVSHMEM bootstraps without host injection (intra-node
  NVLink P2P first, then inter-node). Also rework the `NVSHMEM_REQUIRED=1` path: a single-PE
  engine session must surface as "needs a dedicated step", not silently pass the `|| true`.

## Candidate models

v1 focuses on **Apertus-70B** (with **Apertus-8B** as its speculative-decoding draft)
and **Kimi-K2.6** only. Other models stay here until they come under active measurement,
at which point they are promoted into SPECIFICATIONS.md §9.2 (planner-template +
benchmark-YAML change per §9.2).

- [ ] **DeepSeek-V4-Pro** (`deepseek-ai/DeepSeek-V4-Pro`) — target. 1,048,576-token (1M)
  context; MoE, 1.6 T total / 49 B activated (expert routing exercises §16.1 *MoE expert
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

- [x] Automate image builds — done: `tools/images/build.sh` drives `podman build` +
  `podman push` over ssh + `srun --overlap` into a held allocation (manifest-driven
  build-args; writes a digest-pinned EDF consumed by `sanity.sbatch`). See
  SPECIFICATIONS.md §9.1. First green image: `vllm:0.22.1-alps.net.v1`.
- [ ] Cross-rebuild layer cache — the per-phase Containerfile caches each phase,
  but with `graphroot` on `/dev/shm` (wiped at job end) the cache persists only
  within an allocation. Iterate interactively; cross-job caching would need a
  persistent (capstor) graphroot.
- [ ] **Embed a library-version manifest in built images** (§9.1) — replace the
  removed build-args provenance: record the resolved versions of CUDA, NCCL,
  libfabric, CXI, NVSHMEM, vLLM, etc. inside the image (e.g. a
  `/opt/alps/env/alps-versions.env` queryable at runtime and surfaced in the
  experiment provenance).
- [ ] **ROCm / RCCL engine image for `beverin` (MI300A)** — the AMD equivalent of the
  NVIDIA engine image: extend a ROCm vLLM base with the HPC network stack (RCCL +
  rccl-tests instead of NCCL/nccl-tests, ROCm instead of CUDA), still over Slingshot 11
  / CXI. The launch path (CXI hook disabled, `--network=disable_rdzv_get`, PMIx) is
  CXI-level and identical to the NVIDIA image — only the build phases differ. Add the
  ROCm image tree + the rccl-tests pre-check path (§8 already references
  `ROCm/rccl-tests`). Fits the image-folder restructuring (NVIDIA vs AMD).

## Prompt / Dataset Generation

- [ ] **Multimodality** — v1 is text-only (SPECIFICATIONS.md §11.5). Add modalities in
  order: **image** first (fixed corpus paired with text prompts, per-image token-cost
  accounting, `image_corpus` source kind), then **audio** and **video** (paired corpora,
  per-second / per-clip token-cost accounting, `audio_corpus` / `video_corpus` source
  kinds). Each step also requires per-modality length distributions in the dataset
  generator and registry-load-time acceptance of the corresponding `modalities: [...]`
  declaration. A per-model per-image token-cost fallback table (probably alongside the
  tokenizer ID in a model-info registry) needs a structured home before image lands.
- [ ] **Confirm scenario-registry replaces `dataset_source`** (SPECIFICATIONS.md §11) — the
  previous design had a top-level `dataset_source` knob; the current spec embeds source
  choice inside the scenario registry. Confirm this is the intended end state (no separate
  `dataset_source` at the benchmark-YAML level beyond `scenario_mix[].source_overrides`).
- [ ] **`source_overrides` schema per source kind** (§11.4) — enumerate the keys each source
  accepts (e.g. LongBench task subset, COCO split, reasoning-trace dataset name). The
  field is declared but its per-source schema is not specified.
- [ ] **Alternative chat corpora** (§11.5) — add `lmsys_chat` (lmsys/lmsys-chat-1m) and
  `oasst1` (OpenAssistant/oasst1) as additional real-chat source kinds alongside
  `wildchat`. LMSYS-Chat-1M brings per-message source-model identity (Vicuna / Llama /
  GPT family) — useful for splitting chat scenarios by source-model family. OASST1
  brings a smaller, fully-open (Apache-2.0) tree-structured conversation corpus.
- [ ] **Validate provisional `followup_input_length` values** (§11.3) —
  `long-context-followup` (~300-token follow-ups) and `agentic-coding` (~1.5K
  heavy-tailed tool results) carry PROVISIONAL follow-up length distributions;
  validate against telemetry alongside the think-time defaults below.
- [ ] **Think-time distribution defaults** (§11.3) — validate the provisional
  `think_time_ms` values now in the registry (`chat-short-turns` ~1.5 s reading delay,
  `long-context-followup` ~4 s, `agentic-coding` ~3 s heavy-tailed tool-execution gap —
  marked PROVISIONAL in the YAML) against real telemetry, and document the validated
  ranges per scenario.
- [ ] **Scenario-registry revision pinning for reproducibility** (§11.1) — define how the
  scenario-registry revision is "recorded alongside" the dataset: git SHA on the
  experiment row? content hash of the YAML file? Pick a storage location and mechanism.
- [ ] **Pin HF dataset revisions** (§11.8) — the WildChat / LongBench / gsm8k loaders
  read the dataset repos at HEAD; pin a `revision=` per experiment (recorded with the
  manifest) so pools regenerate identically across time and machines.
- [ ] **Additional reasoning-trace datasets** (§11.5) — `_REASONING_TRACE_DATASETS`
  supports gsm8k; add MATH, AIME, and R1-distill traces (field mapping + licence check
  each).
- [ ] **Precise agentic / tool-calling measurement** — v1 approximates agentic workloads
  as multi-turn sessions with bursty fan-out (high `turns_per_session`, mixed output
  sizes) so that the operator can derive supportable-user-count from the SLO-attained
  rate λ* (SPECIFICATIONS.md §13.4, §15.1). The
  precise approach was specified in earlier drafts of §11.7 / §13.4 / §14.3 / §14.7 and
  is kept here so the work can resume from the current thinking. When picked up,
  re-introduce:
  - **Tool registry** (`tools/tools/<tool-name>.yaml`) — per-tool JSON schema,
    `result_size` distribution, `result_content_source` for synthesis (`synthetic` /
    `longbench` / `static`).
  - **`tool_registry` source kind** in §11.5 — drives agentic scenarios from the
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
  - **`agent_task_id` + `structured_output_valid` columns** in `requests` (§14.3).
  - **§16 *Features Under Test*** — re-add the "Schema-constrained decoding (JSON / XML)"
    row.

## Metrics & Analysis

- [ ] **Per-class `sessions_per_user_per_hour` defaults** — the supportable-users
  estimate (SPECIFICATIONS.md §15.1) needs a defensible default per scenario class
  (agentic-coding tasks/dev/hour, chat sessions/user/hour); currently the notebook
  parameters have no recommended values.
- [ ] **Heavy-tailed (Pareto) arrival process** — v1 keeps only `poisson` + `burst_mmpp` in
  `arrival_process` (SPECIFICATIONS.md §12.3). The distinctive signal Pareto adds (prefix-
  cache aging across multi-minute idle gaps followed by a cold-cache burst) is approximated
  by `burst_mmpp` configured with a short on-phase and a long off-phase. Promote Pareto to
  a first-class process if a future scenario needs the heavy tail explicitly (or if MMPP
  approximation turns out to materially mis-rank deployments for cache-aging-sensitive
  workloads).

## Quality Evaluation

- [ ] **M11 lm-eval validation + grader limits (E1)** — `LmEvalBackend` (`quality_eval/
  lm_eval_backend.py`) is unit-tested only via the `BuiltinEvalBackend`; the lm-eval
  `local-chat-completions` model-args and the results-dict metric parsing are provisional
  (seeded from documented shapes) and must be validated against the pinned lm-eval on the
  M5 Benchmarker image at E1 — capture a `results.json` fixture and pin the parse. Also:
  `BuiltinEvalBackend` grades via the load-gen streaming client, which sends no
  `temperature`/`top_p` (server defaults apply), so suite-defined sampling is lm-eval's job;
  and both stages target `instances[0]` only (multi-instance routing / `instance_id=NULL`
  per §14.9 deferred).

- [ ] **Harder eval suites** (SPECIFICATIONS.md §13.5) — extend Stage B beyond
  GSM8K / GPQA-Diamond (both approach saturation on frontier models): MATH-500, HLE,
  SWE-bench-style task suites. Converges with the README roadmap item on
  task-efficiency evaluation.
- [ ] **Quality under load at λ\*** — v1 Stage B uses eval-concurrency as the load
  proxy; replaying background sweep traffic at exactly λ\* while grading would measure
  quality at the SLO operating point proper.
- [ ] **Thinking-model answer parsing** (§13.5) — lm-eval needs reasoning-delimiter
  handling (e.g. DeepSeek `<think>` / `</think>`) before Stage B grades thinking modes
  correctly; InferenceX carries an equivalent patch.
- [ ] **Per-model Stage-A floor tuning** (§13.5) — the default 0.5 GSM8K floor is a
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

## Code quality & test coverage

Surfaced by the post-merge audit (2026-06-14). (The E1-run-surfaced primer-cap item lives
under *Experiment Execution → E1-surfaced follow-ups*.)

- [ ] **Unit/dry-run test for `tools/benchmarker/launchers.py`** — the real SLURM/K8s
  engine-spawn logic is only exercised via `MockLauncher` in `test_orchestrator`; add a
  construction/dry-run test asserting the rendered `srun`/`kubectl` invocation shape
  (distinct from the E1/E5 cluster validation already tracked under *Architecture & Engine*).
- [ ] **Make `tools/pre-flight-checks.py` importable for testing** — the hyphenated
  filename blocks `import`, so the §4 pre-flight logic has no unit test. Rename to
  `pre_flight_checks.py` (keep a thin CLI shim if the hyphenated invocation is relied on).
- [ ] **Direct unit test for `load_gen/client.py`** — the streaming/SSE `_failure`
  classification (§13.1) is only covered indirectly via the mock server; isolate it.
- [ ] **Deferred low-priority simplifications** (audit, low value / refactor risk on paths
  validated only on-cluster): (a) shared reader for the `prechecks/results.json` contract
  consumed by both `orchestrator` (inline gate enforcement) and `coordinator/policy`
  (observe/surface) — today they parse the same `rows`/`status`/`gate_exit_code` keys
  independently; (b) factor the duplicated TTFT/TPOT math in `load_gen/client.py` success
  vs `_failure` paths into one helper.
- [ ] **Multi-node pre-check reference scopes** (§8.3/§8.4) — `find_reference` matches the
  planner's `precheck_scope` string exactly, but `system_prechecks_reference.yaml` only
  enumerates 1- and 2-node scopes, so larger multi-node deployments grade as
  "pass (informational)" silently. Enumerate every node-count scope a config can render
  (or interpolate) when populating the reference values.

## Documentation & spec consistency

Minor SPECIFICATIONS.md self-consistency follow-ups from the post-merge audit (2026-06-14):

- [ ] **`request_timeout_s` scope** — §12.2/§13.1 describe it as a TTFT cutoff; §14.3's
  `success` column describes a whole-request completion cutoff. State once whether it bounds
  TTFT or end-to-end and make the other references consistent.
- [ ] **ITL ≡ TPOT terminology** — inter-token latency is called "ITL" in §12.2/§15.1 plot
  labels and "TPOT" (the `tpot_ms` schema field) elsewhere. Standardise on one term, or add
  an explicit "ITL ≡ TPOT (`tpot_ms`)" note.
- [ ] **§14.4 composite-key claim** — §14.4 (`server_stats`) asserts its key "matches"
  §14.5/§14.6, which don't declare matching composite keys. Either declare the keys or soften
  the wording to "shares the run_id/instance_id scoping".
