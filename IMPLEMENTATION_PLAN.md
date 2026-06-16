# Implementation Plan

Build order for the tool specified in `SPECIFICATIONS.md`. The spec says *what*; this
document says *in what order, with what intermediate proof*. It is a living document:
milestones get checked off, re-scoped, or re-ordered as implementation reveals new facts
— any structural change to a milestone goes through the working-agreement process
(`CLAUDE.md` *How we work together*).

**Process status**: all component milestones **M0–M11 are implemented** (laptop halves plus
the cluster-validated pre-check / engine / reports paths). Experiments **E1, E2a, E2b are
✅ done** on `clariden` (walking skeleton + 1-node & 2-node §8 foundation references
populated). Next on the critical path: the **real-scenario capacity runs** — Apertus-70B
single-node (E3a, config drafted) as the lower-risk precursor to the **Kimi-K2.6 flagship
(E3)**, whose Ray-in-image blocker is now resolved in-repo
(`nvidia-gh200-vllm-0.23.0-net.v1`, pending build + JFrog push).

---

## 1. Objective and success criterion

The driving deliverable is the framework's primary v1 question (README, §13.4, §15.1):

> **How many users can a model instance support under a mixed workload of 80%
> agentic-assisted coding + 20% user chats, at declared per-class SLOs — and how do
> backend features move that number?**

The plan is complete when experiment **E3** (below) produces a report containing the
SLO-attained rate λ\*, per-class SLO attainment, and the supportable-users estimate for
the 80/20 mix on `clariden` — and **E5** answers whether the same workload runs slower
on Kubernetes (`breithorn`) than on SLURM.

All spec pillars are in scope — system pre-checks (§8), image building (§9.1),
model-load decomposition (§10.2), platform comparison (§16.1), curated reports (§15.3).
None are descoped.

## 2. Current state

| Asset | Status |
|---|---|
| `SPECIFICATIONS.md`, scenario registry (`tools/scenarios/*.yaml` × 4, incl. `smoke-synthetic`) | Done |
| `tools/pre-flight-checks.py` (§4) | Done |
| `tools/system_prechecks_reference.yaml` (§8.3) | `clariden` **1-node + 2-node rows populated** (E2a/E2b: NCCL all_reduce/all_gather/alltoall + PP-link sendrecv + NVSHMEM alltoall + capstor storage); `breithorn` rows still TBD (E2c) |
| `examples/nccl-tests/` (standalone manual pre-check example) | Adapted into M4 (done); the canonical §8 scripts now live in `tools/benchmarker/prechecks/` |
| `examples/slurm-deployment/` (Apertus-8B vLLM sbatch + EDF) | Done — seed for M6 templates |
| `examples/k8s-deployment/` (deployment/service/ingress/PVC) | Done — seed for M6 templates |
| `examples/docker-images-build/` (Dockerfile) | Superseded by `tools/images/` (M5 done) — retained as a historical monolithic example |
| `tools/common/` (global.yaml §3.3, benchmark-YAML schema + CLI, run-ID §7.2) + `examples/benchmark-configs/mixed-80-20.yaml` + `tools/tests/` (20 tests) | Done (M0, 2026-06-12) |
| `tools/benchmarker/dataset_gen/` (registry loader, seeded sampling, all four §11.5 sources, manifest emitter, offline CLI) — validated against real LongBench / WildChat / gsm8k downloads | Done (M1, 2026-06-12) |
| `tools/benchmarker/load_gen/` (arrival, client + §13.1 taxonomy, session scheduler with §12.2 phase accounting, `server_stats` scraper, readiness/model-load/primer) + `tools/testing/mock_openai_server.py` | Done (M2, 2026-06-12) |
| `tools/common/results_db.py` (seven §14 tables, WAL + single-writer, smoke-mode suppression, NDJSON ingestion) + `tools/benchmarker/hw_sampler.py` (stdlib-only, engine-node placement) | Done (M3, 2026-06-12) |
| `tools/benchmarker/prechecks/` (runner + parsers + §8.3/§8.4 grading) and `tools/planner/` + `tools/templates/` (EDF, engine + benchmarker sbatch, K8s engine manifest) | M4 **cluster-validated at E2** (dedicated multi-rank pre-check step, prebuilt-binary discovery, richer mount-aware storage, model-load dissection); M6 laptop half done + **K8s engine/ingress path landed** (worktree-k8s merge, E5 prep); `sbatch --test-only` / `kubectl --dry-run` at E5 |
| `tools/benchmarker/orchestrator.py` (`run_experiment` phase driver, `EngineLauncher` + `QualityEvaluator`/M11 seams) + `launchers.py` (Slurm/K8s) + `main.py` CLI + `tools/tests/test_orchestrator.py` (7 mock-integration tests) + planner `--deployment-index` wiring | Done (M7, 2026-06-13) — DoD met on the laptop half against the mock server; SlurmEngineLauncher/K8sEngineLauncher subprocess paths (sbatch/squeue/scancel, kubectl) exercised on-cluster at E1/E5 |
| `tools/coordinator/` (`state` resumable run file, `merge` idempotent central-DB merge, `policy` §8.4 gate, `teardown` §7 plan, `backend` ClusterBackend + Fake/Kubectl, `coordinator` phase loop, `main` CLI) + `tools/tests/test_coordinator.py` (11 tests) | Done (M8, 2026-06-13) — deterministic logic + >100MB staged-download round-trip unit-tested vs a fake backend; SLURM FirecREST effects assistant-driven via MCP in-session (decision 5), K8s `kubectl` path headless (staging is E5); live FirecREST validation at E1 |
| `tools/benchmarker/quality_eval/` (`runner` implementing M7's QualityEvaluator seam, `BuiltinEvalBackend` in-process grader, `LmEvalBackend` lm-eval-harness wrapper, `suites`/`base`) + `tools/tests/test_quality_eval.py` (7 tests); M7 `main.py` wires `QualityEvalRunner(LmEvalBackend())` | Done (M11, 2026-06-13) — Stage-A gate (floor pass/fail) + Stage-B comparison (suites × eval-concurrency → `quality_evals` rows) verified vs the mock's canned answers; gate-abort path exercised by `test_orchestrator`. lm-eval invocation/parse provisional → validated at E1; GPQA-Diamond gating documented (decision 8) |
| `tools/reports/` (`analysis` §15.1 computations, `plots` matplotlib figures, `notebook` builder, `render` nbclient executor, `fixtures` known-answer DB) + `experiments/template_report.ipynb` + `reports/STYLE.md` + `tools/tests/test_reports.py` (12 tests) | Done (M9, 2026-06-13) — λ\*/supportable-users/capacity-vs-quality asserted against a crafted fixture; notebook executes headless (PNGs + λ\* captured); validated against the real E1 DB later |
| `tools/cleaner.py` (§7.7 `identify` policy + `prune`, ClusterBackend seam: Kubectl/Fake, `scratch_candidates` for the MCP-driven SLURM path, `reminder_due`) + `tools/tests/test_cleaner.py` (5 tests) | Done (M10, 2026-06-13) — identification + skip policy (model-cache PVC §7.6, recent-N JFrog, active-job scratch, age threshold) and approval-gated pruning tested vs a fake backend; K8s headless; SLURM scratch assistant-driven; JFrog `jf` backend is a follow-up |
| `tools/images/` (multi-image catalogue: Alps netstack `core/{nvidia,amd}/netstack/v1` + variants `nvidia-gh200-vllm-0.22.1-net.v1`, `nvidia-gh200-vllm-0.23.0-net.v1` (Ray baked), `amd-mi300a-vllm-0.23.0-net.v1`; `build.sh` SLURM build → digest-pinned EDF → `sanity.sbatch` acceptance gate) | Done (M5, 2026-06-14) — first Alps vLLM image green on clariden; 0.23.0+Ray + AMD MI300A variants added 2026-06-15 (build/push pending) |

## 3. Build strategy

1. **Local-first**: the dataset generator, load generator, results DB, and report
   notebook are pure Python — built and tested on the laptop against a mock
   OpenAI-compatible server before any cluster time is spent.
2. **Walking skeleton before scale**: the first end-to-end run (E1) uses the
   `smoke-synthetic` scenario (registered for pipeline validation, exempt from the §9.2
   model/scenario pairings; its results are never findings) on Apertus-8B and one GH200
   node — the cheapest legal full-pipeline shakeout before the multi-node MoE bring-up
   that E3 requires.
3. **Manual before automated**: every component is a standalone CLI first (per the
   README operating model); the Coordinator automates an already-working manual path.
4. **Experiments are milestones too**: E1–E5 carry their own definitions of done and
   pass through the working-agreement phases (pre-execution assessment, monitoring,
   results review, adversarial review of results).

## 4. Component milestones

Sizes are relative complexity (S < M < L), not time promises.

### M0 — Scaffolding and benchmark YAML schema (S–M) — ✅ done 2026-06-12

- **Deliverables**: `tools/common/config.py` (typed schema + validation for the full
  benchmark YAML: `deployment_target`, `backend` + version, model, `BackendConfig`
  §16.2, `dataset_config` with `scenario_mix` §11.4, `slos` §13.4, `rate_levels`,
  `arrival_process` §12.3, `routing_strategy` §12.4, timeouts, `output_length_mode`
  §11.6, the `quality_eval` block §13.5, and the §8 pre-check surface: `skip_system_prechecks`, `system_prechecks_on_warn` / `_on_fail`,
  `system_prechecks_timeout_s`, `collective_tests_version`,
  `collective_tests_cache_dir`, `shmem_required`); `tools/common/runid.py` (§7.2); a
  committed canonical example `examples/benchmark-configs/mixed-80-20.yaml`.
- **Validation rules enforced**: weights sum to 1.0; registered scenario names; SLO
  metric/percentile enums; `tensor_parallel_size ≤ gpus_per_node` (§6.1). (Modality
  rejection happens at registry-load time in M1, per §11.5 — not here.)
- **DoD**: pytest green; `python -m tools.common.config <yaml>` accepts the canonical
  example and rejects each violation class with a one-line error.

### M1 — Dataset generator (L) — ✅ done 2026-06-12 (all four sources; traces support gsm8k, further trace datasets + HF revision pinning tracked in TODOs)

- **Deliverables**: `tools/benchmarker/dataset_gen/` — registry loader (rejects
  non-`[text]` modalities at load time, §11.5), mix planner (per-class sub-pools,
  `num_prompts` split ∝ `weight × E[turns_per_session]`), sources in order
  **synthetic → longbench → wildchat → reasoning_trace_replay** (§11.5), session builder
  (`append_delta`, turns, think-time, `followup_input_length` §11.3), per-class seeding (§11.8), global unique
  headers (§11.6), `thinking: true` widening, tokenizer handling (§11.6), source-failure
  abort (§11.1), manifest emitter (§14.7: `mix` / `classes[]` / `run_assumptions`,
  including the derived per-class expected request rates per §12.3 *What λ counts*).
- **Artifact format** (decision embedded here): JSONL prompt pool, one record per turn —
  `{scenario, session_idx, turn_idx, prompt_text, max_tokens}` — plus `manifest.json`.
- **DoD**: byte-identical regeneration with same config (the §11.8 contract, as a test);
  empirical per-class length/turn distributions match declared params within tolerance;
  manifest schema-validates; runs on laptop with no cluster access (synthetic + cached
  HF datasets).

### M2 — Load generator (L) — ✅ done 2026-06-12 (vLLM model-load log regexes seeded from upstream message shapes — re-capture fixtures from the pinned vLLM image at E1)

- **Deliverables**: `tools/benchmarker/load_gen/` — asyncio streaming client
  (forced/natural `output_length_mode` per §11.6, sampled `max_tokens`); arrival
  processes `poisson` + `burst_mmpp`
  scheduling **session starts** (λ semantics per §12.3 *What λ counts*); session modes
  `open_loop` / `sequential` with `think_time_ms` (§11.7); class inheritance per
  session; routing `random` / `session_affinity` (§12.4); warmup / measurement / drain
  phases with sweep-step session accounting (§12.2: session-population warmup, no new
  sessions after measurement end, incomplete-session truncation per §13.2); per-request
  metrics + §13.1 error taxonomy; **per-instance
  `server_stats` scraper** (§14.4: `requests_running`, `requests_waiting`,
  `gpu_cache_pct`, `spec_accept_rate` from the backend's metrics endpoint); readiness
  wait + per-instance model-load parsing (vLLM logs/API first, §10.2/§12.1); inductor
  primer (§10.3) including the primer-missed-target warning.
- **Test harness**: `tools/testing/mock_openai_server.py` — OpenAI-compatible SSE server
  with configurable TTFT/TPOT delays, a mock metrics endpoint, fault injection, and a
  **deterministic canned-answer mode** (fixed responses per prompt pattern — required by
  M11's eval-logic tests), so latency math, scraping, error classification, and grading
  are all verified against ground truth.
- **DoD**: inter-arrival statistical tests pass under fixed seeds (Poisson CV≈1, MMPP
  burst factor as configured); sequential sessions never overlap turns; measured
  TTFT/TPOT against the mock match its configured delays; `server_stats` rows captured
  from the mock metrics endpoint; client-side saturation is detected and reported
  (event-loop lag guard), demonstrating the target λ range is generated without client
  bottleneck.

### M3 — Results DB and hardware sampler (M) — ✅ done 2026-06-12 (DCGM profiling counters emit NULL until wired + fixture-tested on a GH200 node at E1; rocm-smi parse best-effort until beverin)

- **Deliverables**: `tools/common/results_db.py` — the seven §14 tables, column-for-column,
  with a concurrency design (WAL mode + single-writer queue, or per-producer DB files
  merged at finalisation — decided at implementation, asserted by a contention test);
  smoke-test-mode suppression hook (§8.2: results not persisted on pre-check cache
  miss). `tools/benchmarker/hw_sampler.py` — **single-file, stdlib-only** script
  (shells out to DCGM / `nvidia-smi dmon` / `rocm-smi` + `/proc`) so it runs inside
  *any engine container without extra dependencies*; 1 Hz per §13.3; writes NDJSON to
  the run's scratch directory; `NULL` for unexposed signals.
- **Placement (per review H3)**: the sampler runs **on the inference-server nodes**,
  backgrounded inside the engine container session before `exec <engine>` (wired by the
  M6 templates); its NDJSON output is ingested into `hardware_stats` by the Benchmarker
  orchestrator (M7) at finalisation.
- **DoD**: schema asserted against §14 in a test; concurrent-writer test passes; sampler
  runs dependency-free in a bare container image locally (all-`NULL` GPU rows on
  laptop); verified on a GH200 node during E1.

### M4 — System performance pre-checks runner (M) — ✅ done 2026-06-15 (cluster-validated at E2a single-node + E2b multi-node on clariden)

- **Deliverables**: `tools/benchmarker/prechecks/` — a **dedicated multi-rank pre-check
  step** (§8.2, refactored 2026-06-15): an `srun --ntasks-per-node=<gpus_per_node>
  --mpi=pmix` SPMD step (one rank per GPU, PMIx) that **gates** a separate one-task-per-node
  engine step, replacing the old welded `run_system_prechecks && exec <engine>`. Discovers
  the Alps image's **prebuilt** nccl-tests / OSU / NVSHMEM-perftest binaries (no MPI dev to
  build → discover, don't build); runs NCCL collectives (all_reduce / all_gather / alltoall,
  plus `sendrecv` when PP>1) at the real per-GPU topology, multi-PE NVSHMEM
  `alltoall_latency`, and **mount-aware storage** (single-stream O_DIRECT floor +
  parallel-aggregate + buffered/readahead, scope derived from the weights mount). Parses
  into `system_prechecks` rows (§14.6); grades against `tools/system_prechecks_reference.yaml`
  (§8.3–8.4) with warn/fail to the orchestrator (M7) and smoke-test mode on cache miss.
- **DoD**: executes as the dedicated one-rank-per-GPU step on `clariden`; rows persisted;
  references populated → §8.4 gate enforceable (**met at E2a single-node + E2b multi-node**).

### M5 — Images and registry workflow (M, parallel track) — ✅ done 2026-06-14 (first Alps vLLM image `vllm:0.22.1-alps.net.v1` built + 2-node sanity green on clariden); catalogue extended 2026-06-15 (nvidia-gh200 0.23.0 + Ray, AMD MI300A)

- **Deliverables**: `tools/images/` as a multi-image catalogue (§9.1) — a shared,
  version-tagged Alps network stack under `core/<vendor>/netstack/<v>/` (Containerfile +
  per-component build `phases/` + `patches/`) plus thin per-image directories
  (`<vendor>-<backend>-<ver>-net.<n>/` with `manifest.yaml` + `tests/`); `build.sh`
  drives the SLURM/podman build (TODOs *Support building Docker images via SLURM jobs*)
  and the JFrog push with canonical tags + provenance; `sanity.sbatch` is the post-push
  acceptance gate. Images are **self-contained** (hook-disabled) so the same image runs on
  SLURM and K8s. **No separate benchmarker image** — per decision 3 the Benchmarker runs
  from a staged `uv` venv on capstor (`tools/benchmarker/requirements.txt`), not a
  container. The hardware sampler is stdlib-only (M3) and runs in the engine image as-is.
- **DoD**: vLLM image builds reproducibly via SLURM job and pushes to JFrog; EDF
  references it by canonical tag; benchmarker image runs M1 dataset generation on the
  cluster. Networking-library correctness (NCCL ↔ Slingshot/libfabric) is proven by M4
  passing inside this image — the two milestones gate each other.
- **Catalogue (2026-06-15)**: `nvidia-gh200-vllm-0.23.0-net.v1` **bakes Ray into the image**
  (`variant/hooks.d/10-ray.sh`), retiring the E2b interim scratch-`--target` Ray install and
  unblocking the E3 multi-node path; `amd-mi300a-vllm-0.23.0-net.v1` + `core/amd/netstack/v1`
  add the `beverin` (MI300A / ROCm) build. Both **pending build + JFrog push** before use; the
  0.23.0 bump also triggers a §16.2 flag-compat check vs the configs pinned at 0.22.1.

### M6 — Planner (M) — 🚧 laptop half done 2026-06-12; extended 2026-06-15 (`engine.sbatch.j2` now renders the dedicated §8.2 pre-check step + the gated engine step; `render.py` wires `precheck_collectives` / `model_id` + the multi-node `disable_custom_all_reduce` / `enforce_eager` flags; **K8s engine + ingress path landed** via worktree-k8s — Ingress host (§6.2), cached-weights PVC + `model_cache_slug`, DNS-1035 `k8s_slug` name bounding, image-pull secret; `sbatch --test-only` / `kubectl --dry-run=server` validation at E5)

- **Deliverables**: `tools/planner/` + templates `tools/templates/slurm/vllm.edf.j2`,
  `tools/templates/slurm/engine.sbatch.j2`, `tools/templates/benchmarker.sbatch.j2` (the
  Benchmarker is **always SLURM**, §2) and `tools/templates/k8s/engine.yaml.j2` (the K8s
  *engine* manifest, seeded from `examples/k8s-deployment/`); renders the full §14.8
  experiment directory from one benchmark YAML; CLI + Claude-driven paths (§5). **The SLURM
  Benchmarker that drives a K8s engine is now rendered too** (2026-06-16): a K8s deployment
  emits `benchmarker.sbatch` configured with `--endpoint-url <ingress>` (ExternalEndpointLauncher)
  running on the `benchmarker_cluster` (a new `ClusterInfo` field, e.g. breithorn→clariden), plus
  the per-run `/results` PVC (`results_storage_class`). Validated end-to-end at the dual-platform
  smoke (§6.2).
- **DoD**: golden-file render tests; rendered sbatch passes `sbatch --test-only` on
  `clariden`; rendered manifests pass `kubectl apply --dry-run=server` on `breithorn`;
  asserted in tests: time-limit alignment (§6.1), the §8.2
  `run_system_prechecks && exec <engine>` concatenation, and the M3 sampler
  backgrounding in the engine container command.

### M7 — Benchmarker orchestrator (M–L) *(added by review H1)* — ✅ laptop half done 2026-06-13 (run_experiment phase driver + EngineLauncher/QualityEvaluator seams + Slurm/K8s launchers + CLI + 7 mock-integration tests; planner gained `--deployment-index`; nested-sbatch / squeue / kubectl paths validated on-cluster at E1/E5; the M11 quality evaluator slots into the seam via `main.py`)

- **Deliverables**: `tools/benchmarker/main.py` — the cluster-side driver that owns the
  §1 phase sequencing: run dataset generation → **submit the inference deployment(s)
  from within the Benchmarker allocation** (sbatch on SLURM, kubectl on K8s) only after
  the dataset is ready → wait for readiness + primer (§12.1) → **Stage-A quality gate**
  (§13.5, via M11) → run load generation → **Stage-B quality comparison** (§13.5, via
  M11) → finalise the per-run DB (ingest sampler NDJSON into `hardware_stats`, §14.5;
  stage outputs per §6.1). Owns warn/fail propagation from pre-checks to the Coordinator (§8.4 pause +
  `system_prechecks_on_warn` default for non-interactive runs) and smoke-test-mode
  propagation (suppress persistence pipeline-wide, warn at launch and termination,
  §8.2).
- **DoD**: phase ordering asserted by an integration test (engine job not submitted
  until the dataset pool exists); forced pre-check warn pauses and resolves per config;
  smoke-test mode produces no persisted results and two unmissable warnings.

### M8 — Coordinator (L) — ✅ laptop half done 2026-06-13 (resumable state machine, idempotent `run_id`-keyed central-DB merge, §7 teardown on success+failure with best-effort partial-DB salvage, §8.4 gate policy, ClusterBackend seam + FakeClusterBackend + KubectlClusterBackend skeleton, CLI; 11 mock-integration/unit tests incl. the >100MB staged-download round-trip. Per decision 5 the SLURM/FirecREST path is assistant-driven via MCP in-session — no autonomous SLURM backend; K8s headless via kubectl, full K8s staging at E5; live FirecREST submit/monitor/download validated at E1)

- **Deliverables**: `tools/coordinator/` — pre-staging of HF datasets to capstor before
  submission (review LOW); submit (FirecREST for SLURM, kubectl for K8s); monitor loop
  with log streaming and the §8.4 **operator abort/proceed interaction** on pre-check
  warns; **resumable state** (state file per run; reattach to a running experiment after
  laptop sleep / network loss); per-run DB download via the **FirecREST staged-transfer
  path with compression** (direct transfer is capped ~5 MB; per-run DBs will reach
  hundreds of MB — review H5); merge into the centralized results DB; teardown on
  success **and** failure (§7.3–7.6).
- **DoD**: drives E1 end-to-end unattended; a deliberately killed Coordinator resumes
  from its state file and completes monitoring + collection; a deliberately killed run
  still tears down all labelled resources; PVC retention honored (§7.6); a >100 MB
  fixture DB round-trips intact through the staged-transfer download.

### M9 — Reports generator (M) — ✅ done 2026-06-13 (analysis module computes every §15.1 panel — measurement-phase filtering, per-class latency-vs-λ, λ\* / SLO attainment, supportable-users via Little's law, quality + capacity-vs-quality deltas, hardware overlays — asserted against a known-answer fixture; matplotlib plots; the §15.1 template notebook `experiments/template_report.ipynb` executes headless via nbclient, writing report.ipynb + ttft/itl/hardware PNGs; `reports/STYLE.md` bootstrapped; real-DB validation at E1, and **exercised against the real E2a/E2b run DBs 2026-06-15** producing ttft/itl PNGs under `experiments/`)

- **Deliverables**: `experiments/template_report.ipynb` with every §15.1 panel —
  scenario/mix manifest panel, pre-checks table, model-load breakdown, TTFT/ITL vs λ
  with per-class SLO lines, failure rates, **per-class group-bys**
  (`GROUP BY scenario, session_idx`), **SLO-attainment table + λ\***, **supportable-users
  estimate** (editable `sessions_per_user_per_hour`, Little's-law concurrent sessions),
  **response-quality panel + capacity-vs-quality table** (§13.5/§15.1: per-config users
  at λ\* paired with quality scores and inter-config deltas; quality-flagged banner),
  hardware-headroom overlays, raw table; headless executor in `tools/reports/`.
- **Bootstrap**: `reports/STYLE.md` created with the first styling decisions; first
  curated report authored after E3 (§15.3).
- **DoD**: notebook executes headless against a synthetic fixture DB with a known λ\*,
  user count, and quality rows (asserted, including the capacity-vs-quality deltas),
  then against the real E1 DB.

### M10 — Cleaner (S–M) — ✅ done 2026-06-13 (read-only `identify()` policy over the §7.1 labels/§7.2 pattern with the skip rules — model-cache PVC §7.6, recent-N JFrog tags, active-job scratch, age threshold — plus approval-gated `prune()`; KubectlCleanerBackend headless, SLURM scratch via the MCP-driven `scratch_candidates()`, JFrog `jf` backend deferred; `reminder_due()` for Claude's periodic nudge — Claude never prunes itself; 5 tests; cluster-resource validation at E1/E5)

- **Deliverables**: `tools/cleaner.py` (§7.7) — identification stage (read-only report
  over the §7.1 labels: K8s objects, scratch dirs, JFrog tags) + pruning stage gated on
  explicit operator approval; age threshold configurable. Includes the §7.7 reminder
  mechanism: Claude surfaces a periodic run-the-cleaner reminder (scheduled wake-up /
  session reminder); Claude never executes the pruning itself.
- **DoD**: identification correctly lists deliberately-orphaned test resources on both
  platforms; pruning removes exactly the approved list; model-cache PVCs skipped.

### M11 — Quality eval runner (M) — ✅ laptop half done 2026-06-13 (QualityEvalRunner implements M7's seam; BuiltinEvalBackend in-process grader is the tested path against the mock's canned answers; LmEvalBackend wraps lm-eval-harness `local-chat-completions` for standard suites — **validated end-to-end on-cluster 2026-06-16 on BOTH SLURM + K8s** (real gsm8k Stage-A gate 0.56 + Stage-B 0.5436 against a deployed Apertus-8B, `quality_evals` rows persisted both platforms). **Gotcha:** the `lm-eval[api]` extra is REQUIRED (plain `lm-eval` ModuleNotFoundError's on `tenacity`); now pinned in `requirements.txt`. Stage-A floor gate + Stage-B suites×eval-concurrency rows persisted to `quality_evals`; wired into M7 `main.py`; GPQA gating + thinking-model parsing documented/deferred)

- **Deliverables**: `tools/benchmarker/quality_eval/` — lm-eval-harness wrapper
  targeting the deployed OpenAI-compatible endpoint(s); **Stage A** sanity gate
  (configurable suite/subset/floor, abort signalling through the M7 orchestrator,
  `skip_quality_gate` honoured) and **Stage B** comparison (full suites × configured
  eval-concurrency levels, natural decoding, suite-defined sampling); rows persisted to
  `quality_evals` (§14.9); eval datasets pre-staged by M8 alongside the workload
  datasets; no standing reference — deltas computed in-report across the experiment's
  deployment configs.
- **DoD**: against the mock server with deterministic canned answers, Stage-A pass/fail
  logic and Stage-B score collection are verified; the gate-abort path is exercised in
  M7's integration test; GPQA-Diamond gated-access setup documented (open decision 8);
  thinking-model answer parsing tracked in TODOs, not blocking.

## 5. Experiments track

| ID | Experiment | Needs | Definition of done |
|---|---|---|---|
| **E1** | **Walking skeleton**: Apertus-8B, 1× GH200 node (`clariden`), single-entry mix of `smoke-synthetic` (§9.2 smoke-run exemption — results are pipeline validation, never findings; **stock NGC vLLM image allowed**, operator decision 2026-06-12 — single-node E1 exercises no Slingshot/CXI path; §9.1 repo-built images mandatory from E2a onward), 3 λ levels | M0–M7 (manual drive acceptable), M9 for the notebook DoD; M8 to re-run automated | Full pipeline executes: pre-checks → engine → primer → §13.5 Stage-A gate → sweep (incl. `server_stats` + `hardware_stats` capture) → Stage-B eval → DB → notebook renders all panels. Teardown leaves zero orphans. **✅ done 2026-06-14** — assistant-driven via FirecREST MCP: 629 requests over λ∈{0.5,1,2}, 0 errors, persisted DB (not smoke) + report.ipynb/ttft/itl PNGs rendered against the real DB, engine torn down cleanly (zero orphans). Stock image → §8 pre-checks skipped (no MPI/NCCL dev to build nccl-tests; validated on the Alps image at E2); sweep never saturated so λ\*=2.0 is the swept ceiling, not a capacity limit; pipeline-validation only (never findings, §9.2). Bring-up fixes in commit 84e3c01. **Re-validated 2026-06-14 post-merge/refactor on the current code** (671 requests over λ∈{0.5,1,2}, 0 errors, persisted real DB + notebook, zero orphans) — exercised the moved `tools.common.results_db` end-to-end on-cluster; the re-run also surfaced + fixed a missing `skip_quality_gate`/`skip_quality_compare` in `e1-walking-skeleton.yaml`. |
| **E2a** | **Single-node characterisation** (`clariden`): populate `tools/system_prechecks_reference.yaml` 1-node rows from repeated E1-class deployments | M4, M5; piggybacks on E1 | 1-node TBD placeholders replaced with measured medians + tolerances; §8.4 gate enforceable at single-node scope. **✅ done 2026-06-15** — Apertus-8B 4×GH200 TP4; intra-node NVLink references populated (NCCL all_reduce 317.7 / all_gather 283.5 / alltoall 306.2 GB/s; NVSHMEM alltoall 12.6 µs; capstor storage floor). Drove the §8 **dedicated multi-rank-step refactor** + prebuilt-binary discovery + richer storage + model-load dissection; M4 cluster-validated here. |
| **E2b** | **Multi-node characterisation** (`clariden`): reference rows at E3's exact rank topology (inter-node collectives + NVSHMEM over Slingshot), gathered during E3 bring-up smokes, *before* graded E3 measurement | E2a; E3's multi-node deployment templates | Inter-node reference rows populated; E3's foundation gate enforceable on the cross-node fabric it actually depends on. **✅ done 2026-06-15** — Apertus-70B TP4×PP2 (8×GH200, 2 nodes); Slingshot references populated (NCCL all_reduce 131.1 / all_gather 86.5 / alltoall 38.3 / PP-link sendrecv 23.5 GB/s; NVSHMEM alltoall 47.9 µs). Surfaced the 70B **storage-bound load** (~0.185 GB/s) + the three multi-node engine fixes (Ray, `disable_custom_all_reduce`, `enforce_eager`) → motivated the M5 Ray-in-image rebuild. |
| **E2c** | **`breithorn` characterisation**: reference rows on the K8s GH200 nodes | M6 K8s templates (landed via worktree-k8s), M7 K8s path | `breithorn` rows populated; prerequisite for E5. |
| **E3a** | **Real-scenario capacity precursor** (`clariden`): Apertus-70B single-node TP4 (dense; **no Ray/PP** → sidesteps the E2b multi-node faults), operator mix 0.8 `chat-short-turns` (wildchat) + 0.2 `agentic-coding` (longbench, ~25k-char turn-1), full 256K context, λ swept to saturation, per-class SLOs. Exercises the **real-scenario capacity pipeline** (real-text datasets + λ\* / SLO-attainment / supportable-users) end-to-end before the Kimi-scale MoE run. Config `examples/benchmark-configs/apertus70b-mixed-single-node-clariden.yaml` **settled** (agentic ≈7000 tok ≈25k-char code; `max_num_batched_tokens` 16384 ≥ agentic cap; §8 gate `on_fail: abort`) — ready to run pending the weights stage + a go/no-go. | E1, E2a, M8, M9; `nvidia-gh200-vllm-0.22.1` image (built); real-text datasets (verified on-cluster 2026-06-15) | Report shows λ\*, per-class SLO attainment, and the supportable-users estimate for the Apertus mix; first **real capacity findings**; capacity pipeline proven for E3. |
| **E3** | **The capacity run (primary goal)**: Kimi-K2.6 on `clariden`, `scenario_mix` 0.8 `agentic-coding` (longbench, `sequential` sessions) + 0.2 `chat-short-turns` (wildchat), `slos` declared, λ sweep. **Prerequisites**: (a) verify Kimi-K2.6 architecture support in the pinned vLLM — a forced version bump triggers §16.2 flag-compat work + image rebuild; (b) MoE bring-up ladder: Apertus-70B PP=2 smoke (dense, cross-node PP) **— done at E2b**, then a small open MoE at EP>1 (exercises expert all-to-all + the §8 NVSHMEM plane) before Kimi-scale (≈1 TB weights → ≥3–4 GH200 nodes, TP4 × PP≥3, §6.1); (c) **Ray-in-image** — resolved in-repo (`nvidia-gh200-vllm-0.23.0-net.v1`), **needs build + JFrog push** + the 0.23.0 §16.2 flag-compat check; (d) E3a (Apertus-70B single-node) validates the real-scenario capacity pipeline first | E1, E2a/E2b, E3a, M8, M9; Kimi image (M5) | Report shows λ\*, per-class SLO attainment, supportable-users estimate; results pass adversarial review (phase 16). |
| **E4** | **Feature-effect sweeps** on the E3 workload: `enable_prefix_caching` on/off, `kv_offloading_size`, `kv_cache_dtype`, `session_affinity` vs `random` (multi-instance); speculative decoding on Apertus-70B chat/long-context (§9.2 pairing, §17.2 placements; `spec_accept_rate` captured via the M2 scraper) | E3 baseline | Marginal effect of each feature on λ\*/users quantified per §16.1, **with the §13.5 capacity-vs-quality pairing for quality-impacting knobs** (quantization / KV dtype) — the flagship *"N× more users at −M pts"* report; findings recorded in §17. |
| **E5** | **Platform comparison**: E3 workload, SLURM (`clariden`) vs K8s (`breithorn`), same GH200 hardware, engine, config; the SLURM Benchmarker drives the K8s engine (M6/M7 K8s engine path) | E3; E2c | Per-platform overlay report isolating the platform contribution (§16.1); "is K8s slower than it could be?" answered with telemetry-backed evidence. |

## 6. Dependency sketch

```
M0 ─┬─ M1 (dataset gen) ───┐
    ├─ M2 (load gen) ──────┤
    ├─ M3 (DB + sampler) ──┼─ M7 (orchestrator) ─→ E1✅ ─→ E2a✅ ─→ E2b✅ ─→ E3a ─→ E3 ─→ E4
    └─ M6 (planner) ───────┘                                          (Apertus) (Kimi)  └─→ E5 ←─ E2c
M5 (images) ── M4 (prechecks)✅ ────┘
M8 (coordinator: after M3+M6+M7) ──→ automated re-run; required for E3a/E3
M9 (reports: fixture-testable after M3; required for E1 DoD)
M10 (cleaner: any time after M0; required before E3 leaves debris at scale)
M11 (quality eval: after M0, mock-testable with M2's server; Stage-A gate in E1, capacity-vs-quality required for E4)
```

Parallelizable from day one: M5 (images) alongside M0–M3; M9 notebook against fixture
DBs alongside M2/M3.

## 7. Testing and verification strategy

- **Unit (laptop, every commit)**: config validation matrix; seeding/regeneration
  byte-identity; distribution statistics under fixed seeds; arrival-process properties
  (session-start semantics); error-taxonomy classification; DB schema conformance +
  write-contention test.
- **Mock integration (laptop)**: load generator vs `mock_openai_server` with known
  delays, mock metrics endpoint, and injected faults — latency math, `server_stats`
  scraping, phases, drain, session sequencing, routing.
- **Cluster smoke (per milestone DoD)**: smallest-footprint single-node checks
  (`sbatch --test-only`, dry-run applies, one-node pre-check, sampler on GH200).
- **End-to-end**: E1 is the integration test; M8's kill/resume and forced-failure
  teardown drills are the resilience tests.
- **Report correctness**: fixture DB with analytically known λ\* and user count.

## 8. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Multi-node MoE bring-up (Kimi-K2.6: ≈1 TB weights, ≥3–4 GH200 nodes, TP4 × PP≥3 + EP) is the heaviest unknown on the critical path; pinned vLLM may not support the architecture | E3 slips or forces a version bump (§16.2 flag-compat + rebuild) | E3 prerequisite ladder: verify model support first; Apertus-70B PP=2 smoke (cross-node PP), then small open MoE at EP>1 (expert all-to-all + NVSHMEM plane) before Kimi-scale; §6.1 TP≤4 rule enforced by M0 validation |
| Per-run DB exceeds FirecREST direct-transfer limits (hundreds of MB of `requests` + 1 Hz `hardware_stats`) | Results stranded on cluster | M8 uses the FirecREST staged-transfer path + compression; round-trip drill with a >100 MB fixture is in M8's DoD |
| HF dataset downloads from compute nodes (egress/proxy) | M1 fails on cluster | M8 pre-stages datasets to capstor before submission; §11.1 abort semantics already specced — no silent fallback |
| DCGM unavailable in engine containers | §13.3 telemetry gaps | Sampler is stdlib-only with `nvidia-smi dmon` fallback; verified during E1, not E3 |
| vLLM log/metrics format drift breaks model-load parsing and `server_stats` scraping | Silent `NULL`s | Pin backend version per experiment (§9.1); parser tests against captured log/metrics fixtures per pinned version |
| FirecREST constraints (5 MB direct upload, API rate limits) | Coordinator instability | Code reaches cluster via benchmarker image / git clone, never file upload; downloads via staged transfer (above) |
| Client-side load-gen saturation at high λ | Measured latencies become client artefacts | Event-loop lag guard (M2 DoD); shard load gen across processes if the guard trips |
| Laptop sleep / network loss during multi-hour runs | Orphaned runs, lost monitoring | M8 resumable state file + reattach drill in its DoD; Cleaner (M10) as backstop |
| K8s ingress TLS/cert issuance delays (`examples/k8s-deployment` note) | E5 setup friction | Readiness wait covers cert issuance; insecure-transport escape hatch only for in-cluster smoke, never for measured runs |

## 9. Open decisions (operator input needed)

1. **Global configuration location** — **resolved 2026-06-12**: `tools/common/global.yaml`
   (SPECIFICATIONS.md §3.3), implemented in M0. JFrog base remains `TBD` inside it
   (decision 2).
2. **JFrog publish path** — **resolved 2026-06-12**:
   `https://jfrog.svc.cscs.ch/artifactory/ml/inference` (in `global.yaml`). Operator
   action open: fix the local `jf` server config (doubled `/artifactory` in the stored
   Artifactory URL); a §4 pre-flight row now guards this.
3. **Code-to-cluster delivery / Benchmarker runtime** — **resolved 2026-06-13 (operator
   decision)**: deps from a **staged uv venv on capstor** (`{scratch_base}/benchmarker-venv`,
   built once by `benchmarker.sbatch`: `uv venv` + `uv pip install -r
   tools/benchmarker/requirements.txt`); the **live `tools/` code is mounted from capstor**
   (`{run_dir_remote}/tools`, staged by the Coordinator, added to PYTHONPATH) so code iterates
   without rebuilds — same pattern the engine job already uses for the precheck/sampler
   scripts. Needs compute-node egress for the one-time `uv pip install` (or a seeded wheel
   cache); exact versions pinned after E1's first build (TODOs).
4. **Centralized results DB shape** — **resolved 2026-06-13 (M8)**: `experiments/results.db`,
   per-run DBs merged on download via `tools/coordinator/merge.py` (`ATTACH` + delete-then-
   insert, idempotent by `run_id`); per-run files remain the §14.8 provenance artifacts.
5. **Programmatic FirecREST client for the Coordinator** — **resolved 2026-06-13 (M8,
   operator decision)**: **MCP-mediated, assistant-driven in-session** — there is no
   autonomous SLURM/FirecREST client; the assistant performs submit/monitor/staged-download/
   teardown via the FirecREST MCP tools using the `tools/coordinator/` helpers. Consequence
   (accepted): orchestrating a SLURM experiment goes through Claude. K8s runs headless via
   `kubectl`. (`pyfirecrest` 3.8.0 stays installed for pre-flight/ad-hoc use.)
6. **`sessions_per_user_per_hour` defaults per class** (TODOs) — needed before E3's
   report narrative is meaningful; literature/telemetry research task.
7. **Small open MoE for the E3 bring-up ladder** — pick a model the pinned backend
   supports (candidate class: Mixtral-/Qwen-MoE-sized) purely as an EP smoke vehicle.
8. **GPQA-Diamond gated access** — **documented 2026-06-13 (M11)**: setup (HF license
   click-through + `HF_TOKEN` on the Benchmarker, reusing the jobs' `HUGGING_FACE_HUB_TOKEN`)
   is in `lm_eval_backend.py`; the default Stage-A gate is ungated GSM8K, and `compare.suites`
   can drop GPQA for an ungated hard suite. Live token/access still to be confirmed at E1
   before Stage-B GPQA defaults run.
9. **K8s multi-rank §8 pre-check launch (multi-node + Ray)** — **resolved 2026-06-15 (operator
   decision: option B)**: K8s has no `srun`/PMIx, and Ray is control-plane only (vLLM's
   collectives still run over NCCL/Slingshot), so the multi-rank pre-check is launched **the way
   the engine is launched on K8s — via Ray**: a **Ray-placed `torch.distributed` collective
   probe** (one task per GPU; all_reduce / all_gather / alltoall busbw on vLLM's own torch+NCCL
   stack), run on the engine's Ray cluster before `vllm serve`, gated (§8.4), then serve.
   Single-node K8s is the 1-pod case of the same probe. **Rejected**: the Kubeflow MPI Operator
   (`MPIJob`) — it would keep one nccl-tests methodology across platforms but adds a cluster
   dependency and a *second* distributed runtime (MPIJob + Ray → two gang-scheduled bring-ups
   per run). Trade-off accepted: a second probe methodology (torch.distributed vs SLURM's
   nccl-tests) — busbw stays comparable, documented for the §16.1 / E5 overlay. **Interim**
   (until the probe lands): single-node K8s runs the existing in-pod single-process nccl-tests
   (`NCCL_TESTS_MPI=0`, `-g <gpus>`). Implementation is E5 / multi-node-K8s scope (TODOs).

## 10. Review log

**Phase 6 adversarial review — completed 2026-06-12** (spec-conformance subagent +
architectural review). 5 HIGH / 9 MEDIUM / 8 LOW findings; all incorporated above.
Operator decisions taken during the review:

- **D1**: `agentic-coding` registry flipped to `session.mode: sequential` with a
  provisional tool-execution `think_time_ms` — open_loop made `session_e2e_ms` an
  arrival-schedule artifact, corrupting the §13.4 SLO and the §15.1 users estimate.
- **D2**: E1 runs the new registered `smoke-synthetic` scenario (pipeline validation
  only; §9.2 exemption) instead of an illegal source-kind override on
  `chat-short-turns`.
- **D3**: λ counts **session starts** in all modes (§12.3 *What λ counts*), aligning
  arrival semantics with `scenario_mix` weights and the users math.

Structural outcomes: new M7 (Benchmarker orchestrator — previously unowned); Coordinator
and later milestones renumbered (M8–M10); `server_stats` scraper assigned to M2;
hardware sampler redesigned as stdlib-only and placed on the engine nodes (M3/M6/M7);
E2 split into E2a/E2b/E2c so the foundation gate is enforceable at E3's actual
multi-node topology; FirecREST staged-transfer + Coordinator resumability added to M8.

**Quality-evaluation extension — 2026-06-12** (follow-up to the comparative review
against SemiAnalysis InferenceMAX/InferenceX). Operator
decisions: **no standing quality anchor** — deltas are experiment-internal across the
experiment's deployment configs; **Stage-A sanity gate default-on** in every experiment,
skippable (`skip_quality_gate`); **Stage-B comparison** pairs capacity gains and quality
deltas in the same report (the "N× users at −M pts" claim); `ignore_eos` parameterized
as `output_length_mode: forced | natural` (§11.6). Structural outcomes: new **M11
Quality eval runner**; M0/M2/M7/M9 amended; E1 exercises the gate; E4's DoD now requires
the capacity-vs-quality pairing; spec gained §13.5 + §14.9 (`quality_evals` table).

**§8 pre-check refactor + E2 characterisation — 2026-06-15.** E2a/E2b ran on `clariden`,
populating the 1-node + 2-node `tools/system_prechecks_reference.yaml` foundation references
and cluster-validating M4. The §8 pre-check was restructured (plan-mode design + review) from
the welded `run_system_prechecks && exec <engine>` into a **dedicated one-rank-per-GPU
`srun --mpi=pmix` step** that gates a separate one-task-per-node engine step (SPEC §8.2
rewritten) — this is what gives multi-node NCCL its real per-GPU topology and enables multi-PE
NVSHMEM; storage grew to three mount-aware metrics and `model_load_weights_gib` was added
(§10.2). E2b's 70B load proved **storage-bound** (~0.185 GB/s) and surfaced three multi-node
engine fixes (Ray, `disable_custom_all_reduce`, `enforce_eager`), motivating the
`nvidia-gh200-vllm-0.23.0-net.v1` **Ray-in-image** rebuild (M5). `worktree-images` (AMD MI300A
image + 0.23.0/Ray) and `worktree-k8s` (K8s engine/ingress path for E5) merged to `main`.
**New experiment E3a** inserted: an Apertus-70B single-node real-scenario capacity run as the
lower-risk precursor that proves the capacity pipeline before the Kimi-K2.6 flagship (E3).

## 11. Process gates

Per `CLAUDE.md` *How we work together*: each component milestone closes with phase 8
(adversarial review of implementation); E1/E3/E5 each get phase 10 (pre-execution
go/no-go), phases 12–16 (monitoring through adversarial results review), and E3
additionally phases 17–18 (conclusions + report). Skips require explicit operator
approval.
