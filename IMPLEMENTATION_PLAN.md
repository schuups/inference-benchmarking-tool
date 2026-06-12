# Implementation Plan

Build order for the tool specified in `SPECIFICATIONS.md`. The spec says *what*; this
document says *in what order, with what intermediate proof*. It is a living document:
milestones get checked off, re-scoped, or re-ordered as implementation reveals new facts
— any structural change to a milestone goes through the working-agreement process
(`CLAUDE.md` *How we work together*).

**Process status**: phase 5 (*Planning and design*) and phase 6 (*Adversarial review of
the plan*) are complete; review findings are incorporated below (see *Review log*). Next:
phase 7 (*Implementation*), starting at M0.

---

## 1. Objective and success criterion

The driving deliverable is the framework's primary v1 question (README, §12.4, §14.1):

> **How many users can a model instance support under a mixed workload of 80%
> agentic-assisted coding + 20% user chats, at declared per-class SLOs — and how do
> backend features move that number?**

The plan is complete when experiment **E3** (below) produces a report containing the
SLO-attained rate λ\*, per-class SLO attainment, and the supportable-users estimate for
the 80/20 mix on `clariden` — and **E5** answers whether the same workload runs slower
on Kubernetes (`breithorn`) than on SLURM.

All spec pillars are in scope — system pre-checks (§7), image building (§8.1),
model-load decomposition (§9.2), platform comparison (§15.1), curated reports (§14.3).
None are descoped.

## 2. Current state

| Asset | Status |
|---|---|
| `SPECIFICATIONS.md`, scenario registry (`tools/scenarios/*.yaml` × 4, incl. `smoke-synthetic`) | Done |
| `tools/pre-flight-checks.py` (§3) | Done |
| `tools/system_prechecks_reference.yaml` (§7.3) | Skeleton — all `expected: TBD` (populated by E2) |
| `examples/nccl-tests/` (fingerprint, build, collectives, NVSHMEM scripts) | Done — to be adapted into M4 |
| `examples/slurm-deployment/` (Apertus-8B vLLM sbatch + EDF) | Done — seed for M6 templates |
| `examples/k8s-deployment/` (deployment/service/ingress/PVC) | Done — seed for M6 templates |
| `examples/docker-images-build/` (Dockerfile) | Partial — seed for M5 |
| `tools/common/` (global.yaml §2.3, benchmark-YAML schema + CLI, run-ID §6.2) + `examples/benchmark-configs/mixed-80-20.yaml` + `tools/tests/` (20 tests) | Done (M0, 2026-06-12) |
| `tools/benchmarker/dataset_gen/` (registry loader, seeded sampling, all four §10.5 sources, manifest emitter, offline CLI) — validated against real LongBench / WildChat / gsm8k downloads | Done (M1, 2026-06-12) |
| Planner, Coordinator, Cleaner, Reports generator, Benchmarker (orchestrator, load gen, prechecks runner, hw sampler, results DB) | **Not implemented** |

## 3. Build strategy

1. **Local-first**: the dataset generator, load generator, results DB, and report
   notebook are pure Python — built and tested on the laptop against a mock
   OpenAI-compatible server before any cluster time is spent.
2. **Walking skeleton before scale**: the first end-to-end run (E1) uses the
   `smoke-synthetic` scenario (registered for pipeline validation, exempt from the §8.2
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
  §15.2, `dataset_config` with `scenario_mix` §10.4, `slos` §12.4, `rate_levels`,
  `arrival_process` §11.3, `routing_strategy` §11.4, timeouts, `output_length_mode`
  §10.6, the `quality_eval` block §12.5, and the §7 pre-check surface: `skip_system_prechecks`, `system_prechecks_on_warn` / `_on_fail`,
  `system_prechecks_timeout_s`, `collective_tests_version`,
  `collective_tests_cache_dir`, `shmem_required`); `tools/common/runid.py` (§6.2); a
  committed canonical example `examples/benchmark-configs/mixed-80-20.yaml`.
- **Validation rules enforced**: weights sum to 1.0; registered scenario names; SLO
  metric/percentile enums; `tensor_parallel_size ≤ gpus_per_node` (§5.1). (Modality
  rejection happens at registry-load time in M1, per §10.5 — not here.)
- **DoD**: pytest green; `python -m tools.common.config <yaml>` accepts the canonical
  example and rejects each violation class with a one-line error.

### M1 — Dataset generator (L) — ✅ done 2026-06-12 (all four sources; traces support gsm8k, further trace datasets + HF revision pinning tracked in TODOs)

- **Deliverables**: `tools/benchmarker/dataset_gen/` — registry loader (rejects
  non-`[text]` modalities at load time, §10.5), mix planner (per-class sub-pools,
  `num_prompts` split ∝ `weight × E[turns_per_session]`), sources in order
  **synthetic → longbench → wildchat → reasoning_trace_replay** (§10.5), session builder
  (`append_delta`, turns, think-time, `followup_input_length` §10.3), per-class seeding (§10.8), global unique
  headers (§10.6), `thinking: true` widening, tokenizer handling (§10.6), source-failure
  abort (§10.1), manifest emitter (§13.7: `mix` / `classes[]` / `run_assumptions`,
  including the derived per-class expected request rates per §11.3 *What λ counts*).
- **Artifact format** (decision embedded here): JSONL prompt pool, one record per turn —
  `{scenario, session_idx, turn_idx, prompt_text, max_tokens}` — plus `manifest.json`.
- **DoD**: byte-identical regeneration with same config (the §10.8 contract, as a test);
  empirical per-class length/turn distributions match declared params within tolerance;
  manifest schema-validates; runs on laptop with no cluster access (synthetic + cached
  HF datasets).

### M2 — Load generator (L)

- **Deliverables**: `tools/benchmarker/load_gen/` — asyncio streaming client
  (forced/natural `output_length_mode` per §10.6, sampled `max_tokens`); arrival
  processes `poisson` + `burst_mmpp`
  scheduling **session starts** (λ semantics per §11.3 *What λ counts*); session modes
  `open_loop` / `sequential` with `think_time_ms` (§10.7); class inheritance per
  session; routing `random` / `session_affinity` (§11.4); warmup / measurement / drain
  phases with sweep-step session accounting (§11.2: session-population warmup, no new
  sessions after measurement end, incomplete-session truncation per §12.2); per-request
  metrics + §12.1 error taxonomy; **per-instance
  `server_stats` scraper** (§13.4: `requests_running`, `requests_waiting`,
  `gpu_cache_pct`, `spec_accept_rate` from the backend's metrics endpoint); readiness
  wait + per-instance model-load parsing (vLLM logs/API first, §9.2/§11.1); inductor
  primer (§9.3) including the primer-missed-target warning.
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

### M3 — Results DB and hardware sampler (M)

- **Deliverables**: `tools/benchmarker/db.py` — the six §13 tables, column-for-column,
  with a concurrency design (WAL mode + single-writer queue, or per-producer DB files
  merged at finalisation — decided at implementation, asserted by a contention test);
  smoke-test-mode suppression hook (§7.2: results not persisted on pre-check cache
  miss). `tools/benchmarker/hw_sampler.py` — **single-file, stdlib-only** script
  (shells out to DCGM / `nvidia-smi dmon` / `rocm-smi` + `/proc`) so it runs inside
  *any engine container without extra dependencies*; 1 Hz per §12.3; writes NDJSON to
  the run's scratch directory; `NULL` for unexposed signals.
- **Placement (per review H3)**: the sampler runs **on the inference-server nodes**,
  backgrounded inside the engine container session before `exec <engine>` (wired by the
  M6 templates); its NDJSON output is ingested into `hardware_stats` by the Benchmarker
  orchestrator (M7) at finalisation.
- **DoD**: schema asserted against §13 in a test; concurrent-writer test passes; sampler
  runs dependency-free in a bare container image locally (all-`NULL` GPU rows on
  laptop); verified on a GH200 node during E1.

### M4 — System performance pre-checks runner (M)

- **Deliverables**: `tools/benchmarker/prechecks/run_system_prechecks` — adapts
  `examples/nccl-tests/` (stack fingerprint, cached build, rank-0 toolchain install,
  collectives, NVSHMEM) into the in-container gate of §7.2; adds the storage
  sequential-read check; parses results into `system_prechecks` rows (§13.6); grades
  against `tools/system_prechecks_reference.yaml` (§7.3–7.4) with warn/fail signalling
  to the orchestrator (M7) and smoke-test mode on cache miss.
- **DoD**: executes inside the vLLM EDF container session on one `clariden` node,
  concatenated `run_system_prechecks && exec <engine>`; rows persisted; with TBD
  references everything logs informational (gate unenforceable until E2).

### M5 — Images and registry workflow (M, parallel track)

- **Deliverables**: `tools/images/vllm/` (Dockerfile, `patches/`, build-args metadata
  per §8.1); SLURM-based build workflow (from `examples/docker-images-build/`,
  TODOs *Support building Docker images via SLURM jobs*); JFrog push with canonical
  tags + provenance record; a lightweight `tools/images/benchmarker/` image (Python +
  tokenizers/datasets/aiohttp/lm-eval-harness — the latter for M11's quality stages)
  for the Benchmarker phases. The hardware sampler needs
  **no** image support — it is stdlib-only by design (M3) and runs in the engine image
  as-is.
- **DoD**: vLLM image builds reproducibly via SLURM job and pushes to JFrog; EDF
  references it by canonical tag; benchmarker image runs M1 dataset generation on the
  cluster. Networking-library correctness (NCCL ↔ Slingshot/libfabric) is proven by M4
  passing inside this image — the two milestones gate each other.

### M6 — Planner (M)

- **Deliverables**: `tools/planner/` + templates `tools/templates/vllm.edf.j2`,
  `tools/templates/benchmarker.sbatch.j2`, **`tools/templates/benchmarker.pod.yaml.j2`**
  (the §4 Benchmarker-as-pod path for K8s, needed by E5), `tools/templates/k8s/*.yaml.j2`
  (engine side, seeded from `examples/k8s-deployment/`); renders the full §13.8
  experiment directory from one benchmark YAML; CLI + Claude-driven paths (§4).
- **DoD**: golden-file render tests; rendered sbatch passes `sbatch --test-only` on
  `clariden`; rendered manifests pass `kubectl apply --dry-run=server` on `breithorn`;
  asserted in tests: time-limit alignment (§5.1), the §7.2
  `run_system_prechecks && exec <engine>` concatenation, and the M3 sampler
  backgrounding in the engine container command.

### M7 — Benchmarker orchestrator (M–L) *(added by review H1)*

- **Deliverables**: `tools/benchmarker/main.py` — the cluster-side driver that owns the
  §1 phase sequencing: run dataset generation → **submit the inference deployment(s)
  from within the Benchmarker allocation** (sbatch on SLURM, kubectl on K8s) only after
  the dataset is ready → wait for readiness + primer (§11.1) → **Stage-A quality gate**
  (§12.5, via M11) → run load generation → **Stage-B quality comparison** (§12.5, via
  M11) → finalise the per-run DB (ingest sampler NDJSON into `hardware_stats`, §13.5;
  stage outputs per §5.1). Owns warn/fail propagation from pre-checks to the Coordinator (§7.4 pause +
  `system_prechecks_on_warn` default for non-interactive runs) and smoke-test-mode
  propagation (suppress persistence pipeline-wide, warn at launch and termination,
  §7.2).
- **DoD**: phase ordering asserted by an integration test (engine job not submitted
  until the dataset pool exists); forced pre-check warn pauses and resolves per config;
  smoke-test mode produces no persisted results and two unmissable warnings.

### M8 — Coordinator (L)

- **Deliverables**: `tools/coordinator/` — pre-staging of HF datasets to capstor before
  submission (review LOW); submit (FirecREST for SLURM, kubectl for K8s); monitor loop
  with log streaming and the §7.4 **operator abort/proceed interaction** on pre-check
  warns; **resumable state** (state file per run; reattach to a running experiment after
  laptop sleep / network loss); per-run DB download via the **FirecREST staged-transfer
  path with compression** (direct transfer is capped ~5 MB; per-run DBs will reach
  hundreds of MB — review H5); merge into the centralized results DB; teardown on
  success **and** failure (§6.3–6.6).
- **DoD**: drives E1 end-to-end unattended; a deliberately killed Coordinator resumes
  from its state file and completes monitoring + collection; a deliberately killed run
  still tears down all labelled resources; PVC retention honored (§6.6); a >100 MB
  fixture DB round-trips intact through the staged-transfer download.

### M9 — Reports generator (M)

- **Deliverables**: `experiments/template_report.ipynb` with every §14.1 panel —
  scenario/mix manifest panel, pre-checks table, model-load breakdown, TTFT/ITL vs λ
  with per-class SLO lines, failure rates, **per-class group-bys**
  (`GROUP BY scenario, session_idx`), **SLO-attainment table + λ\***, **supportable-users
  estimate** (editable `sessions_per_user_per_hour`, Little's-law concurrent sessions),
  **response-quality panel + capacity-vs-quality table** (§12.5/§14.1: per-config users
  at λ\* paired with quality scores and inter-config deltas; quality-flagged banner),
  hardware-headroom overlays, raw table; headless executor in `tools/reports/`.
- **Bootstrap**: `reports/STYLE.md` created with the first styling decisions; first
  curated report authored after E3 (§14.3).
- **DoD**: notebook executes headless against a synthetic fixture DB with a known λ\*,
  user count, and quality rows (asserted, including the capacity-vs-quality deltas),
  then against the real E1 DB.

### M10 — Cleaner (S–M)

- **Deliverables**: `tools/cleaner.py` (§6.7) — identification stage (read-only report
  over the §6.1 labels: K8s objects, scratch dirs, JFrog tags) + pruning stage gated on
  explicit operator approval; age threshold configurable. Includes the §6.7 reminder
  mechanism: Claude surfaces a periodic run-the-cleaner reminder (scheduled wake-up /
  session reminder); Claude never executes the pruning itself.
- **DoD**: identification correctly lists deliberately-orphaned test resources on both
  platforms; pruning removes exactly the approved list; model-cache PVCs skipped.

### M11 — Quality eval runner (M)

- **Deliverables**: `tools/benchmarker/quality_eval/` — lm-eval-harness wrapper
  targeting the deployed OpenAI-compatible endpoint(s); **Stage A** sanity gate
  (configurable suite/subset/floor, abort signalling through the M7 orchestrator,
  `skip_quality_gate` honoured) and **Stage B** comparison (full suites × configured
  eval-concurrency levels, natural decoding, suite-defined sampling); rows persisted to
  `quality_evals` (§13.9); eval datasets pre-staged by M8 alongside the workload
  datasets; no standing reference — deltas computed in-report across the experiment's
  deployment configs.
- **DoD**: against the mock server with deterministic canned answers, Stage-A pass/fail
  logic and Stage-B score collection are verified; the gate-abort path is exercised in
  M7's integration test; GPQA-Diamond gated-access setup documented (open decision 8);
  thinking-model answer parsing tracked in TODOs, not blocking.

## 5. Experiments track

| ID | Experiment | Needs | Definition of done |
|---|---|---|---|
| **E1** | **Walking skeleton**: Apertus-8B, 1× GH200 node (`clariden`), single-entry mix of `smoke-synthetic` (§8.2 smoke-run exemption — results are pipeline validation, never findings), 3 λ levels | M0–M7 (manual drive acceptable), M9 for the notebook DoD; M8 to re-run automated | Full pipeline executes: pre-checks → engine → primer → §12.5 Stage-A gate → sweep (incl. `server_stats` + `hardware_stats` capture) → Stage-B eval → DB → notebook renders all panels. Teardown leaves zero orphans. |
| **E2a** | **Single-node characterisation** (`clariden`): populate `tools/system_prechecks_reference.yaml` 1-node rows from repeated E1-class deployments | M4, M5; piggybacks on E1 | 1-node TBD placeholders replaced with measured medians + tolerances; §7.4 gate enforceable at single-node scope. |
| **E2b** | **Multi-node characterisation** (`clariden`): reference rows at E3's exact rank topology (inter-node collectives + NVSHMEM over Slingshot), gathered during E3 bring-up smokes, *before* graded E3 measurement | E2a; E3's multi-node deployment templates | Inter-node reference rows populated; E3's foundation gate enforceable on the cross-node fabric it actually depends on. |
| **E2c** | **`breithorn` characterisation**: reference rows on the K8s GH200 nodes | M6 K8s templates, M7 K8s path | `breithorn` rows populated; prerequisite for E5. |
| **E3** | **The capacity run (primary goal)**: Kimi-K2.6 on `clariden`, `scenario_mix` 0.8 `agentic-coding` (longbench, `sequential` sessions) + 0.2 `chat-short-turns` (wildchat), `slos` declared, λ sweep. **Prerequisites**: (a) verify Kimi-K2.6 architecture support in the pinned vllm-cxi — a forced version bump triggers §15.2 flag-compat work + image rebuild; (b) MoE bring-up ladder: Apertus-70B PP=2 smoke (dense, cross-node PP), then a small open MoE at EP>1 (exercises expert all-to-all + the §7 NVSHMEM plane) before Kimi-scale (≈1 TB weights → ≥3–4 GH200 nodes, TP4 × PP≥3, §5.1) | E1, E2a/E2b, M8, M9; Kimi image (M5) | Report shows λ\*, per-class SLO attainment, supportable-users estimate; results pass adversarial review (phase 16). |
| **E4** | **Feature-effect sweeps** on the E3 workload: `enable_prefix_caching` on/off, `kv_offloading_size`, `kv_cache_dtype`, `session_affinity` vs `random` (multi-instance); speculative decoding on Apertus-70B chat/long-context (§8.2 pairing, §16.2 placements; `spec_accept_rate` captured via the M2 scraper) | E3 baseline | Marginal effect of each feature on λ\*/users quantified per §15.1, **with the §12.5 capacity-vs-quality pairing for quality-impacting knobs** (quantization / KV dtype) — the flagship *"N× more users at −M pts"* report; findings recorded in §16. |
| **E5** | **Platform comparison**: E3 workload, SLURM (`clariden`) vs K8s (`breithorn`), same GH200 hardware, engine, config; Benchmarker-as-pod (M6/M7 K8s path) | E3; E2c | Per-platform overlay report isolating the platform contribution (§15.1); "is K8s slower than it could be?" answered with telemetry-backed evidence. |

## 6. Dependency sketch

```
M0 ─┬─ M1 (dataset gen) ───┐
    ├─ M2 (load gen) ──────┤
    ├─ M3 (DB + sampler) ──┼─ M7 (orchestrator) ─→ E1 ─→ E2a ─→ E2b ─→ E3 ─→ E4
    └─ M6 (planner) ───────┘        ↑                ↑               ↑    └─→ E5 ←─ E2c
M5 (images) ── M4 (prechecks) ──────┘                │               │
M8 (coordinator: after M3+M6+M7) ────────────────────┘ (automated re-run; required for E3)
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
| Multi-node MoE bring-up (Kimi-K2.6: ≈1 TB weights, ≥3–4 GH200 nodes, TP4 × PP≥3 + EP) is the heaviest unknown on the critical path; pinned vllm-cxi may not support the architecture | E3 slips or forces a version bump (§15.2 flag-compat + rebuild) | E3 prerequisite ladder: verify model support first; Apertus-70B PP=2 smoke (cross-node PP), then small open MoE at EP>1 (expert all-to-all + NVSHMEM plane) before Kimi-scale; §5.1 TP≤4 rule enforced by M0 validation |
| Per-run DB exceeds FirecREST direct-transfer limits (hundreds of MB of `requests` + 1 Hz `hardware_stats`) | Results stranded on cluster | M8 uses the FirecREST staged-transfer path + compression; round-trip drill with a >100 MB fixture is in M8's DoD |
| HF dataset downloads from compute nodes (egress/proxy) | M1 fails on cluster | M8 pre-stages datasets to capstor before submission; §10.1 abort semantics already specced — no silent fallback |
| DCGM unavailable in engine containers | §12.3 telemetry gaps | Sampler is stdlib-only with `nvidia-smi dmon` fallback; verified during E1, not E3 |
| vLLM log/metrics format drift breaks model-load parsing and `server_stats` scraping | Silent `NULL`s | Pin backend version per experiment (§8.1); parser tests against captured log/metrics fixtures per pinned version |
| FirecREST constraints (5 MB direct upload, API rate limits) | Coordinator instability | Code reaches cluster via benchmarker image / git clone, never file upload; downloads via staged transfer (above) |
| Client-side load-gen saturation at high λ | Measured latencies become client artefacts | Event-loop lag guard (M2 DoD); shard load gen across processes if the guard trips |
| Laptop sleep / network loss during multi-hour runs | Orphaned runs, lost monitoring | M8 resumable state file + reattach drill in its DoD; Cleaner (M10) as backstop |
| K8s ingress TLS/cert issuance delays (`examples/k8s-deployment` note) | E5 setup friction | Readiness wait covers cert issuance; insecure-transport escape hatch only for in-cluster smoke, never for measured runs |

## 9. Open decisions (operator input needed)

1. **Global configuration location** — **resolved 2026-06-12**: `tools/common/global.yaml`
   (SPECIFICATIONS.md §2.3), implemented in M0. JFrog base remains `TBD` inside it
   (decision 2).
2. **JFrog publish path** (TODOs) — blocks the M5 push step only.
3. **Code-to-cluster delivery** — proposal: baked into the benchmarker image (M5), git
   clone as fallback.
4. **Centralized results DB shape** — proposal: `experiments/results.db`, per-run DBs
   merged on download (`run_id` keys make this idempotent); per-run files remain the
   §13.8 provenance artifacts.
5. **Programmatic FirecREST client for the Coordinator CLI path** — MCP serves the
   Claude-driven path; the plain-CLI path likely needs `pyfirecrest`. Decide at M8.
6. **`sessions_per_user_per_hour` defaults per class** (TODOs) — needed before E3's
   report narrative is meaningful; literature/telemetry research task.
7. **Small open MoE for the E3 bring-up ladder** — pick a model the pinned backend
   supports (candidate class: Mixtral-/Qwen-MoE-sized) purely as an EP smoke vehicle.
8. **GPQA-Diamond gated access** — the dataset is HF-gated (license click-through +
   auth token on the Benchmarker). Needed before Stage-B defaults run; fallback: swap
   in an ungated hard suite.

## 10. Review log

**Phase 6 adversarial review — completed 2026-06-12** (spec-conformance subagent +
architectural review). 5 HIGH / 9 MEDIUM / 8 LOW findings; all incorporated above.
Operator decisions taken during the review:

- **D1**: `agentic-coding` registry flipped to `session.mode: sequential` with a
  provisional tool-execution `think_time_ms` — open_loop made `session_e2e_ms` an
  arrival-schedule artifact, corrupting the §12.4 SLO and the §14.1 users estimate.
- **D2**: E1 runs the new registered `smoke-synthetic` scenario (pipeline validation
  only; §8.2 exemption) instead of an illegal source-kind override on
  `chat-short-turns`.
- **D3**: λ counts **session starts** in all modes (§11.3 *What λ counts*), aligning
  arrival semantics with `scenario_mix` weights and the users math.

Structural outcomes: new M7 (Benchmarker orchestrator — previously unowned); Coordinator
and later milestones renumbered (M8–M10); `server_stats` scraper assigned to M2;
hardware sampler redesigned as stdlib-only and placed on the engine nodes (M3/M6/M7);
E2 split into E2a/E2b/E2c so the foundation gate is enforceable at E3's actual
multi-node topology; FirecREST staged-transfer + Coordinator resumability added to M8.

**Quality-evaluation extension — 2026-06-12** (follow-up to the comparative review
against SemiAnalysis InferenceMAX/InferenceX, see `COMPARATIVE_REVIEW.md`). Operator
decisions: **no standing quality anchor** — deltas are experiment-internal across the
experiment's deployment configs; **Stage-A sanity gate default-on** in every experiment,
skippable (`skip_quality_gate`); **Stage-B comparison** pairs capacity gains and quality
deltas in the same report (the "N× users at −M pts" claim); `ignore_eos` parameterized
as `output_length_mode: forced | natural` (§10.6). Structural outcomes: new **M11
Quality eval runner**; M0/M2/M7/M9 amended; E1 exercises the gate; E4's DoD now requires
the capacity-vs-quality pairing; spec gained §12.5 + §13.9 (`quality_evals` table).

## 11. Process gates

Per `CLAUDE.md` *How we work together*: each component milestone closes with phase 8
(adversarial review of implementation); E1/E3/E5 each get phase 10 (pre-execution
go/no-go), phases 12–16 (monitoring through adversarial results review), and E3
additionally phases 17–18 (conclusions + report). Skips require explicit operator
approval.
