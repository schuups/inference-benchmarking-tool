# TODOs

## Architecture & Engine

- [ ] Define which files and folders are immutable (cannot be modified when running experiments)
- [ ] Establish a global configuration location for shared values (e.g. Python binary path, file paths like `/capstor/.../stefschu`)
- [ ] Structure the tool modularly so the core engine is rarely (ideally never) modified; new capabilities added as modules

## Experiment Execution

- [ ] Run NCCL benchmarks (using the same Docker images) before inference benchmarks
- [ ] Support testing endpoints provided via URL (not only SLURM and Kubernetes deployments)

## Docker Image Builds

- [ ] Support building Docker images via SLURM jobs
- [ ] Define and configure JFrog folder/path for publishing built images

## Prompt / Dataset Generation

- [ ] Support multi-modal prompt length distributions in the dataset generator
- [ ] **Audio and video modalities** — v1 covers text and image only (SPECIFICATIONS.md
  §7.5, §7.11); add `audio` and `video` as additional modalities, with `audio_corpus` /
  `video_corpus` source kinds, per-second / per-clip token-cost accounting, and
  registry-load-time acceptance of `modalities: [audio]` / `modalities: [video]`.
- [ ] **Confirm scenario-registry replaces `dataset_source`** (SPECIFICATIONS.md §7) — the
  previous design had a top-level `dataset_source` knob; the current spec embeds source
  choice inside the scenario registry. Confirm this is the intended end state (no separate
  `dataset_source` at the benchmark-YAML level beyond `source_overrides`).
- [ ] **`source_overrides` schema per source kind** (§7.4) — enumerate the keys each source
  accepts (e.g. LongBench task subset, COCO split, reasoning-trace dataset name). The
  field is declared but its per-source schema is not specified.
- [ ] **Fan-out template DSL for agentic scenarios** (§7.10) — specify the grammar of the
  fan-out template (turn types, allowed transitions, cycle-count distribution). Options:
  inline YAML state-machine in the scenario entry, a small Python DSL, or replay-only
  from recorded agent traces (SWE-Bench Verified, τ-bench, …).
- [ ] **Image token-cost fallback table** (§7.11) — the per-model per-image token estimate
  needs a structured home (probably alongside the tokenizer ID in a model-info registry).
  Numbers are not in the registry yet.
- [ ] **Think-time distribution defaults** (§7.3) — recommend a default range per scenario
  for the sequential-session `think_time_ms` field (chat short, agentic-coding longer, …).
  Currently no defaults are specified.
- [ ] **Scenario-registry revision pinning for reproducibility** (§7.14) — define how the
  scenario-registry revision is "recorded alongside" the dataset: git SHA on the
  experiment row? content hash of the YAML file? Pick a storage location and mechanism.
- [ ] **Per-tool `result_content_source` synthesis** (§7.10) — specify how a tool-result
  body of a sampled token length is produced from each `result_content_source.kind`
  (`synthetic` / `longbench` / `static`). The enum exists; the synthesis mechanics do
  not.

## Metrics & Analysis

- [ ] Derive and expose the number of supportable concurrent users from λ (load-to-users translation)

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
