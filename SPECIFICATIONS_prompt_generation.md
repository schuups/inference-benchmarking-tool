# Prompt Generation — Working Draft

> Extracted from `SPECIFICATIONS.md` §6 for focused editing. Once finalized, merge back
> into `SPECIFICATIONS.md` as the new §6 (subsection numbering will follow the parent file).

## 6. Prompt Generation

### 6.1 Location

Prompts are generated **on the SLURM benchmarker node** from `dataset_config` in `run_config.json`.
This avoids the FirecREST 5 MB direct-upload limit and allows arbitrarily large prompt pools.
The coordinator no longer uploads `prompts.json`.

### 6.2 Prompt uniqueness requirement

Every prompt must start with a distinct token block so that the vLLM prefix cache does not serve
synthetic cache hits. Requirements:
- Each prompt begins with a unique `[prompt-NNNNNN]` or `[session-NNNNNN]` header.
- Without this, filler-text prompts share identical first blocks → 100% cache hit rate →
  TTFT drops to ~100 ms regardless of server load (artefact, not real performance).

### 6.3 Supported dataset sources

Controlled by `dataset_config.dataset_source` in the benchmark YAML:

| Value | Description |
|---|---|
| `"synthetic"` (default) | Filler text with unique `[prompt-NNNNNN]` headers. No network required. |
| `"longbench"` | LongBench code tasks (`lcc` + `repobench-p`) downloaded from HuggingFace via `urllib` as a single `data.zip`. No extra libraries required. Falls back to synthetic on failure. |

### 6.4 LongBench specifics

- Tasks: `lcc` (Long Code Completion, real Python/C++/Java files, ~13–22 K tokens) and
  `repobench-p` (repository-level Python completion, ~14–22 K tokens).
- Content: real GitHub repositories, appropriate for speculative decoding acceptance rate
  measurement with same-family draft/target model pairs.
- Length filter: examples are accepted if their token count falls within 40–160% of `input_length.mean`.
- Pool is repeated (with unique session headers) if `num_prompts` exceeds available examples.

### 6.5 Notes on dataset suitability

- **Synthetic prompts**: acceptable for latency and throughput benchmarking but produce
  near-zero speculative decoding acceptance rates (random text is unpredictable).
- **LongBench / real code**: required for meaningful speculative decoding acceptance rate
  measurements. Both Apertus-8B and Apertus-70B were trained on the same data, so
  same-family speculative acceptance rates should be 0.5–0.7 on real code.
