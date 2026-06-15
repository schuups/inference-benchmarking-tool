# E2 foundation & pipeline validation — clariden (2026-06-15)

**Curated synthesis (§15.3)** of the E2a/E2b runs. These are **pipeline- and foundation-validation
runs on the `smoke-synthetic` scenario — NOT capacity findings** (the scenario forbids capacity /
Pareto / procurement use; runs are under-saturated, so each λ\* below is the *swept ceiling*, not a
measured limit). Per-run notebooks + plots are under `experiments/`.

## What these runs establish

| | E2a — single-node | E2b — multi-node |
|---|---|---|
| Deployment | Apertus-8B, 4× GH200, 1 node, TP4 | Apertus-70B, 8× GH200, 2 nodes, TP4×PP2 |
| Run | `…_73e6` | `…_6199` |
| Result | 228 req, 0 err, persisted | 228 req, 0 err, persisted |
| Report | [experiments/2026-06-15_e2a-single-node-clariden/…_73e6/](../experiments/2026-06-15_e2a-single-node-clariden/) | [experiments/2026-06-15_e2b-multinode-clariden/…_6199/](../experiments/2026-06-15_e2b-multinode-clariden/) |

## §8 foundation reference (now enforceable)

NCCL busbw @128 MiB / NVSHMEM alltoall @128 KiB — intra-node NVLink (1-node) vs inter-node Slingshot-11 (2-node):

| metric | 4× GH200, 1 node | 8× GH200, 2 nodes |
|---|---|---|
| NCCL all_reduce | 317.7 | 131.1 |
| NCCL all_gather | 283.5 | 86.5 |
| NCCL alltoall | 306.2 | 38.3 |
| NCCL sendrecv (PP link) | — | 23.5 (≈ one Slingshot NIC) |
| NVSHMEM alltoall_latency (µs) | 12.6 | 47.9 |

Storage (capstor, single-stream O_DIRECT floor): ~0.07–0.19 GB/s; parallel ~0.14; buffered/readahead ~3.9
(state-dependent). All populated in `tools/system_prechecks_reference.yaml`.

## Notable findings

- **70B weight load is storage-bound**: `model_load_weights_s` = 95.6 s for 16.45 GiB/worker ≈ **0.185 GB/s**,
  matching the single-stream capstor floor — 8 concurrent workers contend, no `safetensors_load_strategy=prefetch`.
  The buffered-read metric (3.9 GB/s) is optimistic vs the real concurrent load. (`model_load_total_s` = 665 s is
  the gross submit→ready wait — scheduling + pre-checks + Ray + load — not pure load.)
- **Multi-node engine** required three fixes (see TODOs / `e2b-multinode-clariden.yaml`): Ray (absent from the
  image → scratch-`--target` install, interim), `--disable-custom-all-reduce` + `--enforce-eager` (cross-node
  GH200 CUDA faults). The §8 dedicated pre-check worked first try; the engine took the iterations.

## Next for real findings

The capacity report (λ\* knee, per-class SLO attainment, supportable-users) needs a **real-scenario** run
(per-model `scenario_mix`, swept to saturation), not smoke. Single-node is unblocked now; multi-node awaits the
Ray-in-image rebuild.
