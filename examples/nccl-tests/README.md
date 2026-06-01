# NCCL pre-check on Alps (HPE Slingshot 11, GH200)

Validates the **collective-communication plane** for the inference benchmarker.
One pre-check covers both:

- **Intra-node** GPU-to-GPU bandwidth (NVLink-C2C between the 4 GH200 on one node)
- **Inter-node** fabric bandwidth (HPE Slingshot 11)

The same NCCL test exercises whichever links the requested rank topology requires.
Alps has no InfiniBand; the Slingshot 11 path is reached through the
**AWS OFI NCCL hook**, injected at launch by the CSCS Container Engine when the
EDF carries the right annotation.

## Operating model: runs in the engine container

Per `SPECIFICATIONS.md` §8.2, the pre-check runs in the **same allocation, the
same EDF, and the same container image** as the engine launch that follows it —
not in a separate image. The example here is a thin pair of sbatch files that
build `nccl-tests` once (cached on persistent storage), then run a small set of
collectives at the engine's rank topology, all using your engine's EDF.

There is therefore **no Containerfile in this directory**. Two prerequisites the
engine image and engine EDF must already satisfy:

1. **Build tools in the engine image** — `nccl-tests` is compiled at first use
   from inside the engine container. The image needs `make`, `g++`, `curl`,
   `tar`, and `libopenmpi-dev` + `openmpi-bin` (or equivalent MPI dev). If
   any are missing, `build-nccl-tests.sh` exits with a clear error. The Alps
   engine image (`examples/docker-images-build/`) is the place to add them.
2. **AWS OFI NCCL hook and env in the engine EDF** — without these the
   multi-node bandwidth collapses to ~5 GB/s. Add to the EDF:
   ```toml
   [annotations]
   "com.hooks.aws_ofi_nccl.enabled" = "true"
   "com.hooks.aws_ofi_nccl.variant" = "cuda12"

   [env]
   NCCL_NET                    = "AWS Libfabric"
   NCCL_NET_GDR_LEVEL          = "PHB"
   NCCL_CROSS_NIC              = "1"
   NCCL_PROTO                  = "^LL128"
   NCCL_NCHANNELS_PER_NET_PEER = "4"
   PMIX_MCA_psec               = "native"
   FI_CXI_DEFAULT_CQ_SIZE      = "131072"
   FI_CXI_DEFAULT_TX_SIZE      = "16384"
   FI_CXI_DISABLE_HOST_REGISTER = "1"
   FI_CXI_RX_MATCH_MODE        = "software"
   FI_MR_CACHE_MONITOR         = "userfaultfd"
   MPICH_GPU_SUPPORT_ENABLED   = "0"
   ```
   **Do not** set `NCCL_NET_PLUGIN` — the hook sets it internally; setting it
   manually disables the hook (CSCS warns explicitly).

## What gets tested

The default collective set is the union of what dense and MoE LLM serving
actually exercises:

| Collective | What it bounds in inference |
|---|---|
| `all_reduce` | Tensor-parallel forward path (every TP step). |
| `all_gather` | TP with sequence-parallel attention; weight-gather paths. |
| `alltoall` | MoE token-to-expert dispatch and the reverse path. **Specifically important for Kimi-K2.6, DeepSeek-V4 Pro, GLM-5.1.** |

Reduce-scatter, send/recv, broadcast, etc. are also built into the cache by
`build-nccl-tests.sh` — add them to `COLLECTIVES` if your engine uses them
(e.g. `sendrecv` for pipeline-parallel serving).

## Files

| File | Role |
|---|---|
| `build-nccl-tests.sh` | Idempotent build of `nccl-tests` `v$NCCL_TESTS_VERSION` into `$NCCL_TESTS_CACHE/v<version>/build/`. Sentinel-gated, atomic-publish. Run via `srun -n 1`. |
| `run-collectives.sh` | Per-rank entrypoint executed under `srun --mpi=pmix`. Loops over `$COLLECTIVES`, runs each `<name>_perf -b 8 -e $MSG_END -f 2 -g 1`. |
| `precheck-intra-node.sbatch` | 1 node × 4 GH200 (NVLink-C2C). Default `MSG_END=8G`. |
| `precheck-inter-node.sbatch` | 2 nodes × 4 GH200 (Slingshot 11). Default `MSG_END=128M` (CSCS reference message size). |

## Sweep parameters (env vars)

All four sbatch invocations honour these env vars; the benchmark coordinator
exports them before `sbatch`. Defaults are sensible standalone values.

| Variable | Default | Meaning |
|---|---|---|
| `NCCL_TESTS_VERSION` | `2.17.1` | git tag of `github.com/NVIDIA/nccl-tests`. Pinned per experiment for reproducibility; bumping it triggers a re-build into a separate cache directory (old cache stays). |
| `NCCL_TESTS_CACHE` | `/capstor/scratch/cscs/$USER/nccl-tests-cache` | Persistent dir for source + compiled binaries. Shared across all experiments by default. |
| `COLLECTIVES` | `all_reduce all_gather alltoall` | Space-separated list of collectives to run. |
| `MSG_END` | `8G` (intra) / `128M` (inter) | Upper bound of the message-size sweep. |
| `ENGINE_EDF` | *(required)* | Path to the engine's EDF. The pre-check uses this exact file — same image, mounts, env, and AWS OFI hook annotation as the engine job. |

## Running

```bash
# Point at the engine EDF the pre-check should use (this is the file that
# encodes the AWS OFI hook annotation + NCCL env above):
export ENGINE_EDF=/capstor/scratch/cscs/$USER/my-experiment/engine.toml

# Optional overrides:
export NCCL_TESTS_VERSION=2.17.1
export COLLECTIVES="all_reduce all_gather alltoall"

# Submit.
sbatch precheck-intra-node.sbatch
sbatch precheck-inter-node.sbatch
```

First run on a fresh cache compiles `nccl-tests` (~30 s); subsequent runs are
build-free.

## Reading the output

Each `<collective>_perf` block prints:

```
       size    count    type   redop    time     algbw    busbw   #wrong
```

The pre-check cares about **`busbw`** at large messages (≥ 128 MiB on inter-node,
≥ 1 GiB on intra-node). CSCS publishes the following reference points for the
`all_reduce` case; use them as a sanity gate:

| Topology | Message | Hook | Expected `busbw` (all_reduce) |
|---|---|---|---|
| 2 nodes × 4 GH200 | 128 MiB | **on** | ≈ 122 GB/s |
| 2 nodes × 4 GH200 | 128 MiB | **off** (broken) | ≈ 5 GB/s |

A 1-node intra-node `all_reduce` should comfortably exceed the multi-node
number. `alltoall` busbw is typically lower than `all_reduce` because every
rank exchanges with every rank — but the same hook-on / hook-off ratio
applies as a degradation indicator.

If a multi-node run reports ≈ 5 GB/s on `all_reduce`, the AWS OFI hook didn't
fire. Check:

- `[annotations]` in the engine EDF carries `com.hooks.aws_ofi_nccl.enabled = "true"`.
- You did **not** set `NCCL_NET_PLUGIN` anywhere.
- The image the EDF references is the one you actually pushed (re-pull / re-tag
  if unsure).

## Why this replaces both `iperf3` and `bandwidthTest`

NCCL exercises the same Slingshot 11 path `iperf3` would, but in the
configuration the engine actually uses (rank-to-rank collectives over AWS OFI
plugin, not point-to-point TCP). HBM bandwidth — what `bandwidthTest` measures —
is a stable per-SKU characteristic that rarely degrades in isolation without
also degrading the collectives above, so a separate test adds maintenance
without adding signal. See
[`SPECIFICATIONS.md` §8](../../SPECIFICATIONS.md#8-system-performance-pre-checks).

## References

- [CSCS NCCL docs](https://docs.cscs.ch/software/communication/nccl/)
- [CSCS Container Engine — AWS OFI NCCL hook](https://docs.cscs.ch/software/container-engine/resource-hook/#aws-ofi-nccl-hook)
- [CSCS Container Engine — EDF reference](https://docs.cscs.ch/software/container-engine/edf/#edf-reference)
- [CSCS libfabric docs](https://docs.cscs.ch/software/communication/libfabric/)
- [NVIDIA NCCL env vars](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [`fi_cxi(7)`](https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html)
- [`NVIDIA/nccl-tests`](https://github.com/NVIDIA/nccl-tests)
