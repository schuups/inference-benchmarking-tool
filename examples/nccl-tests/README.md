# Communication-plane pre-check on Alps (HPE Slingshot 11, GH200)

Validates the **collective-communication plane** for the inference benchmarker.
One pre-check job covers:

- **NCCL collectives** — `all_reduce`, `all_gather`, `alltoall` (configurable).
- **NVSHMEM perftest** — one-sided put/get bandwidth and all-to-all latency,
  the path MoE engines (DeepEP, TRT-LLM MoE kernels) use when bypassing NCCL.

Both intra-node (NVLink-C2C between 4 GH200) and inter-node (Slingshot 11) cuts
are provided; NCCL exercises whichever links the rank topology requires.
Alps has no InfiniBand — the Slingshot 11 path reaches NCCL through the
**AWS OFI NCCL hook**, injected at launch by the CSCS Container Engine when the
engine EDF carries the right annotation.

> **Launch model — superseded for repo-built engine images.** The AWS OFI NCCL hook
> described here is the **hook-injection** model, for stock images that ship *without*
> the Alps network stack. The framework's repo-built engine images are
> **self-contained** — the stack is baked in, so they run with the CXI hook
> **disabled** and **no** aws-ofi hook (see `SPECIFICATIONS.md` §9.1). The live §8
> in-container pre-checks are implemented in `tools/benchmarker/prechecks/` (rendered
> into the engine launch by the Planner); the scripts in *this* directory are a
> standalone manual example / historical seed and may lag the canonical copies.

## Operating model: same container, concatenated commands

Per `SPECIFICATIONS.md` §8.2, the pre-check runs in the **same container instance**
as the engine launch that follows it — not in a separate image, not in an init
container. On both SLURM and Kubernetes the pattern is identical:

```
srun --environment=<edf> bash -c "precheck && exec <engine>"      # SLURM
command: ["bash", "-c", "precheck && exec <engine>"]              # K8s
```

A single container session means the pre-check and the engine see exactly the
same libfabric, CUDA, NCCL, and OpenMPI libraries — what the test measures is
the foundation the engine actually sits on.

The example sbatch files here run the precheck only (no `exec <engine>` tail) —
add the engine command when wiring this into a real experiment.

## Cache key includes the stack

Tests built against one CUDA / NCCL / OpenMPI / libfabric / `nccl-tests` combo
are not ABI-portable to another. The cache directory is therefore keyed by both
the `nccl-tests` version **and** a fingerprint of the tool versions detected in
the engine container at build time:

```
$NCCL_TESTS_CACHE/v<version>-<fingerprint>/build/
```

Any change to the engine image (new CUDA, new NCCL, new MPI, new libfabric)
lands in a fresh cache directory; older caches survive in parallel and are
reused if you revert. See `_stack-fingerprint.sh` for the exact fingerprint
inputs.

## Install-on-missing build dependencies

`build-nccl-tests.sh` tries to install `make`, `g++`, OpenMPI dev headers, `curl`,
and `tar` via `apt-get` / `dnf` / `yum` if they are missing. Only the rank holding
the build lock performs the install, so apt is not hammered by `N` concurrent ranks.
Install is skipped on cache-hit.

> **Limitation surfaced at E1.** The CSCS Container Engine runs containers
> **non-root**, so this apt/dnf fallback fails (`mpi.h: No such file or directory`)
> on a stock image that lacks the toolchain. The §8 pre-check therefore requires the
> MPI/NCCL build toolchain **baked into the engine image** — which the framework's
> self-contained Alps image provides (see `SPECIFICATIONS.md` §8.5 and `TODOs.md`).

## Files

| File | Role |
|---|---|
| `_stack-fingerprint.sh`, `build-nccl-tests.sh`, `run-collectives.sh`, `run-nvshmem.sh` | **Canonical copies live in `tools/benchmarker/prechecks/`** — the §8 pre-check scripts the tool actually runs. The sbatch files below source them from there via `PRECHECK_DIR`; no copies are kept in this directory. |
| `precheck-intra-node.sbatch` | 1 node × 4 GH200 (NVLink-C2C). `MSG_END=8G`. |
| `precheck-inter-node.sbatch` | 2 nodes × 4 GH200 (Slingshot 11). `MSG_END=128M` (CSCS reference). |

## Engine-EDF prerequisites

For multi-node NCCL bandwidth to be meaningful, the engine EDF that
`$ENGINE_EDF` points to must enable the AWS OFI NCCL hook and set the
CSCS-published env. Add to the engine EDF:

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
manually disables the hook and multi-node bandwidth collapses to ~5 GB/s
(CSCS warns explicitly).

## Sweep parameters (env vars)

The benchmark coordinator exports these before `sbatch`. Defaults are
sensible standalone values.

| Variable | Default | Meaning |
|---|---|---|
| `NCCL_TESTS_VERSION` | `2.17.1` | git tag of `github.com/NVIDIA/nccl-tests`. Pinned per experiment. |
| `NCCL_TESTS_CACHE` | `/capstor/scratch/cscs/$USER/nccl-tests-cache` | Persistent dir for source + compiled binaries; per-stack subdirs live underneath. |
| `COLLECTIVES` | `all_reduce all_gather alltoall` | Space-separated list. Append `reduce_scatter`, `sendrecv`, `broadcast` as needed. |
| `MSG_END` | `8G` (intra) / `128M` (inter) | Upper bound of the message-size sweep. |
| `NVSHMEM_REQUIRED` | `0` | `1` makes missing NVSHMEM in the engine image a hard failure. |
| `NVSHMEM_BIN_DIR` | *(auto-discover)* | Override path to the NVSHMEM perftest directory. |
| `NVSHMEM_TESTS` | `device/coll/alltoall_latency device/pt-to-pt/shmem_put_bw` | Relative paths from the perftest directory. |
| `ENGINE_EDF` | *(required)* | Path to the engine EDF. The pre-check uses this exact file — same image, mounts, env, and AWS OFI hook annotation as the engine job. |

## Running

```bash
export ENGINE_EDF=/capstor/scratch/cscs/$USER/my-experiment/engine.toml
sbatch precheck-intra-node.sbatch
sbatch precheck-inter-node.sbatch
```

First run on a fresh stack fingerprint compiles `nccl-tests` (~30 s) and may
install build deps (~10–20 s); subsequent runs are cache-hit.

## Reading the output

Each `<collective>_perf` block prints:

```
       size    count    type   redop    time     algbw    busbw   #wrong
```

The pre-check cares about **`busbw`** at large messages (≥ 128 MiB inter-node,
≥ 1 GiB intra-node). CSCS publishes the following reference points for the
`all_reduce` case:

| Topology | Message | Hook | Expected `busbw` (all_reduce) |
|---|---|---|---|
| 2 nodes × 4 GH200 | 128 MiB | **on** | ≈ 122 GB/s |
| 2 nodes × 4 GH200 | 128 MiB | **off** (broken) | ≈ 5 GB/s |

If a multi-node run reports ≈ 5 GB/s, the AWS OFI hook didn't fire. Check:

- `[annotations]` in the engine EDF carries `com.hooks.aws_ofi_nccl.enabled = "true"`.
- You did **not** set `NCCL_NET_PLUGIN` anywhere.
- The image the EDF references is the one you actually pushed.

NVSHMEM `alltoall_latency` reports microseconds; `shmem_put_bw` reports GB/s.
Reference values are per-image and tracked in `SPECIFICATIONS.md` §8.3.

## References

- [CSCS NCCL docs](https://docs.cscs.ch/software/communication/nccl/)
- [CSCS Container Engine — AWS OFI NCCL hook](https://docs.cscs.ch/software/container-engine/resource-hook/#aws-ofi-nccl-hook)
- [CSCS Container Engine — EDF reference](https://docs.cscs.ch/software/container-engine/edf/#edf-reference)
- [CSCS libfabric docs](https://docs.cscs.ch/software/communication/libfabric/)
- [NVIDIA NCCL env vars](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA NVSHMEM](https://developer.nvidia.com/nvshmem)
- [`fi_cxi(7)`](https://ofiwg.github.io/libfabric/v2.1.0/man/fi_cxi.7.html)
- [`NVIDIA/nccl-tests`](https://github.com/NVIDIA/nccl-tests)
