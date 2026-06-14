# Engine images

Inference-engine container images extended with the **Alps HPC network stack**
(Slingshot-11 / GH200 / MI300A), built from sources in this repo and published to
the CSCS JFrog Artifactory (SPECIFICATIONS.md §9.1). Each image is identified by
**vendor × backend × backend-version × netstack-version** and is self-contained:
the network libraries are baked in, so no container-engine CXI / aws-ofi-nccl hook
is needed at runtime.

The network-stack component set, pins, runtime env vars, and Slingshot-11 launch
rules follow the [CSCS — Alps extended images documentation](https://docs.cscs.ch/software/alps-extended-images/).

## Image catalogue

Status legend: `pending-build` → `building` → `built` → `pushed` →
`sanity-pending` → **`verified`** → **`benchmarked`** · `failed`.

- **`verified`** — netstack sanity green: components load and inter-node collectives
  hit the Slingshot reference bandwidth. Functionally usable; inference
  **performance scaling not yet tested**.
- **`benchmarked`** — additionally validated through **inference performance-scaling**
  experiments (real serving workloads, multi-GPU/multi-node scaling characterised).

| Image (slug) | Vendor | Backend | Version | Netstack | Tag | Status | Sanity |
|---|---|---|---|---|---|---|---|
| [`nvidia-vllm-0.22.1-net.v1`](nvidia-vllm-0.22.1-net.v1/) | nvidia | vllm | 0.22.1 | nvidia/v1 | `…/vllm:0.22.1-alps.net.v1` | **verified** | pass (2-node) |

<!-- Keep this table in sync with each image's manifest.yaml (status + sanity). -->

## Layout

```
tools/images/
  README.md                      this catalogue (mirrors each image's manifest.yaml)
  core/                          shared, reused across images
    common/                      generic, version-agnostic env/warning installers
    nvidia/netstack/v1/          NVIDIA Alps stack v1: Containerfile + phases/ + patches/
      runtime/                   the env tuning (NCCL/FI_CXI/NVSHMEM/...) + rdzv warning
  <vendor>-<backend>-<ver>-net.<n>/   one directory per image
    manifest.yaml                identity, base, pins, status, provenance
    variant/hooks.d/             optional per-image late build hooks
    tests/                       post-push sanity (component-load + collectives)
```

- **Netstack is shared and versioned.** All NVIDIA backends (vllm / sglang /
  dynamo) reuse `core/nvidia/netstack/<v>`; AMD gets its own family
  (`core/amd/netstack/<v>`, RCCL / ROCm-SHMEM) when added. Maturity is the `vN`
  axis: `v1` is today's proven stack; a `v2` lands as a sibling, leaving `v1`
  reproducible.
- **Each image is a thin manifest** selecting a base image + a netstack version +
  pin overrides; the Containerfile lives once per netstack version.

## Building (persistent allocation + `srun --overlap`)

Image builds are expensive and run **in node-local `/dev/shm`** (podman
`storage.conf`), so the layer cache lives in RAM and is lost when the node is
released. FirecREST does not expose `srun`, so builds are driven over SSH against
a **held allocation**:

```sh
# 1. Hold a build node for the session (a 1-node sleep holder job submitted via
#    FirecREST, or salloc) and note its JOBID — keeps the /dev/shm cache warm.

# 2. Stage context + build + push (reuses the warm cache each iteration); writes a
#    digest-pinned EDF to the image's tests/ on cluster scratch for sanity:
tools/images/build.sh <image-slug> --jobid <JOBID>

# 3. Post-push sanity — submit the short-lived 2-node job via FirecREST with the
#    digest-pinned EDF build.sh wrote:
#      sbatch (env SANITY_EDF=.../<slug>/tests/edf-pinned.toml) tools/images/sanity.sbatch
```

Per-phase layering means a failed/edited phase rebuilds only from that step onward
(everything earlier is a cache hit) — provided the same allocation (hence the same
`/dev/shm`) is reused.

- **`build.sh`** reads the image `manifest.yaml` (base, tag, netstack pins → `--build-arg`),
  stages the composed context, and drives `podman build`/`push` over `ssh`+`srun`.
- **`sanity.sbatch`** runs the 2-node NCCL/OSU/NVSHMEM validation and passes only if the
  inter-node `all_reduce` busbw clears a Slingshot floor (catches the "plugin didn't fire"
  ~5 GB/s failure). Pinning the EDF by registry digest avoids enroot's stale tag cache.
