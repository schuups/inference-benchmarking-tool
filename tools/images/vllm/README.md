# vLLM engine image (§8.1)

Two-stage build, vendored from
[schuups/alps-extended-images](https://github.com/schuups/alps-extended-images)
`vllm-image-test` @ `0431e103` (provenance + pins in `build-args.env`):

1. **`Containerfile.alps-base`** — NGC vLLM base + the Alps HPC network stack:
   Cassini headers, CXI driver, libcxi (SHS 13.0), patched libfabric, pinned +
   patched NCCL, aws-ofi-nccl, NVSHMEM, UCX/UCC/OMPI, nccl-tests, OSU. The
   `patches/` and `common/` trees are tracked here per §8.1 review discipline.
2. **`Containerfile`** — pins vLLM (`VLLM_VERSION`) on top of the alps base.

**Build** (SLURM job on clariden; podman; ~no laptop path — no NVIDIA GPU /
HPC interconnect there):

```sh
# stage the build context to the cluster, then:
sbatch build.sbatch
```

Pushes both tags to `jfrog.svc.cscs.ch/ml/inference` (docker-type repo).
Operator prerequisites (once): §3 podman storage.conf; `~/.ib_jfrog_token`.

**Status**: images from this lineage are NOT yet inference-validated —
iterations expected. The §7 pre-checks run inside this image are the
validation instrument (NCCL/NVSHMEM over CXI must hit reference bandwidth).
