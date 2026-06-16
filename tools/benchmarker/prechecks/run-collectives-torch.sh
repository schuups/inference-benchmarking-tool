#!/bin/bash
#
# Launch the torch.distributed collective probe (collective_probe.py) one rank per GPU — the
# K8s §8 foundation check (SPECIFICATIONS.md §8.2, IMPLEMENTATION_PLAN.md decision 9). It
# replaces the MPI nccl-tests SLURM uses, because K8s has no srun/PMIx; the probe measures the
# same all_reduce / all_gather / alltoall busbw on vLLM's own torch+NCCL stack and writes the
# same collective_<c>.out rows grade.py reads.
#
# SINGLE-NODE / 1 pod (this script): torchrun --standalone spawns the ranks locally.
# MULTI-NODE (future, decision 9): a Ray placement launcher sets RANK/WORLD_SIZE/MASTER_ADDR
# across pods and runs collective_probe.py directly — same probe, different rendezvous (TODOs).
#
# Required env: PRECHECK_OUT (writable), PRECHECK_GPUS (ranks = GPUs in the pod).
# Optional: COLLECTIVES (default "all_reduce all_gather alltoall"), MSG_BYTES (default 128 MiB).

set -eo pipefail

: "${PRECHECK_OUT:?must be set}"
: "${PRECHECK_GPUS:?must be set}"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun --standalone --nnodes=1 --nproc-per-node="${PRECHECK_GPUS}" \
    "${THIS_DIR}/collective_probe.py" \
    --collectives "${COLLECTIVES:-all_reduce all_gather alltoall}" \
    --bytes "${MSG_BYTES:-134217728}" \
    --out-dir "${PRECHECK_OUT}"
