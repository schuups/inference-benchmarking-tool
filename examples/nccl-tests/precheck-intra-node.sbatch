#!/bin/bash
#
# Intra-node NCCL + NVSHMEM pre-check: 1 node × 4 GH200 GPUs over NVLink-C2C.
#
# All steps run in the SAME engine container session (one srun, concatenated
# commands), matching the Kubernetes "bash -c 'precheck && exec engine'" form
# (SPECIFICATIONS.md §9.2). Pass the engine EDF via $ENGINE_EDF.
#
# Builds nccl-tests on first run (rank 0 only, others wait on sentinel) and
# caches under a stack-fingerprinted directory so a bumped engine image lands
# in a fresh cache. NVSHMEM is skipped with a warning if the engine image
# does not ship the perftest binaries.

#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=nccl-precheck-intra
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
#SBATCH --comment=inference-benchmarking

# Sweep parameters — the benchmark coordinator sets these before sbatch.
: "${NCCL_TESTS_VERSION:=2.17.1}"
: "${NCCL_TESTS_CACHE:=/capstor/scratch/cscs/$USER/nccl-tests-cache}"
: "${COLLECTIVES:=all_reduce all_gather alltoall}"
: "${MSG_END:=8G}"
: "${NVSHMEM_REQUIRED:=0}"
: "${NCCL_TESTS_MPI:=1}"   # this example launches one rank per GPU via srun --mpi=pmix
: "${ENGINE_EDF:?point ENGINE_EDF at the engine EDF path, e.g. /capstor/scratch/cscs/USER/experiment/engine.toml}"
export NCCL_TESTS_VERSION NCCL_TESTS_CACHE COLLECTIVES MSG_END NVSHMEM_REQUIRED NCCL_TESTS_MPI

# Canonical §8 pre-check scripts (single source of truth — this directory no longer
# keeps its own copies; see README.md).
PRECHECK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../tools/benchmarker/prechecks" && pwd)"
export PRECHECK_DIR

# One srun, one container session per rank. Each rank runs the full chain:
# build (rank 0 only, others wait), run NCCL collectives, run NVSHMEM perftest.
# In the real benchmarker, append `&& exec <engine command>` so the engine
# also runs in the same container session — identical libfabric/CUDA/NCCL.
srun --mpi=pmix --environment="${ENGINE_EDF}" bash -c '
    set -eo pipefail
    "${PRECHECK_DIR}/build-nccl-tests.sh"
    "${PRECHECK_DIR}/run-collectives.sh"
    "${PRECHECK_DIR}/run-nvshmem.sh"
'
