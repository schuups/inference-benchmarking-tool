#!/bin/bash
#
# Inter-node NCCL + NVSHMEM pre-check: 2 nodes × 4 GH200 GPUs over HPE Slingshot 11.
#
# All steps run in the SAME engine container session (one srun, concatenated
# commands), matching the Kubernetes "bash -c 'precheck && exec engine'" form
# (SPECIFICATIONS.md §8.2). Pass the engine EDF via $ENGINE_EDF.
#
# This is the bandwidth that bounds multi-node TP / PP / MoE inference. If
# NCCL all_reduce busbw at 128 MiB is far below the CSCS reference (~122 GB/s),
# the AWS OFI NCCL hook likely didn't fire — see README.md.

#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=nccl-precheck-inter
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
#SBATCH --comment=inference-benchmarking

: "${NCCL_TESTS_VERSION:=2.17.1}"
: "${NCCL_TESTS_CACHE:=/capstor/scratch/cscs/$USER/nccl-tests-cache}"
: "${COLLECTIVES:=all_reduce all_gather alltoall}"
: "${MSG_END:=128M}"
: "${NVSHMEM_REQUIRED:=0}"
: "${NCCL_TESTS_MPI:=1}"   # this example launches one rank per GPU via srun --mpi=pmix
: "${ENGINE_EDF:?point ENGINE_EDF at the engine EDF path, e.g. /capstor/scratch/cscs/USER/experiment/engine.toml}"
export NCCL_TESTS_VERSION NCCL_TESTS_CACHE COLLECTIVES MSG_END NVSHMEM_REQUIRED NCCL_TESTS_MPI

# Canonical §7 pre-check scripts (single source of truth — this directory no longer
# keeps its own copies; see README.md).
PRECHECK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../tools/benchmarker/prechecks" && pwd)"
export PRECHECK_DIR

srun --mpi=pmix --environment="${ENGINE_EDF}" bash -c '
    set -eo pipefail
    "${PRECHECK_DIR}/build-nccl-tests.sh"
    "${PRECHECK_DIR}/run-collectives.sh"
    "${PRECHECK_DIR}/run-nvshmem.sh"
'
