#!/bin/bash
#
# Inter-node NCCL pre-check: 2 nodes × 4 GH200 GPUs over HPE Slingshot 11.
#
# Runs in the SAME engine container as the vLLM job would (engine EDF passed
# via $ENGINE_EDF), per SPECIFICATIONS.md §8.2. Build is cached; first run
# compiles, subsequent runs reuse.
#
# This is the bandwidth that bounds multi-node TP / PP / MoE inference. If
# busbw at 128 MiB is far below the CSCS reference for the engaged collective,
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
: "${ENGINE_EDF:?point ENGINE_EDF at the engine EDF path, e.g. /capstor/scratch/cscs/USER/experiment/engine.toml}"
export NCCL_TESTS_VERSION NCCL_TESTS_CACHE COLLECTIVES MSG_END

PRECHECK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

srun -n 1 --environment="${ENGINE_EDF}" \
    "${PRECHECK_DIR}/build-nccl-tests.sh"

srun --mpi=pmix --environment="${ENGINE_EDF}" \
    "${PRECHECK_DIR}/run-collectives.sh"
