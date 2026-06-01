#!/bin/bash
#
# Intra-node NCCL pre-check: 1 node × 4 GH200 GPUs over NVLink-C2C.
#
# Runs in the SAME engine container as the vLLM job would (engine EDF passed
# via $ENGINE_EDF), per SPECIFICATIONS.md §8.2. Build is cached on persistent
# storage; first run compiles nccl-tests, subsequent runs reuse the binaries.
#
# Collective set covers TP, sequence-parallel, and MoE expert dispatch:
#   all_reduce, all_gather, alltoall. Override via COLLECTIVES.

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
: "${ENGINE_EDF:?point ENGINE_EDF at the engine EDF path, e.g. /capstor/scratch/cscs/USER/experiment/engine.toml}"
export NCCL_TESTS_VERSION NCCL_TESTS_CACHE COLLECTIVES MSG_END

PRECHECK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Build once (cached). One rank only.
srun -n 1 --environment="${ENGINE_EDF}" \
    "${PRECHECK_DIR}/build-nccl-tests.sh"

# 2) Run the collective set at the engine's intra-node rank topology.
srun --mpi=pmix --environment="${ENGINE_EDF}" \
    "${PRECHECK_DIR}/run-collectives.sh"

# In the real benchmark, the engine srun follows here (same allocation, same
# EDF). This example stops after the pre-check.
