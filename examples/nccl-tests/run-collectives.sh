#!/bin/bash
#
# Run a configurable set of NCCL collectives at the current rank topology.
# Executed by every rank under `srun --mpi=pmix`; nccl-tests binaries use
# PMIx to discover rank/size and only rank 0 prints results.
#
# Default collective set covers what LLM serving actually exercises:
#   - all_reduce   : tensor-parallel forward path (every TP step)
#   - all_gather   : TP with sequence parallelism; weight-gather paths
#   - alltoall     : MoE token-to-expert dispatch and reverse path
#
# nccl-tests also ships reduce_scatter_perf, broadcast_perf, sendrecv_perf,
# scatter_perf, gather_perf, reduce_perf, hypercube_perf — append to
# $COLLECTIVES to include them (e.g. "sendrecv" for pipeline-parallel
# serving).
#
# Required env:
#   NCCL_TESTS_VERSION   matches what build-nccl-tests.sh produced
#   NCCL_TESTS_CACHE     same as above
#   MSG_END              upper bound of the message-size sweep (e.g. "128M",
#                        "8G"). Lower bound is 8 B; factor x2.
# Optional:
#   COLLECTIVES          space-separated list (default: all_reduce all_gather alltoall)

set -eo pipefail

: "${NCCL_TESTS_VERSION:?must be set}"
: "${NCCL_TESTS_CACHE:?must be set}"
: "${MSG_END:?must be set (e.g. 128M or 8G)}"
: "${COLLECTIVES:=all_reduce all_gather alltoall}"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_stack-fingerprint.sh
source "${THIS_DIR}/_stack-fingerprint.sh"
BUILD_DIR="$(cache_dir_for_stack)/build"

rank0() { [ "${SLURM_PROCID:-0}" = "0" ]; }

fail=0
for c in ${COLLECTIVES}; do
    bin="${BUILD_DIR}/${c}_perf"
    if [ ! -x "${bin}" ]; then
        rank0 && echo "[nccl] missing ${bin} — was build-nccl-tests.sh called first?" >&2
        fail=1
        continue
    fi
    rank0 && echo "===== ${c} ====="
    "${bin}" -b 8 -e "${MSG_END}" -f 2 -g 1 \
        || { rank0 && echo "[nccl] ${c} failed (non-zero exit)" >&2; fail=1; }
done

exit "${fail}"
