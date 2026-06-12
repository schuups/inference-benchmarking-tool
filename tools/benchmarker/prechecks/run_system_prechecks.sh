#!/bin/bash
#
# System performance pre-checks entry point (SPECIFICATIONS.md §7.2).
#
# Runs INSIDE the engine container session, concatenated with the engine
# launch by the M6 templates:
#     srun --environment=<edf> bash -c "run_system_prechecks.sh && exec <engine>"
# A non-zero exit here stops the chain — the engine never starts on an
# aborting gate (§7.4).
#
# Required env (rendered by the Planner from the benchmark YAML + global.yaml):
#   PRECHECK_CLUSTER        e.g. clariden
#   PRECHECK_SCOPE          reference scope, e.g. "4× GH200, 1 node"
#   PRECHECK_OUT            writable dir for captured outputs + results.json
#   NCCL_TESTS_VERSION      collective_tests_version (§7.2)
#   NCCL_TESTS_CACHE        collective_tests_cache_dir (§7.2)
#   MSG_END                 collectives sweep upper bound (e.g. 128M)
# Optional:
#   SKIP_SYSTEM_PRECHECKS   1 → skip everything (§7.5; use sparingly)
#   PRECHECK_ON_WARN        abort|continue (default abort, §7.4)
#   PRECHECK_ON_FAIL        abort|continue (default abort, §7.4)
#   PRECHECK_STORAGE_SCOPE  reference scope of the weights mount
#   STORAGE_BENCH_FILE      large file on the weights mount to read (storage check
#                           skipped if unset)
#   NVSHMEM_REQUIRED        1 → missing NVSHMEM SDK is a failure (§7.1)
#
# Exit codes: 0 proceed | 3 warn+abort | 4 fail+abort | 1 internal error.

set -eo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank0() { [ "${SLURM_PROCID:-0}" = "0" ]; }

if [ "${SKIP_SYSTEM_PRECHECKS:-0}" = "1" ]; then
    rank0 && echo "[prechecks] SKIPPED via skip_system_prechecks (§7.5) — use sparingly"
    exit 0
fi

: "${PRECHECK_CLUSTER:?must be set}"
: "${PRECHECK_SCOPE:?must be set}"
: "${PRECHECK_OUT:?must be set}"
: "${NCCL_TESTS_VERSION:?must be set}"
: "${NCCL_TESTS_CACHE:?must be set}"
: "${MSG_END:=128M}"

mkdir -p "${PRECHECK_OUT}"

# ------------------------------------------------------------- build (cached)
# Cache miss => smoke-test mode (§7.2): pipeline still runs end-to-end but the
# orchestrator must not persist results. The flag is part of results.json.
# shellcheck source=_stack-fingerprint.sh
source "${THIS_DIR}/_stack-fingerprint.sh"
SMOKE_FLAG=""
if [ ! -f "$(cache_dir_for_stack)/build/.built" ]; then
    rank0 && echo "[prechecks] collective-tests cache MISS — entering smoke-test mode (§7.2)"
    SMOKE_FLAG="--smoke"
fi
bash "${THIS_DIR}/build-nccl-tests.sh"

# ------------------------------------------------------------- collectives
# run-collectives.sh prints one nccl-tests table per collective; split the
# capture per collective so grade.py maps files onto reference rows.
for c in all_reduce all_gather alltoall; do
    COLLECTIVES="$c" bash "${THIS_DIR}/run-collectives.sh" \
        > "${PRECHECK_OUT}/collective_${c}.out" 2>&1 || {
        rank0 && echo "[prechecks] ${c} errored (graded as fail):" && tail -5 "${PRECHECK_OUT}/collective_${c}.out"
    }
done

# ------------------------------------------------------------- NVSHMEM / SHMEM
NVSHMEM_TESTS="device/coll/alltoall_latency" \
    bash "${THIS_DIR}/run-nvshmem.sh" > "${PRECHECK_OUT}/nvshmem_alltoall_latency.out" 2>&1 || true
NVSHMEM_TESTS="device/pt-to-pt/shmem_put_bw" \
    bash "${THIS_DIR}/run-nvshmem.sh" > "${PRECHECK_OUT}/nvshmem_put_bw.out" 2>&1 || true
# run-nvshmem.sh exits 0 with a warning when the SDK is absent (skipped row,
# §7.1) unless NVSHMEM_REQUIRED=1, in which case the `|| true` above must not
# mask it — re-check explicitly:
if [ "${NVSHMEM_REQUIRED:-0}" = "1" ]; then
    grep -q "NVSHMEM perftest not found" "${PRECHECK_OUT}/nvshmem_alltoall_latency.out" && {
        rank0 && echo "[prechecks] NVSHMEM required but absent — fail (§7.1)" >&2
        exit 4
    }
fi

# ------------------------------------------------------------- storage read
# O_DIRECT sequential read of a real weights file (§7.1) — rank 0 only.
if rank0 && [ -n "${STORAGE_BENCH_FILE:-}" ] && [ -r "${STORAGE_BENCH_FILE}" ]; then
    dd if="${STORAGE_BENCH_FILE}" of=/dev/null bs=1M count=4096 iflag=direct \
        > "${PRECHECK_OUT}/storage_read.out" 2>&1 || true
fi

# ------------------------------------------------------------- grade (rank 0)
if rank0; then
    python3 "${THIS_DIR}/grade.py" \
        --out-dir "${PRECHECK_OUT}" \
        --cluster "${PRECHECK_CLUSTER}" \
        --scope "${PRECHECK_SCOPE}" \
        --storage-scope "${PRECHECK_STORAGE_SCOPE:-}" \
        --on-warn "${PRECHECK_ON_WARN:-abort}" \
        --on-fail "${PRECHECK_ON_FAIL:-abort}" \
        ${SMOKE_FLAG} \
        --results "${PRECHECK_OUT}/results.json"
fi
