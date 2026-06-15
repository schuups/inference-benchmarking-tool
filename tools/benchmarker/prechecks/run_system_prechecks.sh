#!/bin/bash
#
# System performance pre-checks entry point (SPECIFICATIONS.md §8.2).
#
# Runs INSIDE the engine container session, concatenated with the engine
# launch by the M6 templates:
#     srun --environment=<edf> bash -c "run_system_prechecks.sh && exec <engine>"
# A non-zero exit here stops the chain — the engine never starts on an
# aborting gate (§8.4).
#
# Required env (rendered by the Planner from the benchmark YAML + global.yaml):
#   PRECHECK_CLUSTER        e.g. clariden
#   PRECHECK_SCOPE          reference scope, e.g. "4× GH200, 1 node"
#   PRECHECK_OUT            writable dir for captured outputs + results.json
#   NCCL_TESTS_VERSION      collective_tests_version (§8.2)
#   NCCL_TESTS_CACHE        collective_tests_cache_dir (§8.2)
#   MSG_END                 collectives sweep upper bound (e.g. 128M)
# Optional:
#   SKIP_SYSTEM_PRECHECKS   1 → skip everything (§8.5; use sparingly)
#   PRECHECK_ON_WARN        abort|continue (default abort, §8.4)
#   PRECHECK_ON_FAIL        abort|continue (default abort, §8.4)
#   PRECHECK_STORAGE_SCOPE  reference scope of the weights mount
#   STORAGE_BENCH_FILE      large file on the weights mount to read (storage check
#                           skipped if unset)
#   NVSHMEM_REQUIRED        1 → missing NVSHMEM SDK is a failure (§8.1)
#
# Exit codes: 0 proceed | 3 warn+abort | 4 fail+abort | 1 internal error.

set -eo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rank0() { [ "${SLURM_PROCID:-0}" = "0" ]; }

if [ "${SKIP_SYSTEM_PRECHECKS:-0}" = "1" ]; then
    rank0 && echo "[prechecks] SKIPPED via skip_system_prechecks (§8.5) — use sparingly"
    exit 0
fi

: "${PRECHECK_CLUSTER:?must be set}"
: "${PRECHECK_SCOPE:?must be set}"
: "${PRECHECK_OUT:?must be set}"
: "${NCCL_TESTS_VERSION:?must be set}"
: "${NCCL_TESTS_CACHE:?must be set}"
: "${MSG_END:=128M}"

mkdir -p "${PRECHECK_OUT}"

# ------------------------------------------------------- collective binaries
# Prefer prebuilt nccl-tests baked into the image (the Alps net image ships them in
# /usr/local/bin); only build — and risk smoke-test mode on a cold cache — when no
# prebuilt set exists (e.g. a stock image). A build cache miss flips the run into
# smoke-test mode (§8.2): the pipeline runs end-to-end but results are NOT persisted.
# shellcheck source=_stack-fingerprint.sh
source "${THIS_DIR}/_stack-fingerprint.sh"
SMOKE_FLAG=""
if nccl_tests_prebuilt; then
    rank0 && echo "[prechecks] using prebuilt nccl-tests from the image: $(nccl_tests_bindir)"
else
    if [ ! -f "$(cache_dir_for_stack)/build/.built" ]; then
        rank0 && echo "[prechecks] collective-tests cache MISS — entering smoke-test mode (§8.2)"
        SMOKE_FLAG="--smoke"
    fi
    bash "${THIS_DIR}/build-nccl-tests.sh"
fi

# ------------------------------------------------------------- collectives
# run-collectives.sh prints one nccl-tests table per collective; split the capture
# per collective so grade.py maps files onto reference rows. The set is the
# deployment's comm pattern (PRECHECK_COLLECTIVES: TP all-reduce / SP weight-gather /
# MoE alltoall, + sendrecv for the PP link). rank 0 grades the captured tables, so a
# per-rank error must NOT abort the run — the trailing `true` keeps the other PEs
# alive (this step runs one rank per GPU; §8.2).
for c in ${PRECHECK_COLLECTIVES:-all_reduce all_gather alltoall}; do
    # Every PE runs the binary (one MPI program across the step's ranks), but only rank 0
    # captures the table to the shared file — otherwise N ranks clobber the same path.
    if rank0; then cap="${PRECHECK_OUT}/collective_${c}.out"; else cap=/dev/null; fi
    COLLECTIVES="$c" bash "${THIS_DIR}/run-collectives.sh" > "${cap}" 2>&1 \
        || { rank0 && { echo "[prechecks] ${c} errored (graded as fail):"; tail -5 "${PRECHECK_OUT}/collective_${c}.out"; }; true; }
done

# ------------------------------------------------------------- NVSHMEM / SHMEM
# alltoall_latency is the MoE-relevant all-to-all collective; it wires up across all
# PEs of this dedicated step (SLURM PMIx). shmem_put_bw needs EXACTLY 2 PEs (pt-to-pt),
# which an N-rank step (N≥4) can't provide — deferred to its own 2-rank step (TODOs.md, §8.1).
# Same rank-0-only capture as the collectives: all PEs run the perftest, rank 0 records it.
if rank0; then nvcap="${PRECHECK_OUT}/nvshmem_alltoall_latency.out"; else nvcap=/dev/null; fi
NVSHMEM_TESTS="device/coll/alltoall_latency" \
    bash "${THIS_DIR}/run-nvshmem.sh" > "${nvcap}" 2>&1 || true
# run-nvshmem.sh exits 0 with a warning when the SDK is absent (skipped row,
# §8.1) unless NVSHMEM_REQUIRED=1, in which case the `|| true` above must not
# mask it — re-check explicitly:
if [ "${NVSHMEM_REQUIRED:-0}" = "1" ]; then
    grep -q "NVSHMEM perftest not found" "${PRECHECK_OUT}/nvshmem_alltoall_latency.out" && {
        rank0 && echo "[prechecks] NVSHMEM required but absent — fail (§8.1)" >&2
        exit 4
    }
fi

# ------------------------------------------------------------- storage read (rank 0)
# run-storage.sh reads the deployed model's real shards off their actual mount: a single-stream
# O_DIRECT floor + a bounded parallel aggregate (vLLM-load-comparable). It resolves the mount →
# reference scope (capstor/iopsstor/fallback) and writes it to storage_scope.txt (§8.1).
if rank0; then
    bash "${THIS_DIR}/run-storage.sh" || true
fi

# ------------------------------------------------------------- grade (rank 0)
if rank0; then
    # Prefer the mount-derived scope from run-storage.sh; fall back to the planner default.
    STORAGE_SCOPE="$(cat "${PRECHECK_OUT}/storage_scope.txt" 2>/dev/null || true)"
    [ -n "${STORAGE_SCOPE}" ] || STORAGE_SCOPE="${PRECHECK_STORAGE_SCOPE:-}"
    python3 "${THIS_DIR}/grade.py" \
        --out-dir "${PRECHECK_OUT}" \
        --cluster "${PRECHECK_CLUSTER}" \
        --scope "${PRECHECK_SCOPE}" \
        --storage-scope "${STORAGE_SCOPE}" \
        --on-warn "${PRECHECK_ON_WARN:-abort}" \
        --on-fail "${PRECHECK_ON_FAIL:-abort}" \
        ${SMOKE_FLAG} \
        --results "${PRECHECK_OUT}/results.json"
fi
