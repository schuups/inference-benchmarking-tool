#!/bin/bash
#
# Storage pre-check (SPECIFICATIONS.md §8.1) — rank 0 only.
#
# Reads the DEPLOYED model's real weight shards off the mount the engine loads from, so the
# numbers are directly comparable to vLLM's actual weight load (model_load_weights_s / _gib):
#
#   1. Sequential read  — single-stream O_DIRECT read of the largest shard. The per-stream
#                         floor of the mount; the stable gate metric.
#   2. Parallel  read   — up to N concurrent O_DIRECT streams across distinct shards, bounded
#                         in total bytes + wall-clock. Mimics vLLM's parallel shard load and is
#                         comparable to its effective weight-load bandwidth.
#
# The mount is resolved from the actual shard path (§8.1: "the benchmark depends entirely on
# where the weights are stored"): /capstor -> capstor scope, /iopsstor -> iopsstor scope, else
# the planner-provided fallback. The resolved scope is written to storage_scope.txt so the
# grader compares against the right reference row.
#
# Required env:
#   PRECHECK_OUT            writable dir for outputs
# Optional env:
#   HF_HOME                 weights cache root (shards discovered under it)
#   MODEL_ID                deployed model id (e.g. org/name) → targets models--<slug>
#   PRECHECK_STORAGE_SCOPE  fallback scope when the mount isn't capstor/iopsstor (e.g. Ceph)
#   STORAGE_BENCH_FILE      explicit single-stream file (overrides discovery)
#   STORAGE_SINGLE_GIB      single-stream read size in GiB (default 4)
#   STORAGE_PARALLEL_STREAMS  max concurrent streams (default 8)
#   STORAGE_PARALLEL_CAP_GIB  total bytes cap across streams, GiB (default 16)
#   STORAGE_PARALLEL_TIMEOUT  per-stream timeout seconds (default 90)

set -eo pipefail

: "${PRECHECK_OUT:?must be set}"
mkdir -p "${PRECHECK_OUT}"

# ----------------------------------------------------------- discover shards
# Prefer the deployed model's snapshot dir; fall back to the largest *.safetensors under HF_HOME.
# `find -L` / `ls -LS` follow the HF-cache snapshot symlinks into blobs/<sha> (the real files).
SHARD_ROOT="${HF_HOME:-}"
if [ -n "${MODEL_ID:-}" ] && [ -n "${HF_HOME:-}" ]; then
    cand="${HF_HOME}/hub/models--${MODEL_ID//\//--}"
    [ -d "${cand}" ] && SHARD_ROOT="${cand}"
fi

mapfile -t SHARDS < <(find -L "${SHARD_ROOT}" -type f -name '*.safetensors' 2>/dev/null \
    | xargs -r ls -LS 2>/dev/null)   # largest first

if [ "${#SHARDS[@]}" -eq 0 ]; then
    echo "[storage] no *.safetensors found under ${SHARD_ROOT:-<unset>} — skipping storage read (§8.1)"
    exit 0
fi

# ----------------------------------------------------------- resolve mount -> scope
# Scope strings MUST match tools/system_prechecks_reference.yaml exactly.
real0="$(readlink -f "${SHARDS[0]}" 2>/dev/null || echo "${SHARDS[0]}")"
case "${real0}" in
    /capstor/*)  STORAGE_SCOPE="capstor weights mount (Lustre, HDD)" ;;
    /iopsstor/*) STORAGE_SCOPE="iopsstor weights mount (Lustre, flash)" ;;
    *)           STORAGE_SCOPE="${PRECHECK_STORAGE_SCOPE:-}" ;;
esac
printf '%s' "${STORAGE_SCOPE}" > "${PRECHECK_OUT}/storage_scope.txt"
echo "[storage] weights on ${real0%%/*}/ → scope '${STORAGE_SCOPE}' (${#SHARDS[@]} shards under ${SHARD_ROOT})"

# ----------------------------------------------------------- 1) sequential (single stream)
single="${STORAGE_BENCH_FILE:-${SHARDS[0]}}"
single_mib=$(( ${STORAGE_SINGLE_GIB:-4} * 1024 ))
if [ -r "${single}" ]; then
    echo "[storage] sequential: $(basename "${single}") ($((single_mib)) MiB, O_DIRECT)"
    dd if="${single}" of=/dev/null bs=1M count="${single_mib}" iflag=direct \
        > "${PRECHECK_OUT}/storage_read.out" 2>&1 || true
fi

# ----------------------------------------------------------- 2) parallel (aggregate)
P="${STORAGE_PARALLEL_STREAMS:-8}"
[ "${#SHARDS[@]}" -lt "${P}" ] && P="${#SHARDS[@]}"
per_mib=$(( ${STORAGE_PARALLEL_CAP_GIB:-16} * 1024 / P ))   # MiB per stream (total ≤ cap)
[ "${per_mib}" -lt 1 ] && per_mib=1

errd="$(mktemp -d "${PRECHECK_OUT}/.par.XXXXXX")"
trap 'rm -rf "${errd}"' EXIT
echo "[storage] parallel: ${P} streams × ${per_mib} MiB, O_DIRECT"
t0="$(date +%s.%N)"
for (( i=0; i<P; i++ )); do
    timeout "${STORAGE_PARALLEL_TIMEOUT:-90}" \
        dd if="${SHARDS[$i]}" of=/dev/null bs=1M count="${per_mib}" iflag=direct 2>"${errd}/${i}.err" || true &
done
wait
t1="$(date +%s.%N)"

# Sum the ACTUAL bytes each dd reported (robust to EOF / timeout), compute aggregate GB/s.
bytes="$(awk '{for(i=1;i<=NF;i++) if($i=="bytes") s+=$(i-1)} END{printf "%d", s+0}' "${errd}"/*.err 2>/dev/null)"
{
    echo "PARALLEL_READ streams=${P} bytes=${bytes} t0=${t0} t1=${t1}"
    awk -v b="${bytes}" -v a="${t0}" -v z="${t1}" \
        'BEGIN{d=z-a; if(d<=0) d=1e-6; printf "seconds=%.4f gbps=%.4f\n", d, b/d/1e9}'
    echo "--- per-stream dd ---"
    cat "${errd}"/*.err 2>/dev/null
} > "${PRECHECK_OUT}/storage_parallel.out"

rank0_gbps="$(awk -F'gbps=' '/gbps=/{print $2}' "${PRECHECK_OUT}/storage_parallel.out" | head -1)"
echo "[storage] parallel aggregate: ${rank0_gbps:-?} GB/s across ${P} streams"

# ----------------------------------------------------------- 3) buffered (readahead)
# Same streams, WITHOUT O_DIRECT: the kernel/Lustre readahead prefetches ahead and pipelines,
# so this tracks vLLM's actual (buffered/mmap) weight load far better than the O_DIRECT floor —
# compare against instances.model_load_weights_gib / model_load_weights_s (§10.2). INFORMATIONAL,
# not a gate: it is state-dependent (a warm node — page cache from a prior job — inflates it).
# CAVEAT: it reads ~the cap into this node's page cache, which can speed up the engine's
# subsequent weight load (and so understate model_load_weights_s, esp. for small models). Set
# STORAGE_BUFFERED_READ=0 when a clean cold model-load measurement matters.
if [ "${STORAGE_BUFFERED_READ:-1}" = "1" ]; then
    bufd="$(mktemp -d "${PRECHECK_OUT}/.buf.XXXXXX")"
    echo "[storage] buffered: ${P} streams × ${per_mib} MiB, readahead (no O_DIRECT)"
    b0="$(date +%s.%N)"
    for (( i=0; i<P; i++ )); do
        timeout "${STORAGE_PARALLEL_TIMEOUT:-90}" \
            dd if="${SHARDS[$i]}" of=/dev/null bs=1M count="${per_mib}" 2>"${bufd}/${i}.err" || true &
    done
    wait
    b1="$(date +%s.%N)"
    bbytes="$(awk '{for(i=1;i<=NF;i++) if($i=="bytes") s+=$(i-1)} END{printf "%d", s+0}' "${bufd}"/*.err 2>/dev/null)"
    {
        echo "BUFFERED_READ streams=${P} bytes=${bbytes} t0=${b0} t1=${b1}"
        awk -v b="${bbytes}" -v a="${b0}" -v z="${b1}" \
            'BEGIN{d=z-a; if(d<=0) d=1e-6; printf "seconds=%.4f gbps=%.4f\n", d, b/d/1e9}'
        echo "--- per-stream dd ---"
        cat "${bufd}"/*.err 2>/dev/null
    } > "${PRECHECK_OUT}/storage_buffered.out"
    rm -rf "${bufd}"
    buf_gbps="$(awk -F'gbps=' '/gbps=/{print $2}' "${PRECHECK_OUT}/storage_buffered.out" | head -1)"
    echo "[storage] buffered aggregate: ${buf_gbps:-?} GB/s across ${P} streams (readahead; state-dependent)"
fi
