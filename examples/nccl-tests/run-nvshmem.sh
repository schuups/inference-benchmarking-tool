#!/bin/bash
#
# Run NVSHMEM perftest at the current rank topology. Executed by every rank
# under `srun --mpi=pmix`; the NVSHMEM perftest binaries use PMIx for wire-up.
#
# NVSHMEM perftest binaries ship with the NVSHMEM SDK that the engine image
# already includes when the engine relies on NVSHMEM for MoE all-to-all
# dispatch (DeepEP, TRT-LLM MoE kernels). This script discovers them in
# standard locations.
#
# If NVSHMEM is not present in the engine image, the row is skipped with a
# warning. Set NVSHMEM_REQUIRED=1 to make absence a failure (use this when
# the engine actually depends on NVSHMEM and a missing SDK is a bug).
#
# Required env (none — script no-ops cleanly if NVSHMEM is absent).
# Optional:
#   NVSHMEM_BIN_DIR     explicit path to a perftest dir; takes priority
#   NVSHMEM_TESTS       relative paths from the perftest dir, space-separated
#                       (default: alltoall_latency + put bandwidth)
#   NVSHMEM_REQUIRED    1 → fail if perftest binaries are not found

set -eo pipefail

rank0() { [ "${SLURM_PROCID:-0}" = "0" ]; }

# ---------------------------------------------------------------- locate perftest
CANDIDATES=(
    "${NVSHMEM_BIN_DIR:-}"
    "/opt/nvshmem/bin/perftest"
    "/usr/local/nvshmem/bin/perftest"
    "/usr/lib/x86_64-linux-gnu/nvshmem/bin/perftest"
    "/usr/local/cuda/nvshmem/bin/perftest"
)

BINDIR=""
for d in "${CANDIDATES[@]}"; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    BINDIR="$d"
    break
done

if [ -z "${BINDIR}" ]; then
    rank0 && {
        echo "[nvshmem] NVSHMEM perftest not found in this engine image"
        echo "[nvshmem]   tried: ${CANDIDATES[*]}"
        echo "[nvshmem]   set NVSHMEM_BIN_DIR to override, or add NVSHMEM to the engine image"
    }
    if [ "${NVSHMEM_REQUIRED:-0}" = "1" ]; then
        rank0 && echo "[nvshmem] NVSHMEM_REQUIRED=1 — failing" >&2
        exit 1
    fi
    exit 0
fi

# ---------------------------------------------------------------- run
# Default to two complementary measurements:
#   alltoall_latency — round-trip latency, MoE-relevant tail behaviour
#   shmem_put_bw     — large-message one-sided put bandwidth, MoE dispatch ceiling
: "${NVSHMEM_TESTS:=device/coll/alltoall_latency device/pt-to-pt/shmem_put_bw}"

rank0 && echo "[nvshmem] using ${BINDIR}"

fail=0
for t in ${NVSHMEM_TESTS}; do
    bin="${BINDIR}/${t}"
    if [ ! -x "${bin}" ]; then
        rank0 && echo "[nvshmem] missing ${bin} — skipping"
        continue
    fi
    rank0 && echo "===== NVSHMEM ${t} ====="
    "${bin}" \
        || { rank0 && echo "[nvshmem] ${t} failed (non-zero exit)" >&2; fail=1; }
done

exit "${fail}"
