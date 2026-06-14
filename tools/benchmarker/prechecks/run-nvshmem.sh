#!/bin/bash
#
# Run NVSHMEM perftest across the PEs of the CURRENT srun step (one PE per task).
#
# NVSHMEM perftest is multi-PROCESS: one PE per process, wired up by the launcher's
# PMIx. Per CSCS guidance (docs.cscs.ch/software/communication/nvshmem) the supported
# launch on Alps is a DEDICATED SLURM step — SLURM (not a nested mpirun) supplies the
# PEs and bootstraps them over PMIx:
#     srun --ntasks-per-node=N --mpi=pmix --environment=<edf> bash run-nvshmem.sh
# Each task execs the binary for its PE; SLURM runs N copies into one N-PE job.
#
# The §8 pre-checks otherwise run INSIDE the engine's single-task srun session, which
# NVSHMEM cannot share (1 task ⇒ 1 PE ⇒ degenerate: a collective reports busbw≡0, a
# pt-to-pt test aborts "requires exactly two processes"). When this script sees a
# single-task step it SKIPS with a clear message instead of emitting a bogus single-PE
# number; a dedicated multi-task NVSHMEM step is the proper home (pending design
# decision — the engine session disables the CXI hook the inter-node libfabric/cxi
# transport needs; see TODOs.md and engine.sbatch.j2).
#
# Required env (set below per CSCS docs):
#   PMIX_MCA_psec=native            native (no-munge) PMIx peer auth on Alps
#   NVSHMEM_DISABLE_CUDA_VMM=1       libfabric transport has no VMM support yet
#   NVSHMEM_REMOTE_TRANSPORT=libfabric / NVSHMEM_LIBFABRIC_PROVIDER=cxi  (inter-node)
# Optional:
#   NVSHMEM_BIN_DIR     explicit path to a perftest dir; takes priority
#   NVSHMEM_TESTS       relative paths from the perftest dir, space-separated
#                       (default: alltoall_latency + put bandwidth)
#   NVSHMEM_REQUIRED    1 → fail if perftest binaries are absent / no PEs to run them

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

# ---------------------------------------------------------------- env (CSCS)
# Native (no-munge) PMIx peer auth for the in-step PEs; libfabric transport has no
# VMM support yet. The inter-node transport is the image's OWN bundled libfabric/cxi
# (the self-contained Alps net image): the host CXI / aws_ofi_nccl hooks stay
# DISABLED — we do not rely on host injection. Intra-node NVSHMEM uses NVLink P2P
# and ignores the libfabric vars.
export PMIX_MCA_psec="${PMIX_MCA_psec:-native}"
export NVSHMEM_DISABLE_CUDA_VMM="${NVSHMEM_DISABLE_CUDA_VMM:-1}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-libfabric}"
export NVSHMEM_LIBFABRIC_PROVIDER="${NVSHMEM_LIBFABRIC_PROVIDER:-cxi}"

# ---------------------------------------------------------------- PE topology
# One PE per task in THIS srun step. NVSHMEM bootstraps over SLURM's PMIx — never a
# nested mpirun (it would re-inherit SLURM's 1-task PMIx server and collapse to
# npes=1, with munge psec auth errors). The §8 engine session is single-task, so
# NVSHMEM cannot run there; a dedicated `srun --ntasks-per-node=N --mpi=pmix` step
# is required. Skip cleanly when the step has fewer than 2 PEs.
pes="${SLURM_STEP_NUM_TASKS:-${SLURM_NTASKS:-1}}"
if [ "${pes}" -lt 2 ]; then
    rank0 && {
        echo "[nvshmem] single-task step (${pes} PE) — NVSHMEM perftest needs a dedicated"
        echo "[nvshmem]   multi-task 'srun --ntasks-per-node=N --mpi=pmix' step (CSCS docs);"
        echo "[nvshmem]   the engine session is single-task. Skipping (§8.1)."
    }
    if [ "${NVSHMEM_REQUIRED:-0}" = "1" ]; then
        rank0 && echo "[nvshmem] NVSHMEM_REQUIRED=1 but step is single-PE — the harness must launch a dedicated NVSHMEM step" >&2
        exit 1
    fi
    exit 0
fi

# ---------------------------------------------------------------- run
# Two complementary measurements (caller sizes the step per test via --ntasks):
#   alltoall_latency — round-trip latency, MoE-relevant tail behaviour (N PEs)
#   shmem_put_bw     — one-sided put bandwidth, MoE dispatch ceiling (exactly 2 PEs)
: "${NVSHMEM_TESTS:=device/coll/alltoall_latency device/pt-to-pt/shmem_put_bw}"
rank0 && echo "[nvshmem] using ${BINDIR} (${pes} PE)"

fail=0
for t in ${NVSHMEM_TESTS}; do
    bin="${BINDIR}/${t}"
    if [ ! -x "${bin}" ]; then
        rank0 && echo "[nvshmem] missing ${bin} — skipping"
        continue
    fi
    rank0 && echo "===== NVSHMEM ${t} (${pes} PE) ====="
    # In-step PE: exec the binary directly; SLURM's --mpi=pmix wires the N tasks
    # into one N-PE job. `timeout` bounds a PMIx-bootstrap hang (the caller has `|| true`).
    timeout "${NVSHMEM_TIMEOUT:-180}" "${bin}" \
        || { rank0 && echo "[nvshmem] ${t} failed (non-zero exit)" >&2; fail=1; }
done

exit "${fail}"
