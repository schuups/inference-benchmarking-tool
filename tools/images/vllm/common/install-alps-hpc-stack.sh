#!/usr/bin/env bash
# One-shot runner for the Alps HPC network stack build phases.
#
# Equivalent to the layered Containerfile but in a single shell — for
# interactive or non-Docker use (debugging a phase inside a running container,
# or building outside podman). The container build runs each phase as its own
# cached layer instead.
#
# The build logic lives entirely in common/phases/<phase>.sh (the single source
# of truth, each sourcing _helpers.sh); this script only sequences them, in the
# same order the Containerfile uses. Keep that order in sync with the
# Containerfile's COPY/RUN sequence.
#
# Usage:
#   ./install-alps-hpc-stack.sh                 # run all phases in order
#   ./install-alps-hpc-stack.sh build_nvshmem   # run specific phase(s)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASES_DIR="${PHASES_DIR:-${HERE}/phases}"

ALL_PHASES=(
    apt_install_build_deps
    remove_efa
    neutralize_base_mpi
    remove_hpcx_plugins
    build_xpmem
    build_gdrcopy
    build_cxi_bits
    build_libfabric
    build_nccl_deb
    build_ucx
    build_ucc
    build_ompi5
    build_aws_ofi_nccl
    build_nvshmem
    build_nccl_tests
    build_osu
    clean_up
)

phases=("$@")
[[ ${#phases[@]} -eq 0 ]] && phases=("${ALL_PHASES[@]}")

for phase in "${phases[@]}"; do
    script="${PHASES_DIR}/${phase}.sh"
    [[ -f "$script" ]] || { echo "ERROR: unknown phase '${phase}' (${script} not found)" >&2; exit 1; }
    echo "==> Running phase: ${phase}"
    bash "$script"
done
