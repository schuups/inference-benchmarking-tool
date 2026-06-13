#!/usr/bin/env bash
# Common helpers sourced by every phase script.
# Keep changes here rare — editing this file invalidates the cache for every
# downstream phase layer.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# Neutralise the base image's bundled HPC-X / MPI environment for every build
# phase. The NGC base sets LD_LIBRARY_PATH, OPAL_PREFIX, HPCX_* and MPI_*
# pointing at the HPC-X stack (/usr/local/mpi, /opt/hpcx) that the
# neutralize_base_mpi phase removes; leaving them set drags the old
# libpmix/libopen-pal into later link steps (e.g. OpenMPI 5). Clearing them
# here — sourced by every phase — reproduces the monolithic build's in-process
# `unset` in each layer, WITHOUT writing them into the final image's ENV, which
# must keep the base's CUDA/torch LD_LIBRARY_PATH for the vLLM runtime.
unset OPAL_PREFIX PMIX_INSTALL_PREFIX PMIX_HOME MPI_HOME MPI_ROOT \
      HPCX_DIR HPCX_HOME HPCX_MPI_DIR HPCX_UCX_DIR HPCX_UCC_DIR \
      HPCX_HCOLL_DIR HPCX_SHARP_DIR HPCX_NVSHMEM_DIR \
      OMPI_MCA_prefix OMPI_HOME OPAL_LIBDIR \
      LD_LIBRARY_PATH LIBRARY_PATH 2>/dev/null || true

die() {
    echo "ERROR: $*" >&2
    exit 1
}

proxied_pip_install() {
    python -m pip install -i https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple "$@"
}

apply_patch_if_set() {
    local patch_rel="${1:-}"
    [[ -z "$patch_rel" ]] && return 0

    local patch="/opt/alps/patches/${patch_rel}"
    [[ -f "$patch" ]] || die "Patch not found: ${patch}"

    git apply --check --whitespace=nowarn "$patch"
    git apply --whitespace=nowarn "$patch"
}

detect_cuda_dir() {
    if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
        echo "${CUDA_HOME}"; return 0
    fi
    if [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}" ]]; then
        echo "${CUDA_PATH}"; return 0
    fi
    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path
        nvcc_path="$(command -v nvcc)"
        echo "$(cd "$(dirname "$nvcc_path")/.." && pwd)"
        return 0
    fi
    if [[ -d /usr/local/cuda ]]; then
        echo /usr/local/cuda; return 0
    fi
    return 1
}

setup_env() {
    CUDA_DIR="$(detect_cuda_dir)" || die "Could not determine CUDA directory..."
    export CUDA_DIR
    export CUDA_HOME="${CUDA_HOME:-$CUDA_DIR}"
    export CUDA_PATH="${CUDA_PATH:-$CUDA_DIR}"
}

# Every phase script begins with: source "$(dirname "$0")/_helpers.sh"; setup_env
# (setup_env is cheap and deterministic.)
