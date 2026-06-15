#!/usr/bin/env bash
# Common helpers sourced by every phase script.
# Keep changes here rare — editing this file invalidates the cache for every
# downstream phase layer.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# Neutralise the base image's bundled HPC-X / MPI environment for every build
# phase. The base may set LD_LIBRARY_PATH, OPAL_PREFIX, HPCX_* and MPI_*
# pointing at an HPC-X stack (/usr/local/mpi, /opt/hpcx) that the
# neutralize_base_mpi phase removes; leaving them set drags the old
# libpmix/libopen-pal into later link steps (e.g. OpenMPI 5). Clearing them
# here — sourced by every phase — reproduces the monolithic build's in-process
# `unset` in each layer, WITHOUT writing them into the final image's ENV, which
# must keep the base's ROCm/torch LD_LIBRARY_PATH for the vLLM runtime.
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

# AMD/ROCm analog of the NVIDIA netstack's detect_cuda_dir. The vllm/vllm-openai-rocm
# base ships ROCm under /opt/rocm (symlink) -> /opt/rocm-<ver>; honour ROCM_PATH /
# HIP_PATH if the base sets them, else fall back to the versioned dir.
detect_rocm_dir() {
    if [[ -n "${ROCM_PATH:-}" && -d "${ROCM_PATH}" ]]; then
        echo "${ROCM_PATH}"; return 0
    fi
    if [[ -n "${HIP_PATH:-}" && -d "${HIP_PATH}" ]]; then
        echo "${HIP_PATH}"; return 0
    fi
    if [[ -d /opt/rocm ]]; then
        echo /opt/rocm; return 0
    fi
    local d
    d="$(ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1 || true)"
    [[ -n "$d" ]] && { echo "$d"; return 0; }
    return 1
}

setup_env() {
    ROCM_DIR="$(detect_rocm_dir)" || die "Could not determine ROCm directory (no /opt/rocm[-*])."
    export ROCM_DIR
    export ROCM_PATH="${ROCM_PATH:-$ROCM_DIR}"
    export HIP_PATH="${HIP_PATH:-$ROCM_DIR}"
    # gfx942 = MI300A/MI300X. Restrict HIP codegen to the target so phases that
    # compile device code (rccl-tests, rocSHMEM) don't fan out over every arch.
    export GPU_TARGETS="${GPU_TARGETS:-gfx942}"
    export AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx942}"
    case ":${PATH}:" in *":${ROCM_DIR}/bin:"*) : ;; *) export PATH="${ROCM_DIR}/bin:${PATH}" ;; esac
}

# Every phase script begins with: source "$(dirname "$0")/_helpers.sh"; setup_env
# (setup_env is cheap and deterministic.)
