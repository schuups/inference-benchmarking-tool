#!/bin/bash
#
# Build nccl-tests inside the engine container, caching binaries on persistent
# storage. Two things make this distinct from a "vanilla" build:
#
# 1. Stack-aware cache key. The cache dir is keyed by both the nccl-tests
#    version AND a fingerprint of the CUDA / NCCL / OpenMPI / libfabric
#    versions detected in the engine container. Any change to the engine
#    image lands in a new cache dir; older caches survive (revertable).
#
# 2. Install-on-missing. If make / g++ / mpicxx / curl / tar are missing,
#    we install them via apt-get / dnf / yum inside the engine container
#    rather than aborting. Only the rank holding the build lock performs
#    the install, so apt is not hit by N concurrent ranks.
#
# Safe to invoke from every rank under srun --mpi=pmix: a SLURM-rank check
# gates the install + build to rank 0 (globally); the remaining ranks wait
# on the sentinel.
#
# Required env:
#   NCCL_TESTS_VERSION   git tag of github.com/NVIDIA/nccl-tests (e.g. "2.17.1")
#   NCCL_TESTS_CACHE     persistent dir for sources + built binaries
#                        (e.g. /capstor/scratch/cscs/$USER/nccl-tests-cache)

set -eo pipefail

: "${NCCL_TESTS_VERSION:?must be set}"
: "${NCCL_TESTS_CACHE:?must be set}"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_stack-fingerprint.sh
source "${THIS_DIR}/_stack-fingerprint.sh"

CACHE_DIR="$(cache_dir_for_stack)"
SENTINEL="${CACHE_DIR}/build/.built"

rank0() { [ "${SLURM_PROCID:-0}" = "0" ]; }

# ---------------------------------------------------------------- cache hit
if [ -f "${SENTINEL}" ]; then
    rank0 && echo "[build] cache hit: ${CACHE_DIR}/build/"
    exit 0
fi

# ---------------------------------------------------------------- non-rank-0 wait
# Only the globally-first rank does the install + build. The rest poll the
# sentinel (which is on the shared cache filesystem, visible across nodes).
if ! rank0; then
    timeout=1200
    while [ ! -f "${SENTINEL}" ] && [ "${timeout}" -gt 0 ]; do
        sleep 2
        timeout=$((timeout - 2))
    done
    if [ ! -f "${SENTINEL}" ]; then
        echo "[build] rank ${SLURM_PROCID}: timed out waiting for rank-0 build at ${CACHE_DIR}" >&2
        exit 1
    fi
    exit 0
fi

# ---------------------------------------------------------------- rank-0: install
echo "[build] cache miss; building nccl-tests v${NCCL_TESTS_VERSION} (stack fp $(stack_fingerprint))"
echo "[build]   target: ${CACHE_DIR}/build/"

need_install=0
for tool in make g++ curl tar; do
    command -v "$tool" >/dev/null 2>&1 || need_install=1
done
command -v mpicxx >/dev/null 2>&1 || command -v mpic++ >/dev/null 2>&1 || need_install=1

if [ "${need_install}" = "1" ]; then
    echo "[build] one or more build tools missing — installing in the engine container"
    SUDO=""
    if [ "$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi

    if command -v apt-get >/dev/null 2>&1; then
        ${SUDO} apt-get update -qq || true
        ${SUDO} DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            ca-certificates curl tar make g++ libopenmpi-dev openmpi-bin \
            || echo "[build] WARN: apt-get install returned non-zero; continuing with whatever is available"
    elif command -v dnf >/dev/null 2>&1; then
        ${SUDO} dnf install -y -q ca-certificates curl tar make gcc-c++ openmpi openmpi-devel \
            || echo "[build] WARN: dnf install returned non-zero; continuing"
    elif command -v yum >/dev/null 2>&1; then
        ${SUDO} yum install -y -q ca-certificates curl tar make gcc-c++ openmpi openmpi-devel \
            || echo "[build] WARN: yum install returned non-zero; continuing"
    else
        echo "[build] WARN: no apt/dnf/yum found in engine image; relying on whatever is already installed"
    fi
fi

# Re-export common OpenMPI paths so mpicxx resolves on RHEL-family images that
# only put it under /usr/lib64/openmpi/bin.
for d in /usr/lib64/openmpi/bin /usr/lib/openmpi/bin; do
    [ -d "$d" ] && PATH="$d:${PATH}"
done
export PATH

# ---------------------------------------------------------------- rank-0: build
mkdir -p "${NCCL_TESTS_CACHE}"
SRC_DIR="$(mktemp -d "${NCCL_TESTS_CACHE}/.tmp.v${NCCL_TESTS_VERSION}.XXXXXX")"
trap 'rm -rf "${SRC_DIR}"' EXIT

curl -fsSL "https://github.com/NVIDIA/nccl-tests/archive/refs/tags/v${NCCL_TESTS_VERSION}.tar.gz" \
    | tar xz -C "${SRC_DIR}" --strip-components=1
(cd "${SRC_DIR}" && MPI=1 make -j"$(nproc)")

# Atomic publish: move into place, then write the sentinel last.
mkdir -p "${CACHE_DIR}"
rm -rf "${CACHE_DIR}/build.partial"
mv "${SRC_DIR}/build" "${CACHE_DIR}/build.partial"
rm -rf "${CACHE_DIR}/build"
mv "${CACHE_DIR}/build.partial" "${CACHE_DIR}/build"
touch "${SENTINEL}"

echo "[build] published at ${CACHE_DIR}/build/"
