#!/bin/bash
#
# Build nccl-tests inside the engine container, caching the compiled binaries
# on persistent storage so subsequent pre-checks skip the compile step. Idempotent
# — safe to invoke before every engine job.
#
# Required env (the precheck sbatch typically sets these):
#   NCCL_TESTS_VERSION   git tag of github.com/NVIDIA/nccl-tests (e.g. "2.17.1")
#   NCCL_TESTS_CACHE     persistent dir for sources + built binaries
#                        (e.g. /capstor/scratch/cscs/$USER/nccl-tests-cache)
#
# Invoke from a single rank:
#   srun -n 1 --environment=<engine-edf> build-nccl-tests.sh

set -euo pipefail

: "${NCCL_TESTS_VERSION:?must be set (e.g. 2.17.1)}"
: "${NCCL_TESTS_CACHE:?must be set (e.g. /capstor/scratch/cscs/\$USER/nccl-tests-cache)}"

CACHE_DIR="${NCCL_TESTS_CACHE}/v${NCCL_TESTS_VERSION}"
SENTINEL="${CACHE_DIR}/build/.built"

if [ -f "${SENTINEL}" ]; then
    echo "[build] cache hit: ${CACHE_DIR}/build/"
    exit 0
fi

# The engine image must carry mpic++/make/g++; we are deliberately not running
# this in a separate, build-tools-only container (per SPECIFICATIONS.md §8.2,
# pre-checks must run in the exact image the engine runs in).
for tool in mpic++ make g++ curl tar; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[build] ERROR: '${tool}' not in the engine container." >&2
        echo "[build] Bake make, g++, libopenmpi-dev, openmpi-bin (and curl/tar) into the engine image," >&2
        echo "[build] or pre-populate ${CACHE_DIR}/build/ from another job." >&2
        exit 1
    fi
done

echo "[build] cache miss; building nccl-tests v${NCCL_TESTS_VERSION} -> ${CACHE_DIR}/build/"

# Build under a tmpdir so a killed build never gets picked up as cached.
mkdir -p "${NCCL_TESTS_CACHE}"
SRC_DIR="$(mktemp -d "${NCCL_TESTS_CACHE}/.build.v${NCCL_TESTS_VERSION}.XXXXXX")"
trap 'rm -rf "${SRC_DIR}"' EXIT

curl -fsSL "https://github.com/NVIDIA/nccl-tests/archive/refs/tags/v${NCCL_TESTS_VERSION}.tar.gz" \
    | tar xz -C "${SRC_DIR}" --strip-components=1
(cd "${SRC_DIR}" && MPI=1 make -j"$(nproc)")

# Publish atomically by moving the build dir into place, then writing the sentinel.
mkdir -p "${CACHE_DIR}"
rm -rf "${CACHE_DIR}/build.partial"
mv "${SRC_DIR}/build" "${CACHE_DIR}/build.partial"
rm -rf "${CACHE_DIR}/build"
mv "${CACHE_DIR}/build.partial" "${CACHE_DIR}/build"
touch "${SENTINEL}"

echo "[build] cached at ${CACHE_DIR}/build/"
