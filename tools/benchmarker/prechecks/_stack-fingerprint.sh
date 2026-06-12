# Shared helper — sourced by build-nccl-tests.sh and run-collectives.sh.
# Computes a deterministic fingerprint of the build-relevant stack inside the
# engine container, so any change to CUDA / NCCL / OpenMPI / libfabric /
# nccl-tests lands in a separate cache directory and never reuses an
# ABI-incompatible build.
#
# Both scripts must compute the same value, which they will as long as they
# run in the same container session (the whole point of running pre-checks
# inside the engine container).

stack_fingerprint() {
    {
        echo "nccl_tests=${NCCL_TESTS_VERSION:-unset}"

        if command -v nvcc >/dev/null 2>&1; then
            nvcc --version 2>/dev/null | grep -iE 'release' | head -1
        else
            echo "cuda=missing"
        fi

        if command -v mpicxx >/dev/null 2>&1; then
            (mpicxx --showme:version 2>&1 || mpicxx --version 2>&1) | head -1
        elif command -v mpic++ >/dev/null 2>&1; then
            mpic++ --version 2>&1 | head -1
        else
            echo "mpi=missing"
        fi

        # NCCL header version — pick the first nccl.h that exists.
        local h
        for h in /usr/include/nccl.h /usr/local/include/nccl.h /usr/local/cuda/include/nccl.h; do
            if [ -r "$h" ]; then
                grep -E '#define NCCL_(MAJOR|MINOR|PATCH)' "$h"
                break
            fi
        done

        # libfabric — loaded at runtime by the AWS OFI hook, but versioning it
        # here so a libfabric bump in the engine image triggers a fresh cache.
        if command -v fi_info >/dev/null 2>&1; then
            fi_info --version 2>&1 | head -1
        elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libfabric 2>/dev/null; then
            echo "libfabric=$(pkg-config --modversion libfabric)"
        else
            echo "libfabric=unknown"
        fi
    } | sha256sum | cut -c1-16
}

# Echoes the cache directory for the current stack.
cache_dir_for_stack() {
    : "${NCCL_TESTS_VERSION:?must be set}"
    : "${NCCL_TESTS_CACHE:?must be set}"
    echo "${NCCL_TESTS_CACHE}/v${NCCL_TESTS_VERSION}-$(stack_fingerprint)"
}
