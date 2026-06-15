#!/usr/bin/env bash
# rccl-tests: RCCL collective benchmarks (all_reduce_perf, …) — the AMD analog
# of nccl-tests, used by the §8 / post-push sanity gate. Built with MPI for
# multi-node, against the base ROCm's RCCL + HIP.
source "$(dirname "$0")/_helpers.sh"
setup_env

git clone --depth 1 --branch "${RCCL_TESTS_REF}" https://github.com/ROCm/rccl-tests.git /tmp/rccl-tests
pushd /tmp/rccl-tests
# rccl-tests' Makefile reuses the NCCL_HOME/CUSTOM_RCCL_LIB knobs; RCCL + HIP
# live under ${ROCM_DIR}. ROCM_PATH/HIP_PATH (exported by setup_env) drive HIP.
MPI=1 MPI_HOME=/opt/hpcx/ompi NCCL_HOME="${ROCM_DIR}" RCCL_HOME="${ROCM_DIR}" \
  make -j"$(nproc)"
install -d /usr/local/bin
find build -maxdepth 1 -type f -executable -name '*_perf' -print -exec install -m 0755 {} /usr/local/bin/ \;
popd
rm -rf /tmp/rccl-tests
