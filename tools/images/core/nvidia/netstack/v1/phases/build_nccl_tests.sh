#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

git clone --depth 1 --branch "v${NCCL_TESTS_VER}" https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests
pushd /tmp/nccl-tests
MPI=1 MPI_HOME=/opt/hpcx/ompi CUDA_HOME="${CUDA_DIR}" make -j"$(nproc)"
install -d /usr/local/bin
find build -maxdepth 1 -type f -executable -name '*_perf' -print -exec install -m 0755 {} /usr/local/bin/ \;
popd
rm -rf /tmp/nccl-tests
