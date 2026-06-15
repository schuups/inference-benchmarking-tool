#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

curl -fsSL "http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz" -o /tmp/osu.tar.gz
tar --no-same-owner --no-same-permissions -C /tmp -xzf /tmp/osu.tar.gz
pushd "/tmp/osu-micro-benchmarks-${OSU_VERSION}"

# AMD: OSU's ROCm device-buffer path (--enable-rocm) replaces NVIDIA --enable-cuda.
# No driver/NVML stubs needed; HIP runtime is in ${ROCM_DIR}.
CC=/opt/hpcx/ompi/bin/mpicc \
CXX=/opt/hpcx/ompi/bin/mpicxx \
./configure \
    --prefix=/usr/local \
    --enable-rocm \
    --with-rocm="${ROCM_DIR}"
make -j"$(nproc)"
make install
popd
rm -rf "/tmp/osu-micro-benchmarks-${OSU_VERSION}" /tmp/osu.tar.gz
ldconfig
