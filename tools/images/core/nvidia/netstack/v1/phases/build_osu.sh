#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

curl -fsSL "http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz" -o /tmp/osu.tar.gz
tar --no-same-owner --no-same-permissions -C /tmp -xzf /tmp/osu.tar.gz
pushd "/tmp/osu-micro-benchmarks-${OSU_VERSION}"

# osu's configure link test and CUDA benchmarks link -lcuda -lnvidia-ml.
# _helpers.sh clears the base LIBRARY_PATH; restore the CUDA stubs dir so the
# driver/NVML stubs resolve at link time (no GPU/driver in the build container).
# Mirrors the upstream image's ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs.
export LIBRARY_PATH="${CUDA_DIR}/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_DIR}/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

CC=/opt/hpcx/ompi/bin/mpicc \
CXX=/opt/hpcx/ompi/bin/mpicxx \
CFLAGS="-O3 -lcuda -lnvidia-ml" \
./configure \
    --prefix=/usr/local \
    --enable-cuda \
    --with-cuda-include="${CUDA_DIR}/include" \
    --with-cuda-libpath="${CUDA_DIR}/lib64"
make -j"$(nproc)"
make install
popd
rm -rf "/tmp/osu-micro-benchmarks-${OSU_VERSION}" /tmp/osu.tar.gz
ldconfig
