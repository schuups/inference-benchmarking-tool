#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

hpcx=/opt/hpcx
rm -rf "${hpcx}/ompi"
curl -fsSL "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OMPI_VER}.tar.gz" -o /tmp/ompi.tar.gz
tar -C /tmp -xzf /tmp/ompi.tar.gz
pushd "/tmp/openmpi-${OMPI_VER}"
./configure \
    --prefix="${hpcx}/ompi" \
    --with-ofi=/usr \
    --with-ucx="${hpcx}/ucx" \
    --with-ucc="${hpcx}/ucc" \
    --with-pmix=internal \
    --with-hwloc=internal \
    --with-libevent=internal \
    --enable-oshmem \
    --with-cuda="${CUDA_DIR}" \
    --with-cuda-libdir="${CUDA_DIR}/lib64/stubs"
make -j"$(nproc)"
make install
mkdir -p /opt/alps/env
printf 'export OMPI_VERSION=%q\n' "${OMPI_VER}" >> /opt/alps/env/alps-versions.env
popd
rm -rf "/tmp/openmpi-${OMPI_VER}" /tmp/ompi.tar.gz
ldconfig
