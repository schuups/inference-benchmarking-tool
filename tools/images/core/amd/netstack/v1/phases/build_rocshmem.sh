#!/usr/bin/env bash
# rocSHMEM: AMD GPU-initiated OpenSHMEM over libfabric/CXI — the analog of
# NVSHMEM (§9.1 SHMEM perftest on AMD targets). HIGHEST-RISK component: rocSHMEM
# is far less mature than NVSHMEM and its libfabric/CXI backend on Slingshot is
# unproven. If this phase blocks, the image ships collectives-first and rocSHMEM
# moves to amd/v2 (set ROCSHMEM_SKIP=1 / drop the Containerfile phase) — it must
# not stall the RCCL/aws-ofi-rccl path.
source "$(dirname "$0")/_helpers.sh"
setup_env

: "${ROCSHMEM_PREFIX:=/opt/rocshmem}"
rm -rf /tmp/rocshmem
git clone --depth 1 --branch "${ROCSHMEM_REF}" https://github.com/ROCm/rocSHMEM.git /tmp/rocshmem
pushd /tmp/rocshmem
mkdir -p build && cd build
# Libfabric/CXI conduit (not IPC/RC), MPI bootstrap via our OpenMPI 5.
CXX="${ROCM_DIR}/bin/hipcc" cmake .. \
    -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${ROCSHMEM_PREFIX}" \
    -DCMAKE_PREFIX_PATH="${ROCM_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    -DGPU_TARGETS="${GPU_TARGETS}" \
    -DUSE_LIBFABRIC=ON \
    -DLIBFABRIC_DIR=/usr \
    -DUSE_MPI=ON \
    -DMPI_HOME=/opt/hpcx/ompi \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=OFF
cmake --build . -j"$(nproc)"
cmake --install .

cat > /etc/ld.so.conf.d/99-rocshmem.conf <<EOF
${ROCSHMEM_PREFIX}/lib
${ROCSHMEM_PREFIX}/lib64
EOF
ldconfig

mkdir -p /opt/alps/env
printf 'export ROCSHMEM_REF=%q\n' "${ROCSHMEM_REF}" >> /opt/alps/env/alps-versions.env
popd
rm -rf /tmp/rocshmem
