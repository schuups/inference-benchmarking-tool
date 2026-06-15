#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

# AMD variant: libfabric's GPU HMEM path uses ROCR (AMD ROCr runtime) instead of
# CUDA. Build the CXI provider with ROCR HMEM (dlopen'd at runtime) + XPMEM, and
# DROP the NVIDIA cuda/gdrcopy/nvml bits entirely. ROCm ships the ROCr headers
# (hsa/) and libhsa-runtime under ${ROCM_DIR}; libfabric autodetects them from
# --with-rocr.
git clone https://github.com/ofiwg/libfabric.git /tmp/libfabric
pushd /tmp/libfabric
git reset --hard "${LIBFABRIC_COMMIT}"
apply_patch_if_set "${LIBFABRIC_PATCH}"
./autogen.sh
./configure --prefix=/usr \
    --with-rocr="${ROCM_DIR}" \
    --enable-rocr-dlopen \
    --enable-xpmem=/usr \
    --enable-cxi
make -j"$(nproc)"
make install
mkdir -p /opt/alps/env
printf 'export LIBFABRIC_VERSION=%q\n' "$(fi_info --version | head -n 1 | awk '{ print $2; }')" >> /opt/alps/env/alps-versions.env
printf 'export LIBFABRIC_COMMIT=%q\n' "${LIBFABRIC_COMMIT}" >> /opt/alps/env/alps-versions.env
popd
rm -rf /tmp/libfabric
ldconfig
