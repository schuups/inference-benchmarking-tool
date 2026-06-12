#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

git clone https://github.com/ofiwg/libfabric.git /tmp/libfabric
pushd /tmp/libfabric
git reset --hard "${LIBFABRIC_COMMIT}"
apply_patch_if_set "${LIBFABRIC_PATCH}"
./autogen.sh
./configure --prefix=/usr \
    --with-cuda="${CUDA_DIR}" \
    --enable-cuda-dlopen \
    --enable-gdrcopy-dlopen \
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
