#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

hpcx=/opt/hpcx
rm -rf "${hpcx}/ucx"
curl -fsSL "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" -o /tmp/ucx.tar.gz
tar -C /tmp -xzf /tmp/ucx.tar.gz
pushd "/tmp/ucx-${UCX_VERSION}"
mkdir -p build && cd build
../configure \
    --prefix="${hpcx}/ucx" \
    --with-cuda="${CUDA_DIR}" \
    --with-gdrcopy=/usr/local \
    --enable-mt \
    --enable-devel-headers
make -j"$(nproc)"
make install
mkdir -p /opt/alps/env
printf 'export UCX_VERSION=%q\n' "${UCX_VERSION}" >> /opt/alps/env/alps-versions.env
popd
rm -rf "/tmp/ucx-${UCX_VERSION}" /tmp/ucx.tar.gz
