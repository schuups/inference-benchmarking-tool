#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

ver="${GDRCOPY_VER}"
git clone --depth 1 --branch "v${ver}" https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
pushd /tmp/gdrcopy
make CC=gcc CUDA="${CUDA_DIR}" lib -j"$(nproc)"
make lib_install
popd
rm -rf /tmp/gdrcopy
ldconfig
