#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

curl -fsSL "https://github.com/NVIDIA/nccl/archive/refs/tags/v${NCCL_VER}.tar.gz" -o /tmp/nccl.tar.gz
tar -C /tmp -xzf /tmp/nccl.tar.gz
pushd "/tmp/nccl-${NCCL_VER}"
apply_patch_if_set "${NCCL_PATCH}"
make -j"$(nproc)" pkg.debian.build CUDA_HOME="${CUDA_DIR}"
dpkg -i build/pkg/deb/*.deb
mkdir -p /opt/alps/env
printf 'export NCCL_VERSION=%q\n' "${NCCL_VER}" >> /opt/alps/env/alps-versions.env
# Produces: ext-profiler/inspector/libnccl-profiler-inspector.so
pushd ext-profiler/inspector
make -j"$(nproc)" CUDA_HOME="${CUDA_DIR}"
install -D -m 0644 libnccl-profiler-inspector.so /usr/local/lib/libnccl-profiler-inspector.so
popd
popd
rm -rf "/tmp/nccl-${NCCL_VER}" /tmp/nccl.tar.gz
ldconfig
