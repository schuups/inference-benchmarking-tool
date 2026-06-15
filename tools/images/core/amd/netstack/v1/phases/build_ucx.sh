#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

hpcx=/opt/hpcx
rm -rf "${hpcx}/ucx"
curl -fsSL "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" -o /tmp/ucx.tar.gz
tar -C /tmp -xzf /tmp/ucx.tar.gz
pushd "/tmp/ucx-${UCX_VERSION}"
mkdir -p build && cd build
# AMD: --with-rocm replaces NVIDIA --with-cuda/--with-gdrcopy.
# Bake the final install + ROCm rpaths at link time via LDFLAGS so libtool sees
# the rpath already correct and SKIPS the install-time relink. The ROCm module
# otherwise forces a relink of libuct at `make install`, which hits a libtool bug
# ("mv: cannot overwrite directory 'libuct.so.0.0.0'") — the NVIDIA/cuda build
# never triggered this. --disable-static drops the .a/.la relink dance entirely.
../configure \
    --prefix="${hpcx}/ucx" \
    --with-rocm="${ROCM_DIR}" \
    --enable-mt \
    --enable-devel-headers \
    --disable-static \
    LDFLAGS="-Wl,-rpath,${hpcx}/ucx/lib -Wl,-rpath,${ROCM_DIR}/lib -Wl,-rpath,${ROCM_DIR}/lib64"
make -j"$(nproc)"

# The ROCm build makes libtool relink libucs/libuct against their just-installed
# sibling libs at `make install`, and that relink hits a libtool-on-overlay bug
# ("mv: cannot overwrite directory 'libXXX.so.0.0.0'"). The cuda build never
# relinks these, so it never hit it. Our LDFLAGS already bake the final rpaths
# into the built .so, so the relink is unnecessary — strip libtool's
# relink_command from the generated .la/.lai files so `make install` installs the
# already-built libraries as-is (standard libtool-in-containers workaround).
find . \( -name '*.la' -o -name '*.lai' \) -print0 \
  | xargs -0 -r sed -i 's/^relink_command=.*/relink_command=/'

make install
mkdir -p /opt/alps/env
printf 'export UCX_VERSION=%q\n' "${UCX_VERSION}" >> /opt/alps/env/alps-versions.env
popd
rm -rf "/tmp/ucx-${UCX_VERSION}" /tmp/ucx.tar.gz
