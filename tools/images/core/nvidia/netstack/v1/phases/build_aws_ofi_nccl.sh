#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

git clone https://github.com/aws/aws-ofi-nccl.git /tmp/aws-ofi-nccl
pushd /tmp/aws-ofi-nccl
git reset --hard "${AWS_OFI_NCCL_COMMIT}"
apply_patch_if_set "${AWS_OFI_NCCL_PATCH}"

unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export CPPFLAGS="${CPPFLAGS:-}"
export CFLAGS="${CFLAGS:-}"
export CXXFLAGS="${CXXFLAGS:-}"
CPPFLAGS="$(echo "$CPPFLAGS" | sed 's| -isystem /usr/include||g')"
CFLAGS="$(echo "$CFLAGS" | sed 's| -isystem /usr/include||g')"
CXXFLAGS="$(echo "$CXXFLAGS" | sed 's| -isystem /usr/include||g')"
export CPPFLAGS CFLAGS CXXFLAGS

./autogen.sh

./configure \
    --prefix=/usr \
    --with-libfabric=/usr \
    --with-cuda="${CUDA_DIR}" \
    --with-mpi=/opt/hpcx/ompi \
    --with-hwloc=/opt/hpcx/ompi

# critical fix: remove /usr/include being injected as -isystem
find . \( \
    -name 'Makefile' -o -name 'Makefile.in' -o -name 'Makefile.am' -o -name '*.mk' -o -name 'config.status' -o -name 'libtool' \
\) -type f -print0 \
| xargs -0 -r sed -i 's| -isystem /usr/include||g'

make -j"$(nproc)"
make install

mkdir -p /opt/alps/env
printf 'export AWS_OFI_NCCL_VERSION=%q\n' "$(./m4/get_version.sh)" >> /opt/alps/env/alps-versions.env
printf 'export AWS_OFI_NCCL_COMMIT=%q\n' "${AWS_OFI_NCCL_COMMIT}" >> /opt/alps/env/alps-versions.env

popd
rm -rf /tmp/aws-ofi-nccl
ldconfig
