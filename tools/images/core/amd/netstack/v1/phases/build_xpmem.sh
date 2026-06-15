#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

ref="${XPMEM_REF}"
git clone https://github.com/hpc/xpmem.git /tmp/xpmem
pushd /tmp/xpmem
git checkout "${ref}"
./autogen.sh
./configure --prefix=/usr --with-default-prefix=/usr --disable-kernel-module
make -j"$(nproc)"
make install
popd
rm -rf /tmp/xpmem
ldconfig
