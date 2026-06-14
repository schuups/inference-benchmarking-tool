#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

hpcx=/opt/hpcx
rm -rf "${hpcx}/ucc"
git clone --depth 1 --branch "v${UCC_VERSION}" https://github.com/openucx/ucc.git /tmp/ucc
pushd /tmp/ucc
./autogen.sh

gencode_sm90='-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90'
gencode_sm90a='-gencode arch=compute_90a,code=sm_90a -gencode arch=compute_90a,code=compute_90a'
gencode="${gencode_sm90} ${gencode_sm90a}"

./configure \
    --prefix="${hpcx}/ucc" \
    --with-ucx="${hpcx}/ucx" \
    --with-cuda="${CUDA_DIR}" \
    --with-nvcc-gencode="${gencode}" \
    --with-nccl
make -j"$(nproc)"
make install
mkdir -p /opt/alps/env
printf 'export UCC_VERSION=%q\n' "${UCC_VERSION}" >> /opt/alps/env/alps-versions.env
popd
rm -rf /tmp/ucc
