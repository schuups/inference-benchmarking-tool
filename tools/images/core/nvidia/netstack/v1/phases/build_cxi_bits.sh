#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

git clone --depth 1 --branch "${CASSINI_HEADERS_VERSION}" https://github.com/HewlettPackard/shs-cassini-headers.git /tmp/shs-cassini-headers
cp -r /tmp/shs-cassini-headers/include/* /usr/include/
cp -r /tmp/shs-cassini-headers/share/* /usr/share/
rm -rf /tmp/shs-cassini-headers

git clone --depth 1 --branch "${CXI_DRIVER_VERSION}" https://github.com/HewlettPackard/shs-cxi-driver.git /tmp/shs-cxi-driver
cp -r /tmp/shs-cxi-driver/include/* /usr/include/
rm -rf /tmp/shs-cxi-driver

git clone --depth 1 --branch "${LIBCXI_VERSION}" https://github.com/HewlettPackard/shs-libcxi.git /tmp/shs-libcxi
pushd /tmp/shs-libcxi
./autogen.sh
./configure --prefix=/usr --with-cuda="${CUDA_DIR}"
make -j"$(nproc)"
make install
popd
rm -rf /tmp/shs-libcxi
ldconfig
