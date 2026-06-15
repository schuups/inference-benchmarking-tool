#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

# Use JFrog Artifactory as an APT proxy/cache for Ubuntu packages, to speed
# up installs and reduce load on Ubuntu mirrors. The ROCm base is Ubuntu 22.04
# (classic /etc/apt/sources.list), NOT 24.04's deb822 ubuntu.sources — handle
# whichever the base ships.
for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do
    [[ -f "$f" ]] && sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://jfrog.svc.cscs.ch/artifactory/ubuntu|' \
        -e 's|http://security.ubuntu.com/ubuntu|https://jfrog.svc.cscs.ch/artifactory/ubuntu|' \
        -e 's|http://ports.ubuntu.com/ubuntu-ports|https://jfrog.svc.cscs.ch/artifactory/ubuntu-ports|' \
        "$f"
done
printf '%s\n%s' "Acquire::http::AllowRedirect "true";" "Acquire::http::Pipeline-Depth "0";" \
    > /etc/apt/apt.conf.d/99-jfrog-proxy
apt -o "Acquire::https::Verify-Peer=false" update
apt -o "Acquire::https::Verify-Peer=false" install -y ca-certificates

apt-get update
apt-get install -y --no-install-recommends \
    build-essential ca-certificates pkg-config automake autoconf libtool cmake \
    bc gdb strace wget curl git bzip2 python3 gfortran ninja-build \
    rdma-core numactl \
    libconfig-dev libuv1-dev libfuse-dev libfuse3-dev libyaml-dev libnl-3-dev \
    libnuma-dev libsensors-dev libcurl4-openssl-dev libjson-c-dev \
    libsox-fmt-all \
    devscripts debhelper fakeroot dh-make
rm -rf /var/lib/apt/lists/*
