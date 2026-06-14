#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

printf 'Pacakages cleanup...\n'
printf 'Marking packages to hold\n'
apt-mark hold libibverbs-dev
printf 'Removing build packages...\n'
apt-get remove --purge -y  \
    pkg-config automake autoconf libtool cmake \
    libconfig-dev libuv1-dev libfuse-dev libfuse3-dev libyaml-dev libnuma-dev libsensors-dev libcurl4-openssl-dev \
    fakeroot dh-make
printf 'Running autoremove...\n'
apt-get autoremove -y
printf 'unhold packages\n'
apt-mark unhold libibverbs-dev
