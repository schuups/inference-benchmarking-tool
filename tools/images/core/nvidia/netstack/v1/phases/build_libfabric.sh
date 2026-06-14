#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

# libfabric's CUDA HMEM path (src/hmem_cuda.c) #includes nvml.h. NGC bases ship
# it, but the slim vllm/vllm-openai CUDA layout omits it. Provide it from the
# base's NVIDIA CUDA apt repo (/etc/apt/sources.list.d/cuda.list, version-matched
# to the toolkit) when missing — a no-op on bases that already have nvml.h.
if [[ ! -f "${CUDA_DIR}/include/nvml.h" ]]; then
    cuda_ver="$(basename "$(readlink -f "${CUDA_DIR}")")"; cuda_ver="${cuda_ver#cuda-}"
    nvml_pkg="cuda-nvml-dev-${cuda_ver/./-}"   # e.g. cuda-13.0 -> cuda-nvml-dev-13-0
    echo "[libfabric] nvml.h missing under ${CUDA_DIR}/include; installing ${nvml_pkg}"
    apt-get update
    apt-get install -y --no-install-recommends "${nvml_pkg}" \
        || apt-get install -y --no-install-recommends cuda-nvml-dev-13-0
    rm -rf /var/lib/apt/lists/*
    # The dev package places nvml.h under targets/<arch>/include; ensure it is
    # also visible at ${CUDA_DIR}/include (where --with-cuda points -I).
    if [[ ! -f "${CUDA_DIR}/include/nvml.h" ]]; then
        nvml_hdr="$(find "$(readlink -f "${CUDA_DIR}")" -name nvml.h 2>/dev/null | head -1)"
        [[ -n "${nvml_hdr}" ]] && ln -sf "${nvml_hdr}" "${CUDA_DIR}/include/nvml.h"
    fi
    [[ -f "${CUDA_DIR}/include/nvml.h" ]] || die "nvml.h still missing after installing ${nvml_pkg}"
fi

git clone https://github.com/ofiwg/libfabric.git /tmp/libfabric
pushd /tmp/libfabric
git reset --hard "${LIBFABRIC_COMMIT}"
apply_patch_if_set "${LIBFABRIC_PATCH}"
./autogen.sh
./configure --prefix=/usr \
    --with-cuda="${CUDA_DIR}" \
    --enable-cuda-dlopen \
    --enable-gdrcopy-dlopen \
    --enable-xpmem=/usr \
    --enable-cxi
make -j"$(nproc)"
make install
mkdir -p /opt/alps/env
printf 'export LIBFABRIC_VERSION=%q\n' "$(fi_info --version | head -n 1 | awk '{ print $2; }')" >> /opt/alps/env/alps-versions.env
printf 'export LIBFABRIC_COMMIT=%q\n' "${LIBFABRIC_COMMIT}" >> /opt/alps/env/alps-versions.env
popd
rm -rf /tmp/libfabric
ldconfig
