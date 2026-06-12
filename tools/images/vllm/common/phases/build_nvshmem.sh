#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

: "${NVSHMEM_PREFIX:=/opt/nvshmem}"
: "${NVSHMEM_BUILDDIR:=/tmp/nvshmem-build}"
: "${NVSHMEM_SRC_DIR:=/tmp/nvshmem-src}"
: "${NVSHMEM_CUDA_ARCH:=90}"
: "${NVSHMEM_ENABLE_PYTHON:=1}"
: "${NVSHMEM_ENABLE_TESTS:=1}"

# Remove preinstalled NVSHMEM
apt-get update
apt-get purge -y 'libnvshmem*-cuda-*' 'nvshmem*' || true
apt-get autoremove -y || true

# Remove CUDA symlinks/copies that can shadow our install
rm -f "${CUDA_DIR}/lib64/libnvshmem"*.so* || true
rm -f "${CUDA_DIR}/targets/"*/lib/libnvshmem*.so* || true
rm -rf /usr/lib/*/nvshmem || true

rm -rf "${NVSHMEM_SRC_DIR}" "${NVSHMEM_BUILDDIR}"
mkdir -p "${NVSHMEM_BUILDDIR}"

# CUDA 13 dropped the legacy NVTX C-API headers (`nvtx3/nvToolsExt.h` etc.)
# from the toolkit, but NVSHMEM 3.6.5-0 still includes them unconditionally.
# Fetch the NVTX v3 headers and install into /usr/include/nvtx3 if missing.
if [[ ! -f /usr/include/nvtx3/nvToolsExt.h ]]; then
    git clone --depth 1 --branch release-v3 https://github.com/NVIDIA/NVTX.git /tmp/NVTX
    mkdir -p /usr/include/nvtx3
    cp -r /tmp/NVTX/c/include/nvtx3/. /usr/include/nvtx3/
    rm -rf /tmp/NVTX
fi

git clone --depth 1 --branch "v${NVSHMEM_VER}" https://github.com/NVIDIA/nvshmem.git "${NVSHMEM_SRC_DIR}"

pushd "${NVSHMEM_SRC_DIR}" >/dev/null
apply_patch_if_set "${NVSHMEM_PATCH}"
popd >/dev/null

# CUDA 13 NGC images ship libnvJitLink as a versioned runtime .so without the
# unversioned dev symlink NVSHMEM's test/common find_library(nvJitLink) requires
# (it searches lib64 with NO_DEFAULT_PATH — no fallback). Tests are on, so
# test/common is configured; create the symlink from the runtime lib if missing.
if ! ls "${CUDA_DIR}"/lib64/libnvJitLink.so >/dev/null 2>&1; then
    njl_rt=$(ls -1 "${CUDA_DIR}"/lib64/libnvJitLink.so.* \
                   "${CUDA_DIR}"/targets/*/lib/libnvJitLink.so.* 2>/dev/null \
             | sort -V | tail -1)
    if [[ -n "${njl_rt}" ]]; then
        ln -sf "${njl_rt}" "${CUDA_DIR}/lib64/libnvJitLink.so"
        echo "[nvshmem] linked ${CUDA_DIR}/lib64/libnvJitLink.so -> ${njl_rt}"
    else
        echo "[nvshmem] WARNING: no libnvJitLink runtime lib under ${CUDA_DIR}; test/common will fail" >&2
        ls -la "${CUDA_DIR}"/lib64/libnvJitLink* "${CUDA_DIR}"/targets/*/lib/libnvJitLink* 2>&1 || true
    fi
fi

NVSHMEM_BUILD_EXAMPLES=0 \
NVSHMEM_BUILD_TESTS="$([[ "${NVSHMEM_ENABLE_TESTS}" == "1" ]] && echo 1 || echo 0)" \
NVSHMEM_DEBUG=0 \
NVSHMEM_DEVEL=0 \
NVSHMEM_DEFAULT_PMI2=0 \
NVSHMEM_DEFAULT_PMIX=1 \
NVSHMEM_DISABLE_COLL_POLL=1 \
NVSHMEM_ENABLE_ALL_DEVICE_INLINING=0 \
NVSHMEM_GPU_COLL_USE_LDST=0 \
NVSHMEM_LIBFABRIC_SUPPORT=1 \
NVSHMEM_MPI_SUPPORT=1 \
NVSHMEM_MPI_IS_OMPI=1 \
NVSHMEM_NVTX=1 \
NVSHMEM_PMIX_SUPPORT=1 \
NVSHMEM_SHMEM_SUPPORT=1 \
NVSHMEM_TEST_STATIC_LIB=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_TRACE=0 \
NVSHMEM_USE_DLMALLOC=0 \
NVSHMEM_USE_NCCL=1 \
NVSHMEM_USE_GDRCOPY=1 \
NVSHMEM_VERBOSE=0 \
NVSHMEM_DEFAULT_UCX=0 \
NVSHMEM_UCX_SUPPORT=1 \
NVSHMEM_IBGDA_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=0 \
NVSHMEM_IBDEVX_SUPPORT=0 \
NVSHMEM_IBRC_SUPPORT=0 \
LIBFABRIC_HOME=/usr \
NCCL_HOME=/usr \
GDRCOPY_HOME=/usr/local \
MPI_HOME=/opt/hpcx/ompi \
PMIX_HOME=/opt/hpcx/ompi \
UCX_HOME=/opt/hpcx/ucx \
cmake -S "${NVSHMEM_SRC_DIR}" -B "${NVSHMEM_BUILDDIR}" -G Ninja \
    -DUCX_DIR=/opt/hpcx/ucx/lib/cmake/ucx \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${NVSHMEM_PREFIX}" \
    -DCMAKE_CUDA_ARCHITECTURES="${NVSHMEM_CUDA_ARCH}" \
    -DCUDAToolkit_ROOT="${CUDA_DIR}" \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_DIR}"

cmake --build "${NVSHMEM_BUILDDIR}" -j"$(nproc)"
cmake --install "${NVSHMEM_BUILDDIR}"

cat > /etc/ld.so.conf.d/99-nvshmem.conf <<EOF
${NVSHMEM_PREFIX}/lib
${NVSHMEM_PREFIX}/lib64
EOF

mkdir -p /opt/alps/env
printf 'export NVSHMEM_VERSION=%q\n' "${NVSHMEM_VER}" >> /opt/alps/env/alps-versions.env

ldconfig

if [[ "${NVSHMEM_ENABLE_PYTHON}" == "1" ]]; then
    if python -c 'import nvshmem.core as _' >/dev/null 2>&1; then
        echo "[nvshmem4py] already importable; skipping wheel install"
    else
        cp_tag="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
        mach="$(python -c 'import platform; print(platform.machine())')"
        cuda_major="$("${CUDA_DIR}/bin/nvcc" --version | awk '
            /release [0-9]+/ {
                for (i = 1; i <= NF; i++) {
                    if ($i == "release") {
                        gsub(",", "", $(i+1))
                        split($(i+1), a, ".")
                        print a[1]
                        exit
                    }
                }
            }'
        )"

        best="$(
            find "${NVSHMEM_BUILDDIR}/dist" "${NVSHMEM_PREFIX}/lib" "${NVSHMEM_PREFIX}/lib64" \
                -type f -name "nvshmem4py_cu${cuda_major}-*.whl" 2>/dev/null \
            | grep -E "${cp_tag}-${cp_tag}-linux_${mach}\.whl$" \
            | sort -V | tail -n1 || true
        )"

        if [[ -z "${best}" ]]; then
            best="$(
                find "${NVSHMEM_BUILDDIR}/dist" "${NVSHMEM_PREFIX}/lib" "${NVSHMEM_PREFIX}/lib64" \
                    -type f -name "nvshmem4py_cu${cuda_major}-*.whl" 2>/dev/null \
                | grep -E "${cp_tag}-${cp_tag}-.*manylinux.*_${mach}\.whl$" \
                | sort -V | tail -n1 || true
            )"
        fi

        [[ -n "${best}" ]] || die "[nvshmem4py] no suitable wheel found (cu=${cuda_major}, cp=${cp_tag}, arch=${mach})"

        proxied_pip_install --no-cache-dir --no-deps --force-reinstall "${best}"

        req="${NVSHMEM_SRC_DIR}/nvshmem4py/requirements_cuda${cuda_major}.txt"
        [[ -f "${req}" ]] || die "nvshmem4py requirements not found: ${req}"

        constraint_file="$(mktemp)"

        REQ_FILE="${req}" CONSTRAINT_FILE="${constraint_file}" python - <<'PY'
import os
import re
from importlib.metadata import version, PackageNotFoundError

try:
    from packaging.requirements import Requirement
except Exception:
    raise SystemExit(0)

req_file = os.environ["REQ_FILE"]
constraint_file = os.environ["CONSTRAINT_FILE"]

def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

pathfinder_req = None
with open(req_file, "r", encoding="utf-8") as f:
    for raw in f:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        if norm(req.name) == "cuda-pathfinder":
            pathfinder_req = req
            break

if pathfinder_req is None:
    raise SystemExit(0)

installed = None
for candidate in ("cuda-pathfinder", "cuda.pathfinder", "cuda_pathfinder"):
    try:
        installed = version(candidate)
        break
    except PackageNotFoundError:
        pass

if installed is None:
    raise SystemExit(0)

if (not pathfinder_req.specifier) or pathfinder_req.specifier.contains(installed, prereleases=True):
    with open(constraint_file, "w", encoding="utf-8") as out:
        out.write(f"cuda-pathfinder=={installed}\n")
    print(f"[nvshmem4py] pinning cuda-pathfinder to installed version: {installed}")
PY

        if [[ -s "${constraint_file}" ]]; then
            proxied_pip_install --no-cache-dir -c "${constraint_file}" -r "${req}"
        else
            proxied_pip_install --no-cache-dir -r "${req}"
        fi

        rm -f "${constraint_file}"

        python -c 'import nvshmem.core as _; print("nvshmem4py ok")'
    fi
fi

rm -rf "${NVSHMEM_SRC_DIR}" "${NVSHMEM_BUILDDIR}"
