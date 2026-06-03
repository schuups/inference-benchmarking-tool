# syntax=docker/dockerfile:1

# Standalone Containerfile: extends nvcr.io/nvidia/vllm:26.04-py3 with the
# Alps HPC networking stack for the HPE Slingshot/CXI interconnect.
# vLLM is already installed in the base image; this layer replaces the generic
# HPC-X/EFA components with Slingshot-optimised builds.
#
# Build from any directory (no repo context needed):
#   podman build \
#       --file /path/to/this/Containerfile \
#       --tag <registry>/ngc-vllm:26.04-py3-alps5-dev \
#       /dev/null

FROM nvcr.io/nvidia/vllm:26.04-py3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG OCI_SOURCE="unknown"
ARG OCI_REVISION="unknown"
ARG OCI_CREATED="unknown"
ARG OCI_DESCRIPTION="unknown"
ARG CSCS_ALPS_GIT_COMMIT_SHORT="unknown"

LABEL org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.revision="${OCI_REVISION}" \
      org.opencontainers.image.created="${OCI_CREATED}" \
      org.opencontainers.image.description="${OCI_DESCRIPTION}" \
      com.cscs.alps.git.repo="${OCI_SOURCE}" \
      com.cscs.alps.git.commit="${OCI_REVISION}" \
      com.cscs.alps.git.commit_short="${CSCS_ALPS_GIT_COMMIT_SHORT}"

# HPC-X / EFA directories to remove before installing the Alps stack.
# Both nccl_spectrum-x_plugin (PyTorch NGC) and ncclnet_plugin (NeMo NGC)
# are listed; removal is best-effort (rm -rf ... || true).
ARG REMOVE_HPCX_DIRS="/opt/hpcx/hcoll \
    /opt/hpcx/nccl_rdma_sharp_plugin \
    /opt/hpcx/nccl_spectrum-x_plugin \
    /opt/hpcx/ncclnet_plugin \
    /opt/hpcx/sharp \
    /opt/amazon/aws-ofi-nccl \
    /opt/amazon/efa"

# Component versions — keep in sync with Containerfile.ngc-alps
ARG XPMEM_REF=3bcab55479489fdd93847fa04c58ab16e9c0b3fd
ARG GDRCOPY_VER=2.5.1

ARG CASSINI_HEADERS_VERSION=release/shs-13.0.0
ARG CXI_DRIVER_VERSION=release/shs-13.0.0
ARG LIBCXI_VERSION=release/shs-13.0.0

ARG LIBFABRIC_COMMIT=79fd4514977a8f15844fbe26df9f5930eb96311a
ARG LIBFABRIC_PATCH="libfabric/prov_cxi_netns_support.79fd451.patch"

ARG NCCL_VER=2.28.7-1
ARG NCCL_PATCH="nccl/net.cc.2.29.7-1.patch"

ARG UCX_VERSION=1.20.0
ARG UCC_VERSION=1.7.0
ARG OMPI_VER=5.0.10

ARG AWS_OFI_NCCL_COMMIT=93fa84ca880325013b2f639356bbb60f075148af
ARG AWS_OFI_NCCL_PATCH=""

ARG NVSHMEM_VER=3.6.5-0
ARG NVSHMEM_PATCH="nvshmem/requirements-v3.6.5-0.patch"

ARG NCCL_TESTS_VER=2.18.2
ARG OSU_VERSION=7.5.2

# =============================================================================
# Patches
# =============================================================================

COPY <<'LIBFABRIC_PATCH_END' /opt/alps/patches/libfabric/prov_cxi_netns_support.79fd451.patch
diff --git i/prov/cxi/src/cxip_nic.c w/prov/cxi/src/cxip_nic.c
index bd2bf1c35..885809af9 100644
--- i/prov/cxi/src/cxip_nic.c
+++ w/prov/cxi/src/cxip_nic.c
@@ -143,33 +143,6 @@ static int cxip_nic_get_rgroup_vni_ss_env(struct cxip_if *nic_if,
 	return FI_SUCCESS;
 }

-/*
- * @brief Retrieve the current process's Network Namespace ID, if available.
- *
- * @param netns  Updated with the detected network namespace ID.
- *
- * @return 0 on success (with *netns set accordingly), or -1 on failure
- *         (e.g., no Network Namespace support or unable to determine it).
- *         On failure, *netns is set to 0, which represents an invalid
- *         inode/netns value.
- */
-static int cxip_process_network_namespace(unsigned int *netns)
-{
-	struct stat st;
-
-	if (stat("/proc/self/ns/net", &st)) {
-		CXIP_WARN("failed to stat /proc/self/ns/net (no namespace support or none set?): %d:%s\n",
-			  errno, strerror(errno));
-		*netns = 0; /* invalid Namespace ID*/
-		return -1;
-	}
-
-	/* inode uniquely identifies the Network Namespace ID */
-	*netns = st.st_ino;
-
-	return 0;
-}
-
 static int cxip_nic_get_best_rgroup_vni(struct cxip_if *nic_if,
 					unsigned int *rgroup,
 					unsigned int *vni)
@@ -178,20 +151,15 @@ static int cxip_nic_get_best_rgroup_vni(struct cxip_if *nic_if,
 	struct cxil_svc_list *svc_list;
 	uid_t uid;
 	gid_t gid;
-	unsigned int cur_netns, netns;
 	int i;
 	int j;
 	struct cxi_svc_desc *desc;
 	int found_uid;
 	int found_gid;
-	int found_netns;
 	int found_unrestricted;

 	uid = geteuid();
 	gid = getegid();
-	ret = cxip_process_network_namespace(&cur_netns);
-	if (ret)
-		CXIP_WARN("cxip_process_network_namespace failed, proceeding without network namespace support\n");

 	ret = cxil_get_svc_list(nic_if->dev, &svc_list);
 	if (ret) {
@@ -201,13 +169,12 @@ static int cxip_nic_get_best_rgroup_vni(struct cxip_if *nic_if,
 	}

 	/* Find the service indexes which can be used by this process. These are
-	 * services which have no membership restrictions, or have a matching UID,
-	 * GID or The Network Namespace ID(NETNS).If there are multiple service
-	 * IDs which match uid, gid, and netns, the priority is netns, uid and then gid.
+	 * services which are unrestricted, have a matching UID, or have a
+	 * matching GID. If there are multiple service IDs which could match
+	 * unrestricted, UID, and GID, only the first one found is selected.
 	 */
 	found_uid = -1;
 	found_gid = -1;
-	found_netns = -1;
 	found_unrestricted = -1;

 	for (i = svc_list->count - 1; i >= 0; i--) {
@@ -232,19 +199,12 @@ static int cxip_nic_get_best_rgroup_vni(struct cxip_if *nic_if,
 			 found_gid == -1)
 			found_gid = i;
 		}
-
-		/* Check if service ID created with netns type and matching with current Network Namespace ID */
-		if (cur_netns && found_netns == -1)
-			if (!cxil_svc_get_netns(nic_if->dev, desc->svc_id, &netns) && (netns == cur_netns))
-				found_netns = i;
 	}

-	/* The service ID matching with the netns is prioritized over service ID with uid or gid. */
-	if (found_netns != -1) {
-		i = found_netns;
-	} else if (found_uid != -1) {
+	/* Prioritized list for matching service ID. */
+	if (found_uid != -1)
 		i = found_uid;
-	} else if (found_gid != -1) {
+	else if (found_gid != -1) {
 		i = found_gid;
 	} else if (found_unrestricted != -1) {
 		i = found_unrestricted;
LIBFABRIC_PATCH_END

COPY <<'NCCL_PATCH_END' /opt/alps/patches/nccl/net.cc.2.29.7-1.patch
diff --git a/src/plugin/net.cc b/src/plugin/net.cc
index 808d686..95c34ce 100644
--- a/src/plugin/net.cc
+++ b/src/plugin/net.cc
@@ -82,8 +82,12 @@ static ncclResult_t ncclNetPluginUnload(netPluginLib_t* pluginLib) {
   if ((pluginLib->dlHandle) && ((pluginLib->ncclNetPluginRefCount) == 0)) {
     INFO(NCCL_INIT|NCCL_NET, "Unloading plugin %s", pluginLib->name);
     NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeNet));
+    char savedName[sizeof(pluginLib->name)];
+    strcpy(savedName,pluginLib->name);
     // memset will reset the status to ncllNetPluginStateLoadReady
     memset(pluginLib, 0, sizeof(netPluginLib_t));
+    // reset plugin name
+    strcpy(pluginLib->name, savedName);
     // reset the count of devices to UNDEF_DEV_COUNT
     pluginLib->netPhysDevs = pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
NCCL_PATCH_END

COPY <<'NVSHMEM_PATCH_END' /opt/alps/patches/nvshmem/requirements-v3.6.5-0.patch
diff --git c/nvshmem4py/MANIFEST.in i/nvshmem4py/MANIFEST.in
index 6e7f7ce..9d1ae9c 100644
--- c/nvshmem4py/MANIFEST.in
+++ i/nvshmem4py/MANIFEST.in
@@ -2,7 +2,6 @@ include pyproject.toml
 include setup.py
 include README.md
 include License.txt
-include requirements_cuda11.txt
 include requirements_cuda12.txt
 include requirements_cuda13.txt
 include nvshmem/bindings/_internal/*.pyx
diff --git c/nvshmem4py/nvshmem/core/interop/__init__.py i/nvshmem4py/nvshmem/core/interop/__init__.py
new file mode 100644
index 0000000..e69de29
diff --git c/nvshmem4py/requirements_build.txt i/nvshmem4py/requirements_build.txt
index e07ecbe..97ca522 100644
--- c/nvshmem4py/requirements_build.txt
+++ i/nvshmem4py/requirements_build.txt
@@ -12,4 +12,3 @@ wheel
 auditwheel
 patchelf
 # The following are Numbast requirements for header search
-cuda-toolkit[cudart,nvcc,curand,cccl,nvvm]==13.*
diff --git c/nvshmem4py/requirements_cuda12.txt i/nvshmem4py/requirements_cuda12.txt
index 7ad0c1c..9e95c24 100644
--- c/nvshmem4py/requirements_cuda12.txt
+++ i/nvshmem4py/requirements_cuda12.txt
@@ -1,6 +1,5 @@
 # These are the run-time dependencies of nvshmem4py
 # They are non-negotiables
-nvidia-nvshmem-cu12
 cuda-python>=12.0,<=12.9
 cuda.core>=0.5.0
 cuda.pathfinder>=1.2.3
diff --git c/nvshmem4py/requirements_cuda13.txt i/nvshmem4py/requirements_cuda13.txt
index b7c22e1..3e92351 100644
--- c/nvshmem4py/requirements_cuda13.txt
+++ i/nvshmem4py/requirements_cuda13.txt
@@ -1,6 +1,5 @@
 # These are the run-time dependencies of nvshmem4py
 # They are non-negotiables
-nvidia-nvshmem-cu13
 cuda-python>=13.0,<14.0
 cuda.core>=0.5.0
 cuda.pathfinder>=1.2.3
diff --git c/nvshmem4py/scripts/generate_pyproject_toml.py i/nvshmem4py/scripts/generate_pyproject_toml.py
index ebc804e..bf8c242 100755
--- c/nvshmem4py/scripts/generate_pyproject_toml.py
+++ i/nvshmem4py/scripts/generate_pyproject_toml.py
@@ -31,9 +31,12 @@ dependencies = [
 packages = [
     "nvshmem",
     "nvshmem.bindings",
+    "nvshmem.bindings._internal",
     "nvshmem.bindings.device",
     "nvshmem.bindings.device.numba",
+    "nvshmem.bindings.device.cute",
     "nvshmem.core",
+    "nvshmem.core.interop",
     "nvshmem.core.device",
     "nvshmem.core.device.numba",
     "nvshmem.core.device.cute"
NVSHMEM_PATCH_END

# =============================================================================
# HPC stack install script
# =============================================================================

COPY <<'HPC_STACK_END' /opt/alps/install-alps-hpc-stack.sh
#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

die() {
    echo "ERROR: $*" >&2
    exit 1
}

apt_install_build_deps() {

    # Use JFrog Artifactory as an APT proxy/cache for Ubuntu packages, to speed
    # up installs and reduce load on Ubuntu mirrors.
    sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://jfrog.svc.cscs.ch/artifactory/ubuntu|' \
        -e 's|http://security.ubuntu.com/ubuntu|https://jfrog.svc.cscs.ch/artifactory/ubuntu|' \
        -e 's|http://ports.ubuntu.com/ubuntu-ports|https://jfrog.svc.cscs.ch/artifactory/ubuntu-ports|' \
        /etc/apt/sources.list.d/ubuntu.sources
    printf '%s\n%s' "Acquire::http::AllowRedirect "true";" "Acquire::http::Pipeline-Depth "0";" \
        > /etc/apt/apt.conf.d/99-jfrog-proxy
    apt -o "Acquire::https::Verify-Peer=false" update
    apt -o "Acquire::https::Verify-Peer=false" install ca-certificates

    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates pkg-config automake autoconf libtool cmake \
        bc gdb strace wget curl git bzip2 python3 gfortran \
        rdma-core numactl \
        libconfig-dev libuv1-dev libfuse-dev libfuse3-dev libyaml-dev libnl-3-dev \
        libnuma-dev libsensors-dev libcurl4-openssl-dev libjson-c-dev \
        libsox-fmt-all \
        devscripts debhelper fakeroot dh-make
    rm -rf /var/lib/apt/lists/*
}

proxied_pip_install() {
    python -m pip install -i https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple "$@"
}

remove_efa() {
    rm -rf /opt/amazon/efa || true
    grep -R "/opt/amazon/efa" -n /etc/ld.so.conf.d || true
    for f in /etc/ld.so.conf.d/*; do
        [[ -f "$f" ]] || continue
        if grep -q "/opt/amazon/efa" "$f"; then rm -f "$f"; fi
    done
    ldconfig
}

remove_hpcx_plugins() {
    # REMOVE_HPCX_DIRS can be space-separated or newline-separated
    if [[ -n "${REMOVE_HPCX_DIRS:-}" ]]; then
        while IFS= read -r d; do
            [[ -z "$d" ]] && continue
            echo "Removing HPCX plugin dir: $d"
            rm -rf "$d" || true
        done < <(printf "%s\n" ${REMOVE_HPCX_DIRS})
        ldconfig
    fi
}

apply_patch_if_set() {
    local patch_rel="${1:-}"
    [[ -z "$patch_rel" ]] && return 0

    local patch="/opt/alps/patches/${patch_rel}"
    [[ -f "$patch" ]] || die "Patch not found: ${patch}"

    git apply --check --whitespace=nowarn "$patch"
    git apply --whitespace=nowarn "$patch"
}

detect_cuda_dir() {
    if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
        echo "${CUDA_HOME}"
        return 0
    fi
    if [[ -n "${CUDA_PATH:-}" && -d "${CUDA_PATH}" ]]; then
        echo "${CUDA_PATH}"
        return 0
    fi
    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path
        nvcc_path="$(command -v nvcc)"
        echo "$(cd "$(dirname "$nvcc_path")/.." && pwd)"
        return 0
    fi
    if [[ -d /usr/local/cuda ]]; then
        echo /usr/local/cuda
        return 0
    fi
    return 1
}

build_xpmem() {
    local ref="${XPMEM_REF}"
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
}

build_gdrcopy() {
    local ver="${GDRCOPY_VER}"
    git clone --depth 1 --branch "v${ver}" https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
    pushd /tmp/gdrcopy
    make CC=gcc CUDA="${CUDA_DIR}" lib -j"$(nproc)"
    make lib_install
    popd
    rm -rf /tmp/gdrcopy
    ldconfig
}

build_cxi_bits() {
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
}

build_libfabric() {
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
}

build_nccl_deb() {
    curl -fsSL "https://github.com/NVIDIA/nccl/archive/refs/tags/v${NCCL_VER}.tar.gz" -o /tmp/nccl.tar.gz
    tar -C /tmp -xzf /tmp/nccl.tar.gz
    pushd "/tmp/nccl-${NCCL_VER}"
    apply_patch_if_set "${NCCL_PATCH}"
    make -j"$(nproc)" pkg.debian.build CUDA_HOME="${CUDA_DIR}"
    dpkg -i build/pkg/deb/*.deb
    mkdir -p /opt/alps/env
    printf 'export NCCL_VERSION=%q\n' "${NCCL_VER}" >> /opt/alps/env/alps-versions.env
    pushd ext-profiler/inspector
    make -j"$(nproc)" CUDA_HOME="${CUDA_DIR}"
    install -D -m 0644 libnccl-profiler-inspector.so /usr/local/lib/libnccl-profiler-inspector.so
    popd
    popd
    rm -rf "/tmp/nccl-${NCCL_VER}" /tmp/nccl.tar.gz
    ldconfig
}

build_ucx() {
    local hpcx=/opt/hpcx
    rm -rf "${hpcx}/ucx"
    curl -fsSL "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" -o /tmp/ucx.tar.gz
    tar -C /tmp -xzf /tmp/ucx.tar.gz
    pushd "/tmp/ucx-${UCX_VERSION}"
    mkdir -p build && cd build
    ../configure \
        --prefix="${hpcx}/ucx" \
        --with-cuda="${CUDA_DIR}" \
        --with-gdrcopy=/usr/local \
        --enable-mt \
        --enable-devel-headers
    make -j"$(nproc)"
    make install
    mkdir -p /opt/alps/env
    printf 'export UCX_VERSION=%q\n' "${UCX_VERSION}" >> /opt/alps/env/alps-versions.env
    popd
    rm -rf "/tmp/ucx-${UCX_VERSION}" /tmp/ucx.tar.gz
}

build_ucc() {
    local hpcx=/opt/hpcx
    rm -rf "${hpcx}/ucc"
    git clone --depth 1 --branch "v${UCC_VERSION}" https://github.com/openucx/ucc.git /tmp/ucc
    pushd /tmp/ucc
    ./autogen.sh

    local gencode_sm90='-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90'
    local gencode_sm90a='-gencode arch=compute_90a,code=sm_90a -gencode arch=compute_90a,code=compute_90a'
    local gencode="${gencode_sm90} ${gencode_sm90a}"

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
}

build_ompi5() {
    local hpcx=/opt/hpcx
    rm -rf "${hpcx}/ompi"
    curl -fsSL "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OMPI_VER}.tar.gz" -o /tmp/ompi.tar.gz
    tar -C /tmp -xzf /tmp/ompi.tar.gz
    pushd "/tmp/openmpi-${OMPI_VER}"
    ./configure \
        --prefix="${hpcx}/ompi" \
        --with-ofi=/usr \
        --with-ucx="${hpcx}/ucx" \
        --with-ucc="${hpcx}/ucc" \
        --enable-oshmem \
        --with-cuda="${CUDA_DIR}" \
        --with-cuda-libdir="${CUDA_DIR}/lib64/stubs"
    make -j"$(nproc)"
    make install
    mkdir -p /opt/alps/env
    printf 'export OMPI_VERSION=%q\n' "${OMPI_VER}" >> /opt/alps/env/alps-versions.env
    popd
    rm -rf "/tmp/openmpi-${OMPI_VER}" /tmp/ompi.tar.gz
    ldconfig
}

build_aws_ofi_nccl() {
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
}

build_nvshmem() {
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

    # Clone repo
    git clone --depth 1 --branch "v${NVSHMEM_VER}" https://github.com/NVIDIA/nvshmem.git "${NVSHMEM_SRC_DIR}"

    # Apply local nvshmem4py patch set
    pushd "${NVSHMEM_SRC_DIR}" >/dev/null
    apply_patch_if_set "${NVSHMEM_PATCH}"
    popd >/dev/null

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
    cmake -S "${NVSHMEM_SRC_DIR}" -B "${NVSHMEM_BUILDDIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${NVSHMEM_PREFIX}" \
        -DCMAKE_CUDA_ARCHITECTURES="${NVSHMEM_CUDA_ARCH}"

    cmake --build "${NVSHMEM_BUILDDIR}" -j"$(nproc)"
    cmake --install "${NVSHMEM_BUILDDIR}"

    # Ensure loader finds our NVSHMEM without LD_LIBRARY_PATH
    cat > /etc/ld.so.conf.d/99-nvshmem.conf <<NVSHMEM_LDCONF
${NVSHMEM_PREFIX}/lib
${NVSHMEM_PREFIX}/lib64
NVSHMEM_LDCONF

    mkdir -p /opt/alps/env
    printf 'export NVSHMEM_VERSION=%q\n' "${NVSHMEM_VER}" >> /opt/alps/env/alps-versions.env

    ldconfig

    # Build installs/copies wheels into the tree, but does not install into python.
    if [[ "${NVSHMEM_ENABLE_PYTHON}" == "1" ]]; then
        if python -c 'import nvshmem.core as _' >/dev/null 2>&1; then
            echo "[nvshmem4py] already importable; skipping wheel install"
        else
            local cp_tag mach cuda_major best req
            local constraint_file

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

            # Prefer the most specific wheel: linux_<arch> > manylinux
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

            # If cuda-pathfinder is already installed and satisfies the upstream
            # requirement, pin it to the installed version for this install.
            REQ_FILE="${req}" CONSTRAINT_FILE="${constraint_file}" python - <<'NVSHMEM4PY_PIN'
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
NVSHMEM4PY_PIN

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
}

build_nccl_tests() {
    git clone --depth 1 --branch "v${NCCL_TESTS_VER}" https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests
    pushd /tmp/nccl-tests
    MPI=1 MPI_HOME=/opt/hpcx/ompi CUDA_HOME="${CUDA_DIR}" make -j"$(nproc)"
    install -d /usr/local/bin
    find build -maxdepth 1 -type f -executable -name '*_perf' -print -exec install -m 0755 {} /usr/local/bin/ \;
    popd
    rm -rf /tmp/nccl-tests
}

build_osu() {
    curl -fsSL "http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz" -o /tmp/osu.tar.gz
    tar --no-same-owner --no-same-permissions -C /tmp -xzf /tmp/osu.tar.gz
    pushd "/tmp/osu-micro-benchmarks-${OSU_VERSION}"
    CC=/opt/hpcx/ompi/bin/mpicc \
    CFLAGS="-O3 -lcuda -lnvidia-ml" \
    ./configure \
        --prefix=/usr/local \
        --enable-cuda \
        --with-cuda-include="${CUDA_DIR}/include" \
        --with-cuda-libpath="${CUDA_DIR}/lib64"
    make -j"$(nproc)"
    make install
    popd
    rm -rf "/tmp/osu-micro-benchmarks-${OSU_VERSION}" /tmp/osu.tar.gz
    ldconfig
}

clean_up() {
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
}

main() {
    CUDA_DIR="$(detect_cuda_dir)" || die "Could not determine CUDA directory..."
    export CUDA_DIR
    export CUDA_HOME="${CUDA_HOME:-$CUDA_DIR}"
    export CUDA_PATH="${CUDA_PATH:-$CUDA_DIR}"

    apt_install_build_deps

    remove_efa
    remove_hpcx_plugins

    build_xpmem
    build_gdrcopy
    build_cxi_bits
    build_libfabric
    build_nccl_deb
    build_ucx
    build_ucc
    build_ompi5
    build_aws_ofi_nccl
    build_nvshmem

    build_nccl_tests
    build_osu

    clean_up
}

main "$@"
HPC_STACK_END

RUN chmod +x /opt/alps/install-alps-hpc-stack.sh && \
    /opt/alps/install-alps-hpc-stack.sh

# =============================================================================
# Runtime environment
# =============================================================================

COPY <<'RUNTIME_ENV_END' /opt/alps/env/alps-runtime.env
# -----------------------------------------------------------------------------
# Helper: set default only if UNSET, then export
# -----------------------------------------------------------------------------
defvar() {
  local name="${1:?var name required}"; shift
  local default="$*"

  if [[ ! ${!name+x} ]]; then
    printf -v "$name" '%s' "$default"
  fi
  export "$name"
}

# -----------------------------------------------------------------------------
# CUDA / general
# -----------------------------------------------------------------------------
defvar CUDA_CACHE_DISABLE "1"

# -----------------------------------------------------------------------------
# NCCL defaults
# -----------------------------------------------------------------------------
defvar NCCL_VERSION "${NCCL_VERSION:-}"
defvar NCCL_NET "AWS Libfabric"
defvar NCCL_SOCKET_IFNAME "hsn"
defvar NCCL_CROSS_NIC "1"
defvar NCCL_P2P_LEVEL "NVL"
defvar NCCL_NET_GDR_LEVEL "PHB"
defvar NCCL_NET_GDR_C2C "1"
defvar NCCL_NET_GDR_READ "1"
defvar NCCL_PROTO "^LL128"
defvar NCCL_NCHANNELS_PER_NET_PEER "4"
defvar AWS_OFI_NCCL_VERSION "${AWS_OFI_NCCL_VERSION:-}"

# -----------------------------------------------------------------------------
# Libfabric / CXI defaults
# -----------------------------------------------------------------------------
defvar FI_PROVIDER "cxi"
defvar FI_CXI_DEFAULT_CQ_SIZE "131072"
defvar FI_CXI_DEFAULT_TX_SIZE "16384"
defvar FI_CXI_DISABLE_HOST_REGISTER "1"
defvar FI_CXI_RDZV_PROTO "alt_read"
defvar FI_CXI_RDZV_EAGER_SIZE "0"
defvar FI_CXI_RDZV_GET_MIN "0"
defvar FI_CXI_RDZV_THRESHOLD "0"
defvar FI_CXI_RX_MATCH_MODE "hybrid"
defvar FI_MR_CACHE_MONITOR "userfaultfd"
defvar FI_MR_CACHE_MAX_SIZE "-1"
defvar FI_MR_CACHE_MAX_COUNT "524288"
defvar FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD "16777216"

# -----------------------------------------------------------------------------
# PMIx / OpenMPI defaults
# -----------------------------------------------------------------------------
defvar PMIX_MCA_psec "native"
defvar OMPI_MCA_btl "^tcp,uct,usnic"
defvar OMPI_MCA_pml "cm"
defvar OMPI_MCA_mtl "ofi"
defvar OMPI_MCA_opal_common_ofi_provider_include "cxi"

# -----------------------------------------------------------------------------
# NVSHMEM defaults
# -----------------------------------------------------------------------------
defvar NVSHMEM_VERSION "${NVSHMEM_VERSION:-}"
defvar NVSHMEM_REMOTE_TRANSPORT "libfabric"
defvar NVSHMEM_LIBFABRIC_PROVIDER "cxi"
defvar NVSHMEM_DISABLE_CUDA_VMM "1"
RUNTIME_ENV_END

RUN chmod 0644 /opt/alps/env/alps-runtime.env && \
    printf '%s\n' \
      'if [ -f /opt/alps/env/alps-versions.env ]; then . /opt/alps/env/alps-versions.env; fi' \
      'if [ -f /opt/alps/env/alps-runtime.env ]; then . /opt/alps/env/alps-runtime.env; fi' \
      > /etc/profile.d/99-alps-env.sh

# =============================================================================
# Runtime warnings
# =============================================================================

COPY <<'WARNING_TXT_END' /opt/alps/env/alps-runtime-warning.txt

WARNING: FI_CXI_RDZV_PROTO=alt_read is configured, but the Slurm network is
not set to disable rendezvous GET. Performance/stability may be impacted.

Expected Slurm launch:
  srun --network=disable_rdzv_get ...
or
  SLURM_NETWORK=disable_rdzv_get srun ...
WARNING_TXT_END

COPY <<'WARNING_SH_END' /opt/alps/env/alps-runtime-warning.sh
#!/usr/bin/env bash

_alps_warn_rdzv_get_enabled() {
  # Only warn in a Slurm task context
  [[ -n "${SLURM_STEP_ID:-}" || -n "${SLURM_JOB_ID:-}" ]] || return 0

  # Only once per process
  [[ -n "${ALPS_WARN_RDZV_DONE:-}" ]] && return 0
  ALPS_WARN_RDZV_DONE=1
  export ALPS_WARN_RDZV_DONE

  # Only relevant when cxi + alt_read
  [[ "${FI_PROVIDER:-}" == "cxi" ]] || return 0
  [[ "${FI_CXI_RDZV_PROTO:-}" == "alt_read" ]] || return 0

  # If SLURM_NETWORK is missing or does not contain disable_rdzv_get, warn.
  local net="${SLURM_NETWORK-}"
  case "$net" in
    *disable_rdzv_get*) return 0 ;;
  esac

  # Only rank 0 warns
  if [[ "${SLURM_PROCID:-0}" != "0" ]]; then
    return 0
  fi

  : "${ALPS_WARNING_TXT:=/opt/alps/env/alps-runtime-warning.txt}"

  {
    if [ -r "${ALPS_WARNING_TXT-}" ]; then
      cat "${ALPS_WARNING_TXT-}" >&2
    else
      echo "WARNING(alps-runtime): Alps runtime warning text missing: ${ALPS_WARNING_TXT-<unset>}" >&2
    fi

    echo "" >&2
    echo "Context:" >&2
    echo "  SLURM_NETWORK=${SLURM_NETWORK-<unset>}" >&2
    echo "  FI_PROVIDER=${FI_PROVIDER-<unset>}" >&2
    echo "  FI_CXI_RDZV_PROTO=${FI_CXI_RDZV_PROTO-<unset>}" >&2
    echo "" >&2
  } || true

  return 0
}

# Call immediately on sourcing.
_alps_warn_rdzv_get_enabled
unset -f _alps_warn_rdzv_get_enabled
WARNING_SH_END

COPY <<'WARNING_PY_END' /opt/alps/env/alps-runtime-warning.py
#!/usr/bin/env python3
import os
import sys

WARNING_TXT = os.environ.get("ALPS_WARNING_TXT", "/opt/alps/env/alps-runtime-warning.txt")


def _slurm_context() -> bool:
    return bool(os.environ.get("SLURM_STEP_ID") or os.environ.get("SLURM_JOB_ID"))


def _relevant() -> bool:
    return (
        os.environ.get("FI_PROVIDER", "") == "cxi"
        and os.environ.get("FI_CXI_RDZV_PROTO", "") == "alt_read"
    )


def _rank_is_zero() -> bool:
    procid = os.environ.get("SLURM_PROCID")
    return procid is None or procid == "" or procid == "0"


def _read_static() -> str:
    try:
        with open(WARNING_TXT, "r", encoding="utf-8") as f:
            return f.read().rstrip()
    except Exception:
        return f"WARNING(alps-runtime): missing static text file: {WARNING_TXT}"


def maybe_warn() -> None:
    if not _slurm_context():
        return
    if not _relevant():
        return

    if os.environ.get("ALPS_WARN_RDZV_DONE") == "1":
        return
    os.environ["ALPS_WARN_RDZV_DONE"] = "1"

    if not _rank_is_zero():
        return

    slurm_network = os.environ.get("SLURM_NETWORK", "<unset>")
    if "disable_rdzv_get" in slurm_network:
        return

    msg = _read_static()
    sys.stderr.write(msg + "\n\n")
    sys.stderr.write("Context:\n")
    sys.stderr.write(f"  SLURM_NETWORK={slurm_network}\n")
    sys.stderr.write(f"  FI_PROVIDER={os.environ.get('FI_PROVIDER','')}\n")
    sys.stderr.write(f"  FI_CXI_RDZV_PROTO={os.environ.get('FI_CXI_RDZV_PROTO','')}\n\n")


if __name__ == "__main__":
    maybe_warn()
WARNING_PY_END

COPY <<'WARNINGS_INSTALLER_END' /opt/alps/env/install-alps-runtime-warnings.sh
#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR(alps-runtime-install): $*" >&2; exit 1; }
info() { echo "INFO(alps-runtime-install): $*" >&2; }

PYTHON_BIN="python3"
BASH_BIN="bash"

WARNING_TXT="/opt/alps/env/alps-runtime-warning.txt"
WARN_SH="/opt/alps/env/alps-runtime-warning.sh"
WARN_PY="/opt/alps/env/alps-runtime-warning.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="${2:?}"; shift 2;;
    --bash)   BASH_BIN="${2:?}"; shift 2;;
    --txt)    WARNING_TXT="${2:?}"; shift 2;;
    --sh)     WARN_SH="${2:?}"; shift 2;;
    --py)     WARN_PY="${2:?}"; shift 2;;
    -h|--help)
      cat <<USAGE
Usage:
  $0 [--python python3] [--bash bash] [--txt /opt/alps/env/alps-runtime-warning.txt] \
     [--sh /opt/alps/env/alps-runtime-warning.sh] [--py /opt/alps/env/alps-runtime-warning.py]
USAGE
      exit 0
      ;;
    *) die "Unknown argument: $1";;
  esac
done

[[ -r "$WARNING_TXT" ]] || die "missing/unreadable static warning text: $WARNING_TXT"
[[ -r "$WARN_SH" ]]     || die "missing/unreadable bash warning script: $WARN_SH"
[[ -r "$WARN_PY" ]]     || die "missing/unreadable python warning script: $WARN_PY"
command -v "$BASH_BIN" >/dev/null || die "bash not found: $BASH_BIN"
command -v "$PYTHON_BIN" >/dev/null || die "python not found: $PYTHON_BIN"

bash_env="$("$BASH_BIN" -c 'printf "%s" "${BASH_ENV-}"' 2>/dev/null || true)"
[[ -n "$bash_env" ]] || die "BASH_ENV is empty/unset in non-interactive bash; cannot install bash warnings"
[[ -f "$bash_env" ]] || die "BASH_ENV points to non-existent file: $bash_env"
[[ -w "$bash_env" ]] || die "BASH_ENV is not writable: $bash_env"
info "Detected BASH_ENV=$bash_env"

bash_marker="# ALPS_RUNTIME_WARNING_HOOK"
bash_hook="$(cat <<BASH_HOOK_CONTENT
$bash_marker
# Source Alps runtime warning
if [[ -r "$WARN_SH" ]]; then
  # shellcheck disable=SC1090
  source "$WARN_SH"
fi

BASH_HOOK_CONTENT
)"

if grep -Fqx "$bash_marker" "$bash_env"; then
  info "Bash hook already present in $bash_env"
else
  tmp="$(mktemp)"
  echo "$bash_hook" > "$tmp"
  cat "$bash_env" >> "$tmp"
  cat "$tmp" > "$bash_env"
  rm -f "$tmp"
  grep -Fqx "$bash_marker" "$bash_env" || die "failed to patch $bash_env"
  info "Prepended bash warning hook to $bash_env"
fi

stdlib_dir="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])' 2>/dev/null || true)"
[[ -n "$stdlib_dir" ]] || die "could not determine stdlib dir via $PYTHON_BIN"
[[ -d "$stdlib_dir" ]] || die "stdlib dir does not exist: $stdlib_dir"

sitecustomize="${stdlib_dir}/sitecustomize.py"
py_marker="# ALPS_RUNTIME_WARNING_HOOK"

py_hook="$(cat <<PY_HOOK_CONTENT
$py_marker
# Loads external Alps runtime warning logic (kept outside stdlib for maintainability).
def _alps_runtime_warn():
    try:
        import importlib.util
        _p = r"${WARN_PY}"
        _spec = importlib.util.spec_from_file_location("alps_runtime_warnings", _p)
        if _spec is None or _spec.loader is None:
            return
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)  # type: ignore[attr-defined]
        if hasattr(_m, "maybe_warn"):
            _m.maybe_warn()
    except Exception:
        # Never break python startup
        return

_alps_runtime_warn()
PY_HOOK_CONTENT
)"

if [[ ! -f "$sitecustomize" ]]; then
  info "Creating $sitecustomize"
  install -m 0644 /dev/null "$sitecustomize"
fi
[[ -w "$sitecustomize" ]] || die "sitecustomize.py not writable: $sitecustomize"

if grep -Fqx "$py_marker" "$sitecustomize"; then
  info "Python hook already present in $sitecustomize"
else
  printf "\n%s\n" "$py_hook" >> "$sitecustomize"
  grep -Fqx "$py_marker" "$sitecustomize" || die "failed to patch $sitecustomize"
  info "Patched python stdlib file: $sitecustomize"
fi

info "Sanity: bash non-interactive sources BASH_ENV"
"$BASH_BIN" -c 'true' >/dev/null 2>&1 || die "bash sanity check failed"
info "Sanity: python imports sitecustomize"
"$PYTHON_BIN" -c 'import site; print("ok")' >/dev/null 2>&1 || die "python sanity check failed"

info "Done."
WARNINGS_INSTALLER_END

RUN chmod +x /opt/alps/env/install-alps-runtime-warnings.sh && \
    /opt/alps/env/install-alps-runtime-warnings.sh
