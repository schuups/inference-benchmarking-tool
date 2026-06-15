#!/usr/bin/env bash
# aws-ofi-nccl (UPSTREAM aws/aws-ofi-nccl) built --with-rocm = the RCCL<->libfabric/CXI
# net plugin (produces librccl-net.so). Upstream aws/aws-ofi-nccl (v1.19.2) supports RCCL
# via --with-rocm (ROCm support landed after v1.17.3, the tag the NVIDIA netstack pins;
# net-plugin ABI v6+) and its configure defines __HIP_PLATFORM_AMD__ itself. The AMD fork
# ROCm/aws-ofi-rccl is abandoned at ABI v4/v5 — too old for ROCm 7.2's RCCL (v6-v10).
# Built against the base ROCm's RCCL (NCCL-API 2.27.x) for runtime ABI compatibility;
# RCCL itself is NOT rebuilt.
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

# configure's CHECK_PKG_ROCM compiles a hip/hip_runtime_api.h probe that FALSE-NEGATIVES
# in this netstack build env (a confdefs.h interaction accumulated by the prior phases'
# autoconf checks) — even though the header compiles cleanly with
# `-D__HIP_PLATFORM_AMD__ -I/opt/rocm/include` here (verified directly). Preset the
# autoconf header-cache vars to bypass the broken probe; ROCm is then detected
# (do_rocm=1, configure AC_DEFINEs __HIP_PLATFORM_AMD__) and the real plugin build
# compiles correctly. TODO: root-cause the confdefs interaction.
export ac_cv_header_hip_hip_runtime_api_h=yes
export ac_cv_header_hsa_hsa_h=yes

# AMD: --with-rocm + --with-rccl replace the NVIDIA --with-cuda; configure defines
# __HIP_PLATFORM_AMD__ itself. Minimal known-good config (verified in isolation):
# NO --with-mpi — adding it makes CHECK_PKG_ROCM's hip/hip_runtime_api.h check fail
# (MPI's include flags pollute the ROCm header probe), and the plugin itself needs no
# MPI (aws-ofi's MPI is only for its own functional tests; we validate via rccl-tests).
# Let hwloc auto-detect from the system.
./configure \
    --prefix=/usr \
    --with-libfabric=/usr \
    --with-rocm="${ROCM_DIR}" \
    --with-rccl="${ROCM_DIR}" \
    --without-mpi

# critical fix: remove /usr/include being injected as -isystem
find . \( \
    -name 'Makefile' -o -name 'Makefile.in' -o -name 'Makefile.am' -o -name '*.mk' -o -name 'config.status' -o -name 'libtool' \
\) -type f -print0 \
| xargs -0 -r sed -i 's| -isystem /usr/include||g'

make -j"$(nproc)"

# Neuter libtool's install-time relink (overlay "mv: cannot overwrite directory"
# bug; same fix as build_ompi5.sh). The built .so already has correct rpaths.
python3 - <<'PY'
import glob, re
pat = re.compile(r'(\n[ \t]*relink_command=\n[ \t]*func_source "\$file"\n)')
n = 0
for lt in glob.glob('**/libtool', recursive=True):
    try: s = open(lt).read()
    except Exception: continue
    s2, c = pat.subn(r'\1\trelink_command=\n', s)
    if c: open(lt, 'w').write(s2); n += c
print(f"[aws-ofi-nccl] neutered libtool install relink at {n} site(s)")
PY

make install

# RCCL loads the net plugin by name librccl-net.so (preferred) or libnccl-net.so.
# Ensure both aliases resolve regardless of which name the build emitted. Glob-based:
# `ls a b | head` would exit non-zero under set -e/pipefail when one name is absent.
for d in /usr/lib /lib; do
  base=""
  for cand in "$d"/librccl-net.so "$d"/libnccl-net.so "$d"/librccl-net.so.* "$d"/libnccl-net.so.*; do
    [ -e "$cand" ] && { base="$cand"; break; }
  done
  [ -n "$base" ] || continue
  for alias in librccl-net.so libnccl-net.so; do
    [ -e "$d/$alias" ] || ln -sf "$(basename "$base")" "$d/$alias"
  done
done

mkdir -p /opt/alps/env
printf 'export AWS_OFI_NCCL_VERSION=%q\n' "$(./m4/get_version.sh 2>/dev/null || echo unknown)" >> /opt/alps/env/alps-versions.env
printf 'export AWS_OFI_NCCL_COMMIT=%q\n' "${AWS_OFI_NCCL_COMMIT}" >> /opt/alps/env/alps-versions.env

popd
rm -rf /tmp/aws-ofi-nccl
ldconfig
