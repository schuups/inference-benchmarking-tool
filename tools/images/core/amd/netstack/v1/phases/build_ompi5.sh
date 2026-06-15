#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

hpcx=/opt/hpcx
rm -rf "${hpcx}/ompi"
curl -fsSL "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OMPI_VER}.tar.gz" -o /tmp/ompi.tar.gz
tar -C /tmp -xzf /tmp/ompi.tar.gz
pushd "/tmp/openmpi-${OMPI_VER}"
# AMD: --with-rocm replaces NVIDIA --with-cuda/--with-cuda-libdir(stubs).
./configure \
    --prefix="${hpcx}/ompi" \
    --with-ofi=/usr \
    --with-ucx="${hpcx}/ucx" \
    --with-ucc="${hpcx}/ucc" \
    --with-pmix=internal \
    --with-hwloc=internal \
    --with-libevent=internal \
    --enable-oshmem \
    --disable-mpi-fortran \
    --with-rocm="${ROCM_DIR}" \
    LDFLAGS="-Wl,-rpath,${hpcx}/ompi/lib -Wl,-rpath,${hpcx}/ucx/lib -Wl,-rpath,${hpcx}/ucc/lib -Wl,-rpath,${ROCM_DIR}/lib -Wl,-rpath,${ROCM_DIR}/lib64"
make -j"$(nproc)"
# The AMD build (ROCm rpaths) makes libtool relink libraries at `make install`, and
# that relink hits an overlay "mv: cannot overwrite directory 'X.so'" bug. The NVIDIA
# build never relinks, so it never hits it. Per-.la stripping doesn't hold because
# `make install` re-links some archives (regenerating relink_command). Fix it at the
# libtool-SCRIPT level: blank relink_command right after install mode sources each
# .la (ltmain.sh: `relink_command=` / `func_source "$file"` / `if test -n
# "$relink_command"`), so the install relink is skipped for every library regardless
# of when its .la was written. Safe because the libs are already built with the final
# rpaths (LDFLAGS above). Covers the bundled openpmix/hwloc/libevent libtool scripts.
python3 - <<'PY'
import glob, re
pat = re.compile(r'(\n[ \t]*relink_command=\n[ \t]*func_source "\$file"\n)')
n = 0
for lt in glob.glob('**/libtool', recursive=True):
    try:
        s = open(lt).read()
    except Exception:
        continue
    s2, c = pat.subn(r'\1\trelink_command=\n', s)
    if c:
        open(lt, 'w').write(s2); n += c
print(f"[ompi] neutered libtool install-time relink at {n} site(s)")
PY
make install
mkdir -p /opt/alps/env
printf 'export OMPI_VERSION=%q\n' "${OMPI_VER}" >> /opt/alps/env/alps-versions.env
popd
rm -rf "/tmp/openmpi-${OMPI_VER}" /tmp/ompi.tar.gz
ldconfig
