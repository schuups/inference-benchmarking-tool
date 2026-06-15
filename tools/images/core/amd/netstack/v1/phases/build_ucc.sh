#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

hpcx=/opt/hpcx
rm -rf "${hpcx}/ucc"
git clone --depth 1 --branch "v${UCC_VERSION}" https://github.com/openucx/ucc.git /tmp/ucc
pushd /tmp/ucc
./autogen.sh

# AMD: --with-rocm replaces NVIDIA --with-cuda/--with-nvcc-gencode; --with-rccl
# (base ROCm RCCL) replaces --with-nccl for the collective TL.
./configure \
    --prefix="${hpcx}/ucc" \
    --with-ucx="${hpcx}/ucx" \
    --with-rocm="${ROCM_DIR}" \
    --with-rccl="${ROCM_DIR}" \
    --disable-static \
    LDFLAGS="-Wl,-rpath,${hpcx}/ucc/lib -Wl,-rpath,${hpcx}/ucx/lib -Wl,-rpath,${ROCM_DIR}/lib -Wl,-rpath,${ROCM_DIR}/lib64"
make -j"$(nproc)"
# Avoid libtool's ROCm install-time relink hitting the overlay "mv: cannot overwrite
# directory" bug (the RCCL TL module libucc_tl_rccl relinks at install). Robust fix
# (same as build_ompi5.sh): blank relink_command right after install mode sources each
# .la in the generated libtool scripts, so it holds even for modules make re-links.
python3 - <<'PY'
import glob, re
pat = re.compile(r'(\n[ \t]*relink_command=\n[ \t]*func_source "\$file"\n)')
n = 0
for lt in glob.glob('**/libtool', recursive=True):
    try: s = open(lt).read()
    except Exception: continue
    s2, c = pat.subn(r'\1\trelink_command=\n', s)
    if c: open(lt, 'w').write(s2); n += c
print(f"[ucc] neutered libtool install relink at {n} site(s)")
PY
make install
mkdir -p /opt/alps/env
printf 'export UCC_VERSION=%q\n' "${UCC_VERSION}" >> /opt/alps/env/alps-versions.env
popd
rm -rf /tmp/ucc
