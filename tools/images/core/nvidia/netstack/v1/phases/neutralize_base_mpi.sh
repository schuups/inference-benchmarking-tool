#!/usr/bin/env bash
source "$(dirname "$0")/_helpers.sh"
setup_env

# The NGC base image ships an HPC-X stack at /usr/local/mpi (and friends)
# built against OMPI 4's internal hwloc/PMIx. When we rebuild OMPI 5 into
# /opt/hpcx/ompi, downstream link steps can drag in /usr/local/mpi's
# libpmix.so.2 alongside OMPI 5's libopen-pal, which causes:
#   undefined reference to `opal_hwloc201_hwloc_topology_set_flags`
# Remove the base stack and any ld.so / env hooks that reference it.
rm -rf /usr/local/mpi /usr/local/ucx /usr/local/ucc /usr/local/hcoll \
       /usr/local/sharp /usr/local/nvshmem /opt/hpcx /opt/hpcx-* 2>/dev/null || true
for f in /etc/ld.so.conf.d/*.conf; do
    [[ -f "$f" ]] || continue
    if grep -Eq '/usr/local/(mpi|ucx|ucc|hcoll|sharp|nvshmem)|/opt/hpcx' "$f"; then
        echo "Removing ld.so.conf.d entry: $f"
        rm -f "$f"
    fi
done
mkdir -p /opt/hpcx
ldconfig
