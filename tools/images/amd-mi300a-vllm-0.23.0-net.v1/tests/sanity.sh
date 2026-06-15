#!/bin/bash
# In-image sanity (AMD): confirm the baked Alps network stack loads and an RCCL
# collective runs. Launched inside the image via the CE EDF, e.g.:
#   srun --jobid=<H> --overlap --environment=tests/edf.toml \
#        --network=disable_rdzv_get bash tests/sanity.sh
# Single-process multi-GPU here validates intra-node (Infinity Fabric / xGMI) +
# that the aws-ofi-rccl net plugin loads; inter-node Slingshot bandwidth is the
# 2-node job.
set +e
echo "=== host $(hostname) | $(date -Is) ==="
echo "self-contained env: NCCL_NET=[$NCCL_NET] FI_PROVIDER=[$FI_PROVIDER] FI_CXI_RDZV_PROTO=[$FI_CXI_RDZV_PROTO] PMIX_MCA_psec=[$PMIX_MCA_psec]"
echo "SLURM_NETWORK=${SLURM_NETWORK:-<unset>}"
echo "--- alps-versions.env ---"; cat /opt/alps/env/alps-versions.env 2>&1
echo "--- libs (ldconfig) ---"; ldconfig -p | grep -iE "librccl-net|librccl.so|libfabric.so|librocshmem|libcxi" | head
echo "--- fi_info -p cxi (CXI provider + device enumeration) ---"
fi_info -p cxi 2>&1 | grep -iE "provider:|fabric:|domain:|nic|version:" | head -30
echo "--- fi_info -p cxi domains found: $(fi_info -p cxi 2>/dev/null | grep -c 'domain:') ---"
# Count MI300A GPUs without needing rocm-smi parsing quirks.
ngpu="$(rocminfo 2>/dev/null | grep -cE '^[[:space:]]*Name:[[:space:]]*gfx94')"
[[ "${ngpu}" -eq 0 ]] && ngpu="$(ls -d /dev/dri/renderD* 2>/dev/null | wc -l)"
echo "--- GPUs visible (gfx94x): ${ngpu} ---"
if [[ "${ngpu}" -ge 1 ]]; then
  echo "--- RCCL all_reduce_perf -g ${ngpu} (NCCL_DEBUG=INFO; expect NET/OFI + AWS Libfabric) ---"
  NCCL_DEBUG=INFO all_reduce_perf -b 8 -e 256M -f 2 -g "${ngpu}" 2>&1 \
    | grep -iE "NET/OFI|AWS Libfabric|Using network|NET/Plugin|Loaded net plugin|busbw|out-of-place|Avg bus|# Out of bounds" \
    | head -60
fi
echo "SANITY_DONE"
