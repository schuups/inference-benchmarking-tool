#!/bin/bash
# Post-push image sanity (SPECIFICATIONS.md §9.1 / §8): a short-lived 2-node job
# that validates the baked Alps network stack loads and achieves Slingshot
# bandwidth inter-node — NCCL (+ aws-ofi/CXI), OSU, and NVSHMEM, no errors.
#
# Submit via FirecREST AFTER build.sh has pushed, passing the digest-pinned EDF
# it wrote (env var SANITY_EDF). Pinning by registry digest sidesteps the stale
# enroot tag cache across rebuilds. Example (FirecREST submit_job):
#   env_vars = { SANITY_EDF = ".../ibt/image-builds/<slug>/tests/edf-pinned.toml" }
#
# Self-contained image model: the EDF disables the CXI hook and we launch with
# --network=disable_rdzv_get (matches the baked FI_CXI_RDZV_PROTO=alt_read).
#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=ibt-image-sanity
#SBATCH --comment=inference-benchmarking
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=ibt-image-sanity-%j.out
set -uo pipefail

: "${SANITY_EDF:?set SANITY_EDF to the digest-pinned EDF written by build.sh}"
[[ -f "${SANITY_EDF}" ]] || { echo "ERROR: SANITY_EDF not found: ${SANITY_EDF}"; exit 2; }
TESTS_DIR="$(cd "$(dirname "${SANITY_EDF}")" && pwd)"
ENV=(--environment="${SANITY_EDF}" --network=disable_rdzv_get)
cd "${TESTS_DIR}"

echo "########## 1. Component dump + intra-node NCCL (rank 0) ##########"
srun "${ENV[@]}" -N1 --ntasks-per-node=1 bash "${TESTS_DIR}/sanity.sh"

echo "########## 2. NCCL all_reduce — 8 ranks over Slingshot ##########"
srun "${ENV[@]}" --mpi=pmix -N2 --ntasks-per-node=4 \
  bash -c 'all_reduce_perf -b 8 -e 1G -f 2 -g 1' 2>&1 | tee allreduce.log

echo "########## 3. OSU osu_bw — inter-node point-to-point ##########"
srun "${ENV[@]}" --mpi=pmix -N2 --ntasks-per-node=1 \
  /usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw 2>&1 | tee osu_bw.log || true

echo "########## 4. NVSHMEM shmem_put_bw — 2 PEs over CXI ##########"
srun "${ENV[@]}" --mpi=pmix -N2 --ntasks-per-node=1 \
  /opt/nvshmem/bin/perftest/device/pt-to-pt/shmem_put_bw 2>&1 | tee nvshmem_put_bw.log || true

echo "########## 5. NVSHMEM alltoall_latency — 8 PEs ##########"
srun "${ENV[@]}" --mpi=pmix -N2 --ntasks-per-node=4 \
  /opt/nvshmem/bin/perftest/device/coll/alltoall_latency 2>&1 | tee nvshmem_alltoall.log || true

# ---- verdict: peak inter-node all_reduce busbw must clear a Slingshot floor ----
# nccl-tests data rows: ... time algbw busbw #wrong | time algbw busbw #wrong.
# Take the in-place busbw ($(NF-1)) of the last (largest-message) row.
BUSBW="$(awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]/{v=$(NF-1)} END{printf "%.1f", v+0}' allreduce.log)"
echo "[sanity] peak inter-node all_reduce busbw: ${BUSBW} GB/s"
awk -v b="${BUSBW}" 'BEGIN{
  if (b < 80) { print "[sanity] FAIL: busbw < 80 GB/s — Slingshot/aws-ofi path likely not engaged (broken ~5 GB/s?)"; exit 1 }
  print "[sanity] PASS: inter-node Slingshot bandwidth OK (target ~122 GB/s @128 MiB)"
}'
