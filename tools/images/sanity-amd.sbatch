#!/bin/bash
# Post-push image sanity, AMD/MI300A variant (SPECIFICATIONS.md §9.1 / §8): a
# short-lived 2-node job on beverin's mi300 partition that validates the baked
# Alps network stack loads and achieves Slingshot bandwidth inter-node — RCCL
# (+ aws-ofi-rccl/CXI), OSU, and rocSHMEM (best-effort), no errors.
#
# Submit via FirecREST (HPC Platform) AFTER build.sh has pushed, passing the
# digest-pinned EDF it wrote (env var SANITY_EDF). Pinning by registry digest
# sidesteps the stale enroot tag cache across rebuilds. Example:
#   env_vars = { SANITY_EDF = ".../ibt/image-builds/<slug>/tests/edf-pinned.toml" }
#
# Self-contained image model: the EDF disables the CXI hook and we launch with
# --network=disable_rdzv_get (matches the baked FI_CXI_RDZV_PROTO=alt_read).
#SBATCH --account=a-csstaff
#SBATCH --partition=mi300
#SBATCH --job-name=ibt-image-sanity-amd
#SBATCH --comment=inference-benchmarking
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=ibt-image-sanity-amd-%j.out
set -uo pipefail

: "${SANITY_EDF:?set SANITY_EDF to the digest-pinned EDF written by build.sh}"
[[ -f "${SANITY_EDF}" ]] || { echo "ERROR: SANITY_EDF not found: ${SANITY_EDF}"; exit 2; }
TESTS_DIR="$(cd "$(dirname "${SANITY_EDF}")" && pwd)"
ENV=(--environment="${SANITY_EDF}" --network=disable_rdzv_get)
cd "${TESTS_DIR}"

echo "########## 1. Component dump + intra-node RCCL (rank 0) ##########"
srun "${ENV[@]}" -N1 --ntasks-per-node=1 bash "${TESTS_DIR}/sanity.sh"

echo "########## 2. RCCL all_reduce — 8 ranks over Slingshot ##########"
srun "${ENV[@]}" --mpi=pmix -N2 --ntasks-per-node=4 \
  bash -c 'all_reduce_perf -b 8 -e 1G -f 2 -g 1' 2>&1 | tee allreduce.log

echo "########## 3. OSU osu_bw — inter-node point-to-point ##########"
srun "${ENV[@]}" --mpi=pmix -N2 --ntasks-per-node=1 \
  /usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw 2>&1 | tee osu_bw.log || true

echo "########## 4. rocSHMEM put bandwidth — best-effort (v1 may lack it) ##########"
srun "${ENV[@]}" --mpi=pmix -N2 --ntasks-per-node=1 \
  bash -lc 'p="$(command -v shmem_put_bw rocshmem_put_bw 2>/dev/null | head -1)"; \
            [ -n "$p" ] && "$p" || echo "[sanity] rocSHMEM put benchmark not found — skip (collectives-first)"' \
  2>&1 | tee rocshmem.log || true

# ---- verdict: peak inter-node all_reduce busbw must clear a Slingshot floor ----
# rccl-tests data rows match nccl-tests columns: ... time algbw busbw #wrong | ...
# Take the in-place busbw ($(NF-1)) of the last (largest-message) row.
# Floor defaults to 80 GB/s (catches the "plugin didn't fire" ~5 GB/s failure);
# the MI300A Slingshot reference (spec §8.3) is established by this run — override
# via SANITY_BUSBW_FLOOR once characterised.
BUSBW="$(awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]/{v=$(NF-1)} END{printf "%.1f", v+0}' allreduce.log)"
echo "[sanity] peak inter-node all_reduce busbw: ${BUSBW} GB/s"
awk -v b="${BUSBW}" -v floor="${SANITY_BUSBW_FLOOR:-80}" 'BEGIN{
  if (b < floor) { printf "[sanity] FAIL: busbw %.1f < %.1f GB/s — Slingshot/aws-ofi-rccl path likely not engaged\n", b, floor; exit 1 }
  printf "[sanity] PASS: inter-node Slingshot bandwidth OK (busbw %.1f GB/s, floor %.1f)\n", b, floor
}'
