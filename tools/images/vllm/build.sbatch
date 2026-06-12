#!/bin/bash
# Image build + JFrog push (SPECIFICATIONS.md §8.1).
#
# One batch job does pull -> build -> push: podman storage lives in
# node-local /dev/shm (§3 storage.conf pre-flight row) and is wiped when the
# job ends, so the three steps must share the allocation. Job shape mirrors
# the proven batch recipe from eth-cscs/alps-extended-images (build-vllm.sh
# in the schuups fork, branch vllm-image-test).
#
#   sbatch build.sbatch                 (from the staged context dir)
#   BUILD_CONTEXT=<dir> sbatch ...      (when submitted via FirecREST)
#
# Prerequisites (operator, once): ~/.config/containers/storage.conf per the
# §3 pre-flight row; jfrog.svc.cscs.ch credentials stored in
# ~/.config/containers/auth.json.

#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=ib-image-build-vllm
#SBATCH --comment=inference-benchmarking
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00

set -eo pipefail
BUILD_CONTEXT="${BUILD_CONTEXT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}"
cd "${BUILD_CONTEXT}"
if [ ! -f build-args.env ]; then
    echo "build-args.env not found in ${BUILD_CONTEXT} — run from the staged context dir or set BUILD_CONTEXT" >&2
    exit 1
fi
source ./build-args.env

# ---------------------------------------------------------------- auth
# First step: fail fast if the stored JFrog credentials are missing, before
# any build time is spent. auth.json is maintained by the operator (§3).
AUTHFILE=~/.config/containers/auth.json
podman login --authfile "${AUTHFILE}" --get-login jfrog.svc.cscs.ch

echo "[build] ${BASE_IMAGE} + Alps network stack -> ${IMAGE_TAG}"

podman build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -f Containerfile \
    -t "${IMAGE_TAG}" . 2>&1 | tee "${BUILD_CONTEXT}/build.log"

# ---------------------------------------------------------------- push
# Push within the same allocation: /dev/shm container storage is wiped when
# the job ends (docs.cscs.ch/build-install/containers).
podman push "docker://${IMAGE_TAG}"
echo "[build] pushed ${IMAGE_TAG}"
