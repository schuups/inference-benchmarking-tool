#!/bin/bash
# SLURM-based image build + JFrog push (SPECIFICATIONS.md §8.1; closes the
# TODOs entry "Support building Docker images via SLURM jobs").
#
# Submit from the staged build context on the cluster:
#   sbatch [--partition=debug] build.sbatch
# Prerequisites on the cluster (operator, once):
#   - ~/.config/containers/storage.conf per the §3 pre-flight row
#   - ~/.ib_jfrog_token containing an identity token for jfrog.svc.cscs.ch
#     (created in the JFrog UI; chmod 600)

#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=ib-image-build-vllm
#SBATCH --comment=inference-benchmarking
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./build-args.env

echo "[build] alps stack ${ALPS_STACK_REPO}@${ALPS_STACK_COMMIT} on ${BASE_IMAGE}"

# ---------------------------------------------------------------- stage 1
podman build --format=docker \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg NGC_VARIANT_DIR="${NGC_VARIANT_DIR}" \
    --build-arg OCI_SOURCE="${ALPS_STACK_REPO}" \
    --build-arg OCI_REVISION="${ALPS_STACK_COMMIT}" \
    --build-arg OCI_CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --build-arg OCI_DESCRIPTION="NGC vLLM base + Alps HPC network stack" \
    --build-arg CSCS_ALPS_GIT_COMMIT_SHORT="${ALPS_STACK_COMMIT}" \
    -f Containerfile.alps-base \
    -t "${ALPS_BASE_TAG}" .

# ---------------------------------------------------------------- stage 2
podman build --format=docker \
    --build-arg BASE_IMAGE="${ALPS_BASE_TAG}" \
    --build-arg VLLM_VERSION="${VLLM_VERSION}" \
    --build-arg OCI_SOURCE="${ALPS_STACK_REPO}" \
    --build-arg OCI_REVISION="${ALPS_STACK_COMMIT}" \
    --build-arg OCI_CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --build-arg OCI_DESCRIPTION="vLLM ${VLLM_VERSION} on Alps HPC stack" \
    --build-arg CSCS_ALPS_GIT_COMMIT_SHORT="${ALPS_STACK_COMMIT}" \
    -f Containerfile \
    -t "${IMAGE_TAG}" .

# ---------------------------------------------------------------- push
podman login jfrog.svc.cscs.ch --username "${USER}" --password-stdin < ~/.ib_jfrog_token
podman push "${ALPS_BASE_TAG}"
podman push "${IMAGE_TAG}"
echo "[build] pushed ${IMAGE_TAG}"
