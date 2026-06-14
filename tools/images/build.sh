#!/usr/bin/env bash
# Build + push an Alps engine image from its manifest (SPECIFICATIONS.md §8.1).
#
# Image builds are expensive and run in node-local /dev/shm (podman storage), so
# the layer cache is RAM-resident and lost when the node is released. FirecREST
# does not expose `srun`, so this drives podman over SSH into a *held* allocation
# (salloc, or a sleep holder job) whose JOBID you pass. Per-phase Containerfile
# layering means a failed/edited phase rebuilds only from that step down — keep
# the same allocation to reuse the warm cache across iterations.
#
# Usage:
#   tools/images/build.sh <image-slug> --jobid <HOLDER_JOBID> [--no-push]
#
# Env overrides: IB_CLUSTER (default clariden), IB_REMOTE_BASE
#   (default /capstor/scratch/cscs/$USER/ibt/image-builds, $USER expanded remotely).
#
# Prereqs on the cluster (operator, once): ~/.config/containers/{storage.conf,
# auth.json} (§3). See the project_image_build_env memory for the full recipe.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO}/.venv/bin/python"
CLUSTER="${IB_CLUSTER:-clariden}"
REMOTE_BASE_TMPL='${IB_REMOTE_BASE:-/capstor/scratch/cscs/$USER/ibt/image-builds}'

SLUG="" ; JOBID="" ; DO_PUSH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobid)   JOBID="$2"; shift 2 ;;
    --no-push) DO_PUSH=0; shift ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    -*)        echo "ERROR: unknown flag $1" >&2; exit 2 ;;
    *)         SLUG="$1"; shift ;;
  esac
done
[[ -n "$SLUG"  ]] || { echo "ERROR: image slug required (e.g. nvidia-vllm-0.22.1-net.v1)" >&2; exit 2; }
[[ -n "$JOBID" ]] || { echo "ERROR: --jobid <held allocation jobid> required" >&2; exit 2; }

IMG_DIR="${REPO}/tools/images/${SLUG}"
MANIFEST="${IMG_DIR}/manifest.yaml"
[[ -f "$MANIFEST" ]] || { echo "ERROR: no manifest at ${MANIFEST}" >&2; exit 1; }

remote() { ssh -o BatchMode=yes "$CLUSTER" "$@"; }

# ---- read identity + per-pin --build-arg list from the manifest ----------------
read -r BASE_IMAGE IMAGE_TAG NETSTACK <<<"$("$PY" - "$MANIFEST" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
print(m["base_image"], m["image_tag"], m["netstack"])
PY
)"
BUILDARGS="$("$PY" - "$MANIFEST" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
pins = m.get("netstack_pins", {}) or {}
# manifest pin key -> Containerfile ARG name(s). cxi_shs drives the three SHS pins.
ARGS = {
  "xpmem_ref": ["XPMEM_REF"], "gdrcopy_ver": ["GDRCOPY_VER"],
  "cxi_shs": ["CASSINI_HEADERS_VERSION", "CXI_DRIVER_VERSION", "LIBCXI_VERSION"],
  "libfabric_commit": ["LIBFABRIC_COMMIT"], "libfabric_patch": ["LIBFABRIC_PATCH"],
  "nccl_ver": ["NCCL_VER"], "nccl_patch": ["NCCL_PATCH"],
  "ucx_version": ["UCX_VERSION"], "ucc_version": ["UCC_VERSION"], "ompi_ver": ["OMPI_VER"],
  "aws_ofi_nccl_commit": ["AWS_OFI_NCCL_COMMIT"], "aws_ofi_nccl_patch": ["AWS_OFI_NCCL_PATCH"],
  "nvshmem_ver": ["NVSHMEM_VER"], "nvshmem_patch": ["NVSHMEM_PATCH"],
  "nccl_tests_ver": ["NCCL_TESTS_VER"], "osu_version": ["OSU_VERSION"],
}
out = []
for k, v in pins.items():
    for arg in ARGS.get(k, []):
        out.append(f"--build-arg {arg}={v}")
print(" ".join(out))
PY
)"

VENDOR="${NETSTACK%%/*}"; NSVER="${NETSTACK##*/}"
NS_DIR="${REPO}/tools/images/core/${VENDOR}/netstack/${NSVER}"
[[ -d "$NS_DIR" ]] || { echo "ERROR: netstack dir not found: ${NS_DIR}" >&2; exit 1; }

# Resolve the remote base (expand $USER on the cluster) and per-image paths.
REMOTE_BASE="$(remote "echo ${REMOTE_BASE_TMPL}")"
CTX="${REMOTE_BASE}/${SLUG}/ctx"
TESTS="${REMOTE_BASE}/${SLUG}/tests"

echo "[build.sh] slug=${SLUG}"
echo "[build.sh] base=${BASE_IMAGE}"
echo "[build.sh] tag =${IMAGE_TAG}"
echo "[build.sh] netstack=${NETSTACK} (${NS_DIR})"
echo "[build.sh] ctx =${CLUSTER}:${CTX}"

# ---- stage the composed build context -----------------------------------------
remote "mkdir -p ${CTX} ${TESTS}"
rsync -a          "${NS_DIR}/Containerfile"         "${CLUSTER}:${CTX}/Containerfile"
rsync -a --delete "${REPO}/tools/images/core/common/" "${CLUSTER}:${CTX}/common/"
rsync -a --delete "${NS_DIR}/runtime/"              "${CLUSTER}:${CTX}/runtime/"
rsync -a --delete "${NS_DIR}/phases/"               "${CLUSTER}:${CTX}/phases/"
rsync -a --delete "${NS_DIR}/patches/"              "${CLUSTER}:${CTX}/patches/"
rsync -a --delete "${IMG_DIR}/variant/"             "${CLUSTER}:${CTX}/variant/"
# stage the per-image in-image sanity script alongside (used by sanity.sbatch)
[[ -f "${IMG_DIR}/tests/sanity.sh" ]] && rsync -a "${IMG_DIR}/tests/sanity.sh" "${CLUSTER}:${TESTS}/sanity.sh"

# ---- build via srun --overlap into the held allocation ------------------------
echo "[build.sh] building (job ${JOBID}) — failed/edited phases rebuild from that layer down ..."
remote "srun --jobid=${JOBID} --overlap bash -c '
  cd ${CTX} &&
  podman build --build-arg BASE_IMAGE=${BASE_IMAGE} ${BUILDARGS} -f Containerfile -t ${IMAGE_TAG} . 2>&1 | tee build.log
  exit \${PIPESTATUS[0]}'"
echo "[build.sh] build OK"

# ---- push + emit a digest-pinned EDF for sanity -------------------------------
if [[ "$DO_PUSH" == "1" ]]; then
  echo "[build.sh] pushing ${IMAGE_TAG} ..."
  # host/repo without tag, first '/' -> '#' for the CE EDF image ref.
  HOSTREPO="${IMAGE_TAG%:*}"; EDF_HOSTREPO="${HOSTREPO/\//#}"
  remote "srun --jobid=${JOBID} --overlap bash -c '
    podman push --digestfile ${CTX}/pushed-digest.txt ${IMAGE_TAG} 2>&1 | tail -3
    D=\$(cat ${CTX}/pushed-digest.txt)
    echo \"[build.sh] registry digest: \${D}\"
    cat > ${TESTS}/edf-pinned.toml <<EOF
image = \"${EDF_HOSTREPO}@\${D}\"
mounts = [\"/users\", \"/capstor\", \"/iopsstor\"]

[annotations]
com.hooks.cxi.enabled = \"false\"
EOF
    echo \"[build.sh] wrote ${TESTS}/edf-pinned.toml\"'"
  echo "[build.sh] pushed. Run sanity with:"
  echo "  SANITY_EDF=${TESTS}/edf-pinned.toml  (submit tools/images/sanity.sbatch via FirecREST)"
fi
