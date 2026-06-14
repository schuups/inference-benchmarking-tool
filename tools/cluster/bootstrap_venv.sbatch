#!/bin/bash
# One-time cluster-side Python env for the Benchmarker (IMPLEMENTATION_PLAN.md
# M7/E1): the benchmarker.sbatch runs tools/benchmarker/main.py on the HOST
# (no container), which needs Python >= 3.10 + tools/requirements.txt.
#
# Creates /capstor/scratch/cscs/$USER/ib/venv. Re-run to refresh deps.
# Submit with: sbatch bootstrap_venv.sbatch   (requirements.txt staged alongside)

#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=ib-bootstrap-venv
#SBATCH --comment=inference-benchmarking
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

set -eo pipefail
VENV=/capstor/scratch/cscs/${USER}/ib/venv
PYPI=https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple
# FirecREST invokes sbatch from its own cwd — pass BOOTSTRAP_DIR via the API.
HERE="${BOOTSTRAP_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
REQS="${HERE}/requirements.txt"
[ -f "${REQS}" ] || { echo "requirements.txt not found at ${REQS}" >&2; exit 1; }

PY=""
for candidate in python3.12 python3.11 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1; then PY="${candidate}"; break; fi
done

if [ -n "${PY}" ]; then
    echo "[bootstrap] using host ${PY} ($(${PY} --version))"
    "${PY}" -m venv --clear "${VENV}"
    "${VENV}/bin/pip" install --quiet --index-url "${PYPI}" --upgrade pip
    "${VENV}/bin/pip" install --quiet --index-url "${PYPI}" -r "${REQS}" transformers
else
    echo "[bootstrap] no suitable host python; installing uv-managed 3.12 (aarch64)"
    export UV_INSTALL_DIR=/capstor/scratch/cscs/${USER}/uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${UV_INSTALL_DIR}:${PATH}"
    uv python install 3.12
    uv venv --clear --python 3.12 "${VENV}"
    uv pip install --python "${VENV}/bin/python" --index-url "${PYPI}" -r "${REQS}" transformers
fi

"${VENV}/bin/python" - <<'EOF'
import sys
import aiohttp, pydantic, transformers, yaml
print(f"[bootstrap] venv OK: python {sys.version.split()[0]}, "
      f"pydantic {pydantic.VERSION}, aiohttp {aiohttp.__version__}, "
      f"transformers {transformers.__version__}")
EOF
