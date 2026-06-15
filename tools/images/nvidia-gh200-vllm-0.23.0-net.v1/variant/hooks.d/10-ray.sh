#!/usr/bin/env bash
# Per-image variant hook: bake Ray into the image so multi-node vLLM
# (`--distributed-executor-backend ray`) can be spawned WITHOUT a per-experiment
# manual `pip install ray`. vLLM itself comes from the BASE IMAGE (0.23, not pip);
# we pin it here only so pip won't modify/reinstall it, and add ray[default]
# (dashboard + cluster/launcher utils) at a version vLLM's [ray] extra is
# compatible with. Uses the JFrog PyPI mirror (matches the netstack phases).
set -euo pipefail

vllm_ver="$(python3 -c 'import vllm; print(vllm.__version__)')"
echo "[ray-hook] base vllm=${vllm_ver}; installing ray[default] (vLLM-compatible)"

python3 -m pip install --no-cache-dir \
  -i https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple \
  "ray[default]" "vllm[ray]==${vllm_ver}"

# Fail the build loudly if Ray didn't land (the whole point of this image).
python3 -c 'import ray; print("[ray-hook] ray " + ray.__version__ + " OK")'
