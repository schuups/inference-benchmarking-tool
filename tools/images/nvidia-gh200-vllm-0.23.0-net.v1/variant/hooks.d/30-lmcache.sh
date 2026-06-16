#!/usr/bin/env bash
# Per-image variant hook: bake LMCache into the image — the KV-cache layer used
# for CPU/disk offloading, cross-instance KV sharing, and (with NIXL as the
# transport) prefill/decode (P/D) disaggregation. vLLM 0.23 ships the LMCache KV
# connector upstream (LMCacheConnectorV1 / kv_connector_module_path); this hook
# provides the `lmcache` runtime that connector imports.
#
# LMCache publishes x86_64-only wheels on PyPI, but on aarch64 the resolver lands
# the newest installable artifact via the JFrog mirror: lmcache 0.4.6. Its
# compiled c_ops backend loads on GH200 and the install does NOT perturb the base
# torch (2.11.0+cu130) / vLLM (0.23.0) — verified on clariden 2026-06-16. Pinned
# for reproducibility; uses the JFrog PyPI mirror. Runs AFTER 20-nixl.sh so the
# NIXL transport is already present.
set -euo pipefail

vllm_ver="$(python3 -c 'import vllm; print(vllm.__version__)')"
echo "[lmcache-hook] base vllm=${vllm_ver}; installing lmcache==0.4.6"
python3 -m pip install --no-cache-dir \
  -i https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple \
  "lmcache==0.4.6"

# Fail the build loudly if LMCache didn't land or its native backend can't load.
python3 -c 'import lmcache; print("[lmcache-hook] lmcache " + lmcache.__version__ + " OK")'
