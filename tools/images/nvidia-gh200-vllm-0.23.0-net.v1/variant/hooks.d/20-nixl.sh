#!/usr/bin/env bash
# Per-image variant hook: bake NIXL (NVIDIA Inference Xfer Library) into the image
# for KV-cache transfer in prefill/decode (P/D) disaggregation (vLLM's
# NixlConnector, and LMCache's NIXL transport — see 30-lmcache.sh).
#
# NIXL ships as a py3-none-any META package that pulls in BOTH the CUDA-12 and
# CUDA-13 backends (nixl-cu12 / nixl-cu13) and selects the right one at runtime
# from the CUDA version torch reports (-> cu13 on this base). aarch64 backend
# wheels exist, so no source build is needed (verified on clariden 2026-06-16:
# the JFrog PyPI mirror resolves nixl 1.2.0 + both backends, and nixl_agent
# imports). Pinned for reproducibility; uses the JFrog PyPI mirror (matches the
# netstack phases + the ray hook). Runs AFTER 10-ray.sh.
#
# NOTE: PyPI has a newer nixl 1.3.0 (2026-06-15) not yet on the JFrog mirror; bump
# this pin once the mirror has synced if a newer NIXL is wanted.
set -euo pipefail

echo "[nixl-hook] installing nixl==1.2.0 (meta package: cu12+cu13 backends, runtime-selected)"
python3 -m pip install --no-cache-dir \
  -i https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple \
  "nixl==1.2.0"

# Fail the build loudly if NIXL didn't land or its agent can't import (the whole
# point of this image). nixl_agent pulls in the selected native backend.
python3 -c 'from nixl._api import nixl_agent; print("[nixl-hook] nixl_agent import OK")'
