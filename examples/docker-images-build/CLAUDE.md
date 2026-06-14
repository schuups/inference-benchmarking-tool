# Docker images building and publishing process

> **Superseded.** This standalone monolithic `Dockerfile` was the M5 seed. The live
> image build is now the multi-image catalogue under `tools/images/` (a shared Alps
> netstack `core/` + per-image manifests, built and pushed via `tools/images/build.sh`).
> This directory is retained only as a historical reference.

Key facts (kept for reference):
- Image builds run **within a SLURM job** (podman in node-local `/dev/shm`).
- See `tools/images/README.md` for the current build / push / sanity workflow.
