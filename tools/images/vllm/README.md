# vLLM engine image (§8.1)

NGC vLLM base (`build-args.env`) + the Alps HPC network stack, built one phase
per layer (see `Containerfile`) so a failed or edited phase rebuilds only from
that step onward. `build.sbatch` pulls, builds, and pushes in a single SLURM
job.

Published tag: `jfrog.svc.cscs.ch/ml/inference/vllm:26.05-py3-alps`
(see `build-args.env: IMAGE_TAG`).

Adapted from
[schuups/alps-extended-images, branch `vllm-image-test`](https://github.com/schuups/alps-extended-images/tree/vllm-image-test),
a fork of
[eth-cscs/alps-extended-images](https://github.com/eth-cscs/alps-extended-images) —
credit to the original repository and its authors.
