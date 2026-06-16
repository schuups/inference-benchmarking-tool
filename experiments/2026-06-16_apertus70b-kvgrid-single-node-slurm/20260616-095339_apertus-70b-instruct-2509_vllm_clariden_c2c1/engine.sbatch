#!/bin/bash
# Inference-deployment job — rendered by the Planner (§5); submitted by the
# Benchmarker orchestrator from within its allocation (§1, M7).
# run_id: 20260616-095339_apertus-70b-instruct-2509_vllm_clariden_c2c1

#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=ibt-engine-20260616-095339_apertus-70b-instruct-2509_vllm_clariden_c2c1
#SBATCH --comment=inference-benchmarking
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --chdir=/capstor/scratch/cscs/stefschu/ibt/20260616-095339_apertus-70b-instruct-2509_vllm_clariden_c2c1
#
# The allocation carries 4 tasks/node so the §8 pre-check step can
# run one rank per GPU (SPMD, --mpi=pmix); the engine step below uses 1 task/node.

export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)
export HF_HOME=/capstor/scratch/cscs/stefschu/ibt/hf-cache

# §8 pre-check surface (graded by tools/benchmarker/prechecks/grade.py)
export PRECHECK_CLUSTER=clariden
export PRECHECK_SCOPE="4× GH200, 1 node"
export PRECHECK_STORAGE_SCOPE="capstor weights mount (Lustre, HDD)"   # fallback; run-storage.sh derives the real mount
export MODEL_ID="swiss-ai/Apertus-70B-Instruct-2509"                              # §8.1 storage check reads THIS model's shards
export PRECHECK_OUT=/capstor/scratch/cscs/stefschu/ibt/20260616-095339_apertus-70b-instruct-2509_vllm_clariden_c2c1/prechecks
export PRECHECK_ON_WARN=continue
export PRECHECK_ON_FAIL=abort
export SKIP_SYSTEM_PRECHECKS=0
export NCCL_TESTS_VERSION=2.18.2
export NCCL_TESTS_CACHE=/capstor/scratch/cscs/stefschu/ibt/collective-tests-cache
export NVSHMEM_REQUIRED=0
export MSG_END=128M
# The pre-check runs as a dedicated step with one rank per GPU, so the collective
# build is always MPI (each rank drives a single GPU via -g 1); §8.2.
export NCCL_TESTS_MPI=1
export PRECHECK_GPUS=1
export PRECHECK_COLLECTIVES="all_reduce all_gather alltoall"





# ---- §8 pre-checks: dedicated step, one rank per GPU (SPMD via PMIx) — gates the engine ----
# Same EDF (image / libfabric / CUDA / NCCL / mounts / --network / --mpi) as the engine, so the
# foundation measured is the engine's; the per-GPU rank topology is the engine's actual one (§8.2).
# --ntasks is set EXPLICITLY (one rank per GPU) — the allocation's SLURM_NTASKS would otherwise
# leak into the step and the engine step below must not inherit it (see its --ntasks).
srun --environment=/capstor/scratch/cscs/stefschu/ibt/20260616-095339_apertus-70b-instruct-2509_vllm_clariden_c2c1/engine.toml --network=disable_rdzv_get --nodes=1 --ntasks=4 --ntasks-per-node=4 --mpi=pmix \
    bash /capstor/scratch/cscs/stefschu/ibt/20260616-095339_apertus-70b-instruct-2509_vllm_clariden_c2c1/tools/benchmarker/prechecks/run_system_prechecks.sh
precheck_rc=$?
if [ "${precheck_rc}" -ne 0 ]; then
    echo "[engine.sbatch] §8 pre-checks gated the run (exit ${precheck_rc}); not starting the engine" >&2
    exit "${precheck_rc}"
fi


# ---- engine step: one task PER NODE (vLLM drives its GPUs in-process / via Ray) ----
# --ntasks=1 is EXPLICIT: the allocation is sized 4 tasks/node for the
# pre-check step, so without this the engine step would inherit SLURM_NTASKS and launch one vLLM
# per GPU (4 servers contending for memory + clobbered hw-sampler). §13.3 sampler backgrounded
# in the same container session.
srun --environment=/capstor/scratch/cscs/stefschu/ibt/20260616-095339_apertus-70b-instruct-2509_vllm_clariden_c2c1/engine.toml --network=disable_rdzv_get --nodes=1 --ntasks=1 --ntasks-per-node=1 --mpi=pmix bash -c '
set -eo pipefail
python3 /capstor/scratch/cscs/stefschu/ibt/20260616-095339_apertus-70b-instruct-2509_vllm_clariden_c2c1/tools/benchmarker/hw_sampler.py \
    --out /capstor/scratch/cscs/stefschu/ibt/20260616-095339_apertus-70b-instruct-2509_vllm_clariden_c2c1/hw-$(hostname).ndjson --interval 1.0 &

exec vllm serve swiss-ai/Apertus-70B-Instruct-2509 --host 0.0.0.0 --port 8000 --served-model-name swiss-ai/Apertus-70B-Instruct-2509 --tensor-parallel-size 4 --max-model-len 262144 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8 --enable-prefix-caching --safetensors-load-strategy prefetch

'
