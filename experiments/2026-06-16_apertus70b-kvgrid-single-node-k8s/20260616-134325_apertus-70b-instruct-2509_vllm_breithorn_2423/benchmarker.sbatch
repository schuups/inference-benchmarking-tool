#!/bin/bash
# Benchmarker allocation — rendered by the Planner (§5). One per experiment;
# sequences dataset gen -> engine spawn -> load gen -> finalisation (§1, M7).
# run_id: 20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423

#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=ibt-benchmarker-20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423
#SBATCH --comment=inference-benchmarking
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/capstor/scratch/cscs/stefschu/ibt/20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423

# Same time limit as the engine job (§6.1 time-limit alignment).
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)
export HF_HOME=/capstor/scratch/cscs/stefschu/ibt/hf-cache

# Runtime (decision 3): deps from a staged uv venv on capstor, built once; the
# live tools/ code is mounted from capstor at /capstor/scratch/cscs/stefschu/ibt/20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423/tools (staged by the
# Coordinator) and put on PYTHONPATH, so code iterates without rebuilding the venv.
VENV="/capstor/scratch/cscs/stefschu/ibt/benchmarker-venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "[benchmarker] building staged venv at $VENV"
    uv venv --python 3.12 "$VENV"   # pin a modern Python (bare-node python3 is ancient 3.6)
    uv pip install --python "$VENV/bin/python" -r "/capstor/scratch/cscs/stefschu/ibt/20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423/tools/benchmarker/requirements.txt"
fi
export PYTHONPATH="/capstor/scratch/cscs/stefschu/ibt/20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423:${PYTHONPATH:-}"   # so `import tools` resolves

exec "$VENV/bin/python" -m tools.benchmarker.main \
    --config /capstor/scratch/cscs/stefschu/ibt/20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423/benchmark_config.yaml \
    --run-id 20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423 \
    --run-dir /capstor/scratch/cscs/stefschu/ibt/20260616-134325_apertus-70b-instruct-2509_vllm_breithorn_2423 \
    --endpoint-url https://ibt-engine-20260616-134325-apertus-70b-instruct-2509-vl-9d893c.breithorn.svc.cscs.ch \
    --deployment-index 0
