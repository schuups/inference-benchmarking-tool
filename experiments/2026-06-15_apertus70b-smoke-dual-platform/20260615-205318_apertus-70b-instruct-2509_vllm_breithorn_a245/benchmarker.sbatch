#!/bin/bash
# Benchmarker for the K8s (breithorn) engine — hand-constructed (the planner emits no
# benchmarker.sbatch for a K8s target; that wiring is the E5 deliverable). Runs on clariden
# (the Benchmarker is ALWAYS SLURM, §2) and load-gens against the already-kubectl-deployed
# breithorn engine via its Ingress (ExternalEndpointLauncher / --endpoint-url, §6.2). No
# engine spawn here. run_id: 20260615-205318_apertus-70b-instruct-2509_vllm_breithorn_a245

#SBATCH --account=csstaff
#SBATCH --partition=normal
#SBATCH --job-name=ibt-benchmarker-20260615-205318_apertus-70b-instruct-2509_vllm_breithorn_a245
#SBATCH --comment=inference-benchmarking
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --chdir=/capstor/scratch/cscs/stefschu/ibt/20260615-205318_apertus-70b-instruct-2509_vllm_breithorn_a245

export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)
export HF_HOME=/capstor/scratch/cscs/stefschu/ibt/hf-cache

VENV="/capstor/scratch/cscs/stefschu/ibt/benchmarker-venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "[benchmarker] building staged venv at $VENV"
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" -r "/capstor/scratch/cscs/stefschu/ibt/20260615-205318_apertus-70b-instruct-2509_vllm_breithorn_a245/tools/benchmarker/requirements.txt"
fi
export PYTHONPATH="/capstor/scratch/cscs/stefschu/ibt/20260615-205318_apertus-70b-instruct-2509_vllm_breithorn_a245:${PYTHONPATH:-}"

exec "$VENV/bin/python" -m tools.benchmarker.main \
    --config /capstor/scratch/cscs/stefschu/ibt/20260615-205318_apertus-70b-instruct-2509_vllm_breithorn_a245/benchmark_config.yaml \
    --run-id 20260615-205318_apertus-70b-instruct-2509_vllm_breithorn_a245 \
    --run-dir /capstor/scratch/cscs/stefschu/ibt/20260615-205318_apertus-70b-instruct-2509_vllm_breithorn_a245 \
    --endpoint-url https://ibt-engine-20260615-205318-apertus-70b-instruct-2509-vl-55aeb5.breithorn.svc.cscs.ch \
    --deployment-index 1
