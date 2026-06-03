#!/bin/bash

#SBATCH --account=csstaff
#SBATCH --partition=debug
#SBATCH --job-name=infra-benchmark
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --chdir=/capstor/scratch/cscs/stefschu/example-slurm-deployment

export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token)
export CUDA_VISIBLE_DEVICES=0

export HF_HOME=/capstor/scratch/cscs/stefschu/example-slurm-deployment/hf-cache

srun --environment=$(pwd)/env.toml \
  python3 -m vllm.entrypoints.openai.api_server \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --host 0.0.0.0 \
    --port 8080 \
    --tensor-parallel-size 1 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509
