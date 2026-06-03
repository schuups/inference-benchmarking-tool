# Deploy Apertus8b with vLLM on SLURM

```sh
[clariden][stefschu@clariden-ln003 example-slurm-deployment]$ pwd
/capstor/scratch/cscs/stefschu/example-slurm-deployment

[clariden][stefschu@clariden-ln003 example-slurm-deployment]$ sbatch apertus8b-vllm.sbatch 
Submitted batch job 2033026

[clariden][stefschu@clariden-ln003 example-slurm-deployment]$ squeue -j 2033026 -h -o "%N"
nid007660
```

On my laptop I setup an ssh tunnel:

```sh
ssh -N -L 8080:nid007660:8080 clariden.cscs.ch
```

Only then I can query the endpoint from my own laptop:

```sh
user@MacBookAir ~ % curl http://localhost:8080/v1/models
{"object":"list","data":[{"id":"swiss-ai/Apertus-8B-Instruct-2509","object":"model","created":1777903758,"owned_by":"vllm","root":"swiss-ai/Apertus-8B-Instruct-2509","parent":null,"max_model_len":65536,"permission":[{"id":"modelperm-97715325774948e5870d47f15f717fbb","object":"model_permission","created":1777903758,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```



