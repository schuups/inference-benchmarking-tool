"""M6 DoD (local half): render invariants for SLURM + K8s artifacts.

sbatch --test-only on clariden and kubectl --dry-run=server on breithorn are
the cluster half, validated at E1/E5.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from tools.common.config import BenchmarkConfig, Deployment, load_global_config
from tools.planner.render import (
    precheck_scope,
    render_experiment,
    total_gpus,
    vllm_command,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = REPO_ROOT / "examples" / "benchmark-configs" / "mixed-80-20.yaml"
NOW = datetime(2026, 6, 12, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def globals_cfg():
    return load_global_config()


def _render(tmp_path, cfg_dict, globals_cfg):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg_dict))
    exp_dir = render_experiment(path, tmp_path / "experiments", globals_cfg, now=NOW)
    run_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    return exp_dir, run_dirs


@pytest.fixture()
def canonical_dict():
    with open(CANONICAL) as f:
        return yaml.safe_load(f)


def test_vllm_command_flag_mapping():
    dep = Deployment.model_validate(
        {
            "target": "clariden", "backend": "vllm", "backend_version": "x",
            "model": "org/m",
            "backend_config": {
                "tensor_parallel_size": 4, "pipeline_parallel_size": 2,
                "max_model_len": 65536, "max_num_batched_tokens": 65536,
                "gpu_memory_utilization": 0.9, "kv_cache_dtype": "fp8",
                "enable_prefix_caching": False, "safetensors_load_strategy": "prefetch",
                "kv_offloading_size": 400,
                "speculative_decoding": {"draft_model": "org/draft", "num_speculative_tokens": 5},
            },
        }
    )
    cmd = vllm_command(dep)
    for expected in (
        "vllm serve org/m", "--tensor-parallel-size 4", "--pipeline-parallel-size 2",
        "--max-model-len 65536", "--max-num-batched-tokens 65536",
        "--gpu-memory-utilization 0.9", "--kv-cache-dtype fp8",
        "--no-enable-prefix-caching", "--safetensors-load-strategy prefetch",
        "--kv-offloading-size 400", '--speculative-config', '"num_speculative_tokens": 5',
    ):
        assert expected in cmd, expected
    assert total_gpus(dep) == 8


def test_precheck_scope_matches_reference_strings(globals_cfg):
    dep1 = Deployment(target="clariden", backend="vllm", backend_version="x", model="m")
    dep1.backend_config.tensor_parallel_size = 4
    assert precheck_scope(dep1, globals_cfg) == "4× GH200, 1 node"
    dep1.backend_config.pipeline_parallel_size = 2
    assert precheck_scope(dep1, globals_cfg) == "8× GH200, 2 nodes"
    dep2 = Deployment(target="breithorn", backend="vllm", backend_version="x", model="m")
    dep2.backend_config.tensor_parallel_size = 4
    assert precheck_scope(dep2, globals_cfg) == "4× gh200, 1 pod"


def test_slurm_render_canonical(tmp_path, canonical_dict, globals_cfg):
    exp_dir, run_dirs = _render(tmp_path, canonical_dict, globals_cfg)
    assert (exp_dir / "benchmark_config.yaml").exists()  # §13.8 provenance copy
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    engine = (run_dir / "engine.sbatch").read_text()
    benchmarker = (run_dir / "benchmarker.sbatch").read_text()
    edf = (run_dir / "engine.toml").read_text()

    # §5.1: same account/partition/time limit on every job of the experiment
    for content in (engine, benchmarker):
        assert "#SBATCH --account=csstaff" in content
        assert "#SBATCH --partition=normal" in content
        assert "#SBATCH --time=04:00:00" in content
        assert "#SBATCH --comment=inference-benchmarking" in content  # §6.1

    # M7: the orchestrator is told which deployment it is driving (§15 sweep)
    assert "--deployment-index 0" in benchmarker

    # decision 3: deps from a staged uv venv on capstor; live tools/ on PYTHONPATH
    assert "uv venv" in benchmarker
    assert "benchmarker/requirements.txt" in benchmarker
    assert '"$VENV/bin/python" -m tools.benchmarker.main' in benchmarker
    assert "PYTHONPATH=" in benchmarker

    # §7.2 concatenation + M3 sampler backgrounding in the same container session
    assert "run_system_prechecks.sh &&" in engine
    assert "hw_sampler.py" in engine and "& " not in engine.split("hw_sampler.py")[0].splitlines()[-1]
    assert 'PRECHECK_SCOPE="16× GH200, 4 nodes"' in engine  # TP4 x PP4 canonical
    assert "#SBATCH --nodes=4" in engine
    assert "--distributed-executor-backend ray" in engine  # multi-node path
    # engine flags rendered from BackendConfig
    assert "--tensor-parallel-size 4" in engine
    assert "--pipeline-parallel-size 4" in engine
    assert "--safetensors-load-strategy prefetch" in engine
    # EDF
    assert 'workdir = ' in edf and "hf-cache" in edf


def test_slurm_single_node_has_no_ray(tmp_path, canonical_dict, globals_cfg):
    canonical_dict["deployments"][0]["backend_config"]["pipeline_parallel_size"] = 1
    _, run_dirs = _render(tmp_path, canonical_dict, globals_cfg)
    engine = (run_dirs[0] / "engine.sbatch").read_text()
    assert "#SBATCH --nodes=1" in engine
    assert "ray start" not in engine
    assert 'PRECHECK_SCOPE="4× GH200, 1 node"' in engine


def test_k8s_render(tmp_path, canonical_dict, globals_cfg):
    canonical_dict["deployments"][0]["target"] = "breithorn"
    canonical_dict["deployments"][0]["backend_config"]["pipeline_parallel_size"] = 1
    _, run_dirs = _render(tmp_path, canonical_dict, globals_cfg)
    run_dir = run_dirs[0]
    engine_yaml = (run_dir / "engine.yaml").read_text()
    pod_yaml = (run_dir / "benchmarker-pod.yaml").read_text()

    docs = list(yaml.safe_load_all(engine_yaml))
    assert [d["kind"] for d in docs] == ["Deployment", "Service"]
    deployment = docs[0]
    labels = deployment["metadata"]["labels"]
    assert labels["app.kubernetes.io/managed-by"] == "inference-benchmarking"  # §6.1
    pod_spec = deployment["spec"]["template"]["spec"]
    assert pod_spec["nodeSelector"]["beta.kubernetes.io/instance-type"] == "gh200"
    container = pod_spec["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 4
    assert "run_system_prechecks.sh" in container["args"][0]  # §7.2 in-pod gate
    assert "hw_sampler.py" in container["args"][0]
    assert any(v["persistentVolumeClaim"]["claimName"].startswith("model-cache-")
               for v in pod_spec["volumes"])  # §6.6 retained PVC

    pod = yaml.safe_load(pod_yaml)
    assert pod["kind"] == "Pod"
    assert pod["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "inference-benchmarking"


def test_explicit_image_respected(tmp_path, canonical_dict, globals_cfg):
    canonical_dict["deployments"][0]["image"] = "jfrog.svc.cscs.ch/ml/inference/vllm:pinned-tag"
    _, run_dirs = _render(tmp_path, canonical_dict, globals_cfg)
    edf = (run_dirs[0] / "engine.toml").read_text()
    assert 'image = "jfrog.svc.cscs.ch/ml/inference/vllm:pinned-tag"' in edf


def test_renders_one_run_dir_per_deployment(tmp_path, canonical_dict, globals_cfg):
    second = dict(canonical_dict["deployments"][0])
    second["backend_config"] = dict(second["backend_config"], kv_cache_dtype="fp8")
    canonical_dict["deployments"].append(second)
    _, run_dirs = _render(tmp_path, canonical_dict, globals_cfg)
    assert len(run_dirs) == 2  # one engine launch == one run_id (§15)
    contents = [(d / "engine.sbatch").read_text() for d in run_dirs]
    assert sum("--kv-cache-dtype fp8" in c for c in contents) == 1
    # each benchmarker carries a distinct --deployment-index (run dirs differ only
    # by the random run-id suffix, so assert the set rather than positional order)
    bench = [(d / "benchmarker.sbatch").read_text() for d in run_dirs]
    idxs = {re.search(r"--deployment-index (\d+)", t).group(1) for t in bench}
    assert idxs == {"0", "1"}
