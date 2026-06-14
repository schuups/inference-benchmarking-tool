"""Planner (SPECIFICATIONS.md §4): benchmark YAML -> §13.8 experiment directory.

Pure laptop-side rendering against the Jinja2 templates in tools/templates/;
nothing is submitted. Every artifact lands in the experiment directory, so a
re-render is fully reproducible from the benchmark YAML alone.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.common.config import (
    Deployment,
    GlobalConfig,
    load_benchmark_config,
    load_global_config,
)
from tools.common.runid import make_run_id, model_slug, run_id_slug

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


def vllm_command(deployment: Deployment) -> str:
    """BackendConfig -> vLLM launch command (§15.2 flag mapping)."""
    bc = deployment.backend_config
    parts = [
        "vllm serve", deployment.model,
        "--host 0.0.0.0", "--port 8000",
        f"--served-model-name {deployment.model}",
        f"--tensor-parallel-size {bc.tensor_parallel_size}",
    ]
    if bc.pipeline_parallel_size > 1:
        parts.append(f"--pipeline-parallel-size {bc.pipeline_parallel_size}")
    if bc.data_parallel_size > 1:
        parts.append(f"--data-parallel-size {bc.data_parallel_size}")
    if bc.expert_parallel_size > 1:
        parts.append(f"--expert-parallel-size {bc.expert_parallel_size}")
    if bc.max_model_len is not None:
        parts.append(f"--max-model-len {bc.max_model_len}")
    if bc.max_num_batched_tokens is not None:
        parts.append(f"--max-num-batched-tokens {bc.max_num_batched_tokens}")
    if bc.gpu_memory_utilization is not None:
        parts.append(f"--gpu-memory-utilization {bc.gpu_memory_utilization}")
    if bc.kv_cache_dtype is not None:
        parts.append(f"--kv-cache-dtype {bc.kv_cache_dtype}")
    parts.append(
        "--enable-prefix-caching" if bc.enable_prefix_caching else "--no-enable-prefix-caching"
    )
    if bc.safetensors_load_strategy is not None:
        parts.append(f"--safetensors-load-strategy {bc.safetensors_load_strategy}")
    if bc.kv_offloading_size is not None:
        parts.append(f"--kv-offloading-size {bc.kv_offloading_size}")
    if bc.kv_offloading_backend is not None:
        parts.append(f"--kv-offloading-backend {bc.kv_offloading_backend}")
    if bc.speculative_decoding is not None:
        spec = {
            "model": bc.speculative_decoding.draft_model,
            "num_speculative_tokens": bc.speculative_decoding.num_speculative_tokens,
            "draft_tensor_parallel_size": bc.speculative_decoding.draft_tensor_parallel_size,
        }
        parts.append(f"--speculative-config '{json.dumps(spec, sort_keys=True)}'")
    return " ".join(parts)


def total_gpus(deployment: Deployment) -> int:
    bc = deployment.backend_config
    return (
        bc.tensor_parallel_size * bc.pipeline_parallel_size
        * bc.data_parallel_size * bc.expert_parallel_size
    )


def precheck_scope(deployment: Deployment, glob: GlobalConfig) -> str:
    """Must match tools/system_prechecks_reference.yaml scope strings exactly."""
    cluster = glob.clusters[deployment.target]
    gpus = total_gpus(deployment)
    if cluster.type == "k8s":
        return f"{gpus}× {cluster.gpu_label}, 1 pod"
    nodes = math.ceil(gpus / cluster.gpus_per_node)
    return f"{gpus}× {cluster.gpu_label}, {nodes} node{'s' if nodes > 1 else ''}"


def default_image(deployment: Deployment, glob: GlobalConfig) -> str:
    if deployment.image:
        return deployment.image
    # Canonical tag scheme is finalised in M5; this default tracks §8.1's shape.
    registry_host = glob.registry.jfrog_base.removeprefix("https://").split("/artifactory")[0]
    return f"{registry_host}/ml/inference/{deployment.backend}:{deployment.backend_version}"


def render_experiment(
    cfg_path: Path,
    out_root: Path,
    glob: GlobalConfig | None = None,
    now: datetime | None = None,
) -> Path:
    """Renders the full experiment directory; returns its path."""
    glob = glob or load_global_config()
    cfg = load_benchmark_config(cfg_path, glob)
    now = now or datetime.now(timezone.utc)
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR), undefined=StrictUndefined,
        keep_trailing_newline=True,
    )

    exp_dir = out_root / f"{now:%Y-%m-%d}_{cfg.name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, exp_dir / "benchmark_config.yaml")

    for deployment_index, deployment in enumerate(cfg.deployments):
        cluster = glob.clusters[deployment.target]
        run_id = make_run_id(deployment.model, deployment.backend, deployment.target, now=now)
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        gpus = total_gpus(deployment)
        context = {
            "run_id": run_id,
            "run_id_slug": run_id_slug(run_id),
            "deployment_index": deployment_index,  # M7 selects cfg.deployments[index]
            "model_slug": model_slug(deployment.model),
            "cluster": deployment.target,
            "image": default_image(deployment, glob),
            # Alps-extended images bundle their own CXI/libfabric stack → disable
            # the host CXI hook so the image's libraries win (§8.1). Drives the EDF
            # annotation + the srun --network=disable_rdzv_get flag.
            "disable_cxi_hook": deployment.alps_extended_image,
            "engine_command": vllm_command(deployment),
            "total_gpus": gpus,
            "gpus_per_node": cluster.gpus_per_node,
            "nodes": math.ceil(gpus / cluster.gpus_per_node),
            "account": glob.slurm.account,
            "partition": cluster.partition,
            "namespace": cluster.namespace,
            "node_type": cluster.node_type,
            "time_limit": cfg.server_time_limit or "04:00:00",
            "scratch_base": glob.scratch_base,
            "run_dir_remote": f"{glob.scratch_base}/{run_id}",
            "tools_remote": f"{glob.scratch_base}/{run_id}/tools",
            "benchmarker_venv": f"{glob.scratch_base}/benchmarker-venv",  # decision 3 (shared)
            "collective_tests_cache_dir": glob.collective_tests_cache_dir,
            "prechecks": cfg.system_prechecks,
            "precheck_scope": precheck_scope(deployment, glob),
            "precheck_storage_scope": (
                "Ceph PVC weights mount" if cluster.type == "k8s"
                else "capstor weights mount (Lustre, HDD)"
            ),
            "hardware_sampling_interval_s": cfg.hardware_sampling_interval_s,
            "benchmarker_image": default_image(
                Deployment(
                    target=deployment.target, backend=deployment.backend,
                    backend_version="benchmarker", model="benchmarker/benchmarker",
                ),
                glob,
            ).rsplit(":", 1)[0].replace(f"/{deployment.backend}", "/benchmarker") + ":latest",
            "startup_failure_threshold": max(
                6, int((cfg.phases.server_ready_timeout_s) / 10)
            ),
        }
        if cluster.type == "slurm":
            _render(env, "vllm.edf.j2", context, run_dir / "engine.toml")
            _render(env, "engine.sbatch.j2", context, run_dir / "engine.sbatch")
            _render(env, "benchmarker.sbatch.j2", context, run_dir / "benchmarker.sbatch")
        else:
            _render(env, "k8s/engine.yaml.j2", context, run_dir / "engine.yaml")
            _render(env, "k8s/benchmarker-pod.yaml.j2", context, run_dir / "benchmarker-pod.yaml")
    return exp_dir


def _render(env: Environment, template: str, context: dict, dest: Path) -> None:
    dest.write_text(env.get_template(template).render(**context))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render an experiment directory (§4)")
    parser.add_argument("yaml_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    exp_dir = render_experiment(args.yaml_path, args.out)
    artifacts = sorted(str(p.relative_to(exp_dir)) for p in exp_dir.rglob("*") if p.is_file())
    print(f"OK: rendered {exp_dir} —")
    for a in artifacts:
        print(f"  {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
