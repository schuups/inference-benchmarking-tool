"""Benchmarker orchestrator CLI entry (SPECIFICATIONS.md §1, IMPLEMENTATION_PLAN.md M7).

Invoked from the Planner-rendered benchmarker.sbatch / benchmarker-pod.yaml:

    python3 -m tools.benchmarker.main \
        --config <yaml> --run-id <id> --run-dir <dir> \
        [--engine-sbatch <path> | --engine-manifest <path> --k8s] \
        --deployment-index <n>

Builds the platform launcher (launchers.py) and drives run_experiment()
(orchestrator.py). The M11 quality evaluator is wired here once it lands; until
then the quality stages are skipped (the orchestrator logs each skip).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from tools.common.config import SCENARIOS_DIR, load_benchmark_config, load_global_config
from tools.common.runid import run_id_slug

from .dataset_gen.tokenizers import load_tokenizer
from .launchers import K8sEngineLauncher, SlurmEngineLauncher
from .orchestrator import RunAborted, run_experiment

log = logging.getLogger("benchmarker")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmarker orchestrator (M7)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine-sbatch", type=Path, help="SLURM engine job (SLURM path)")
    parser.add_argument("--engine-manifest", type=Path, help="K8s engine manifest (K8s path)")
    parser.add_argument("--k8s", action="store_true", help="Kubernetes deployment target")
    parser.add_argument(
        "--deployment-index", type=int, default=0,
        help="index into config.deployments this run corresponds to (§15 deployment sweep)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    glob = load_global_config()
    cfg = load_benchmark_config(args.config, glob)
    if not 0 <= args.deployment_index < len(cfg.deployments):
        print(
            f"--deployment-index {args.deployment_index} out of range "
            f"(config has {len(cfg.deployments)} deployment(s))",
            file=sys.stderr,
        )
        return 2
    deployment = cfg.deployments[args.deployment_index]

    tokenizer = load_tokenizer(cfg.dataset_config.tokenizer_id or deployment.model)

    if args.k8s:
        cluster = glob.clusters[deployment.target]
        launcher = K8sEngineLauncher(
            engine_manifest=args.engine_manifest or (args.run_dir / "engine.yaml"),
            namespace=cluster.namespace,
            run_id_slug=run_id_slug(args.run_id),
        )
    else:
        launcher = SlurmEngineLauncher(
            engine_sbatch=args.engine_sbatch or (args.run_dir / "engine.sbatch"),
            run_dir=args.run_dir,
            run_id=args.run_id,
        )

    try:
        summary = asyncio.run(
            run_experiment(
                cfg, deployment, args.run_id, args.run_dir, tokenizer, launcher, SCENARIOS_DIR,
                quality=None,  # M11 wires the real QualityEvaluator here
            )
        )
    except RunAborted as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 3
    print(f"OK: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
