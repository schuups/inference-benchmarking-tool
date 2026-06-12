"""Benchmarker CLI — entry point invoked by the planner-rendered
benchmarker.sbatch / benchmarker pod (§4, M7).

    python3 -m tools.benchmarker.main --config benchmark_config.yaml \
        --run-id <id> --run-dir <dir> --engine-sbatch <path> [--deployment-index 0]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from tools.common.config import SCENARIOS_DIR, load_benchmark_config
from tools.benchmarker.dataset_gen.tokenizers import load_tokenizer
from tools.benchmarker.orchestrator import RunAborted, SlurmEngineLauncher, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmarker orchestrator (M7)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--engine-sbatch", type=Path, required=True)
    parser.add_argument("--deployment-index", type=int, default=0)
    parser.add_argument("--tokenizer", default=None, help="override tokenizer id ('word' = offline)")
    parser.add_argument("--registry", type=Path, default=SCENARIOS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_benchmark_config(args.config, registry_dir=args.registry)
    deployment = cfg.deployments[args.deployment_index]
    tokenizer = load_tokenizer(
        args.tokenizer or cfg.dataset_config.tokenizer_id or deployment.model
    )
    launcher = SlurmEngineLauncher(engine_sbatch=args.engine_sbatch)
    try:
        summary = asyncio.run(
            run_experiment(
                cfg, deployment, args.run_id, args.run_dir, tokenizer, launcher, args.registry
            )
        )
    except RunAborted as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 3
    print(
        f"OK: run {summary.run_id} — {summary.requests} requests, "
        f"{summary.sessions_started} sessions ({summary.sessions_truncated} truncated), "
        f"persisted={summary.persisted}"
        + (" [SMOKE-TEST MODE — NOT PERSISTED]" if summary.smoke_test_mode else "")
        + (f" [quality stages pending M11: {summary.quality_stages_pending}]"
           if summary.quality_stages_pending else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
