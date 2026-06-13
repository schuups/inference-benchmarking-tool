"""Coordinator CLI (IMPLEMENTATION_PLAN.md M8).

Subcommands:
  run    drive an experiment end-to-end. Headless for K8s (kubectl). For SLURM,
         FirecREST is assistant-driven via MCP (open decision 5), so the autonomous
         loop is not used — the assistant performs the phases in-session using the
         tools.coordinator helpers; this command reports that.
  merge  headless: merge a downloaded per-run DB into the centralized results DB
         (usable on any platform — e.g. after the assistant downloads a SLURM DB
         via the FirecREST MCP staged transfer).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from tools.common.config import REPO_ROOT, load_global_config
from tools.common.runid import run_id_slug

from .backend import KubectlClusterBackend
from .coordinator import Coordinator
from .merge import merge_run_db
from .state import RunState

log = logging.getLogger("coordinator")
DEFAULT_CENTRAL_DB = REPO_ROOT / "experiments" / "results.db"


def _target_from_run_id(run_id: str) -> str:
    # run_id = <ts>_<model-slug>_<backend>_<target>_<hex>; slugs carry no underscores.
    parts = run_id.split("_")
    if len(parts) != 5:
        raise SystemExit(f"cannot parse target from run_id {run_id!r} (expected 5 _-fields)")
    return parts[3]


def _sole_run_id(exp_dir: Path) -> str:
    run_dirs = [d.name for d in exp_dir.iterdir() if d.is_dir()]
    if len(run_dirs) != 1:
        raise SystemExit(
            f"{exp_dir} has {len(run_dirs)} run dirs; pass --run-id to pick one ({sorted(run_dirs)})"
        )
    return run_dirs[0]


def _build_state(exp_dir: Path, run_id: str, glob) -> RunState:
    target = _target_from_run_id(run_id)
    cluster = glob.clusters.get(target)
    if cluster is None:
        raise SystemExit(f"unknown target {target!r} from run_id (known: {sorted(glob.clusters)})")
    return RunState(
        run_id=run_id,
        platform=cluster.type,
        target=target,
        run_dir_local=str(exp_dir / run_id),
        run_dir_remote=f"{glob.scratch_base}/{run_id}",
    )


def _cmd_run(args) -> int:
    glob = load_global_config()
    exp_dir = args.exp_dir
    run_id = args.run_id or _sole_run_id(exp_dir)
    run_dir_local = exp_dir / run_id

    if args.resume and RunState.exists(run_dir_local):
        state = RunState.load(run_dir_local)
        log.info("resuming run %s from phase %s", run_id, state.phase)
    else:
        state = _build_state(exp_dir, run_id, glob)
        state.save()

    if state.platform != "k8s":
        print(
            f"SLURM run {run_id}: FirecREST is assistant-driven via MCP (open decision 5) — "
            "drive the phases through Claude using the tools.coordinator helpers + FirecREST MCP "
            "tools. The autonomous loop here only covers the K8s (kubectl) path.",
            file=sys.stderr,
        )
        return 2

    cluster = glob.clusters[state.target]
    backend = KubectlClusterBackend(
        namespace=cluster.namespace, run_id_slug=run_id_slug(run_id), manifest_dir=run_dir_local
    )
    coordinator = Coordinator(
        state, backend, args.central_db, poll_interval_s=args.poll_interval,
    )
    final = asyncio.run(coordinator.run())
    print(f"OK: {run_id} → phase={final.phase}")
    return 0


def _cmd_merge(args) -> int:
    counts = merge_run_db(args.per_run_db, args.central_db, args.run_id)
    print(f"OK: merged {args.run_id} into {args.central_db}: {counts}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinator (M8)")
    parser.add_argument("--central-db", type=Path, default=DEFAULT_CENTRAL_DB)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="drive an experiment (headless for K8s)")
    run.add_argument("--exp-dir", type=Path, required=True)
    run.add_argument("--run-id", default=None)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--poll-interval", type=float, default=30.0)
    run.set_defaults(func=_cmd_run)

    merge = sub.add_parser("merge", help="merge a downloaded per-run DB into the central DB")
    merge.add_argument("--per-run-db", type=Path, required=True)
    merge.add_argument("--run-id", required=True)
    merge.set_defaults(func=_cmd_merge)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
