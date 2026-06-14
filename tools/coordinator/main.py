"""Coordinator CLI (IMPLEMENTATION_PLAN.md M8).

Subcommands:
  run    drive an experiment end-to-end. The Benchmarker is always a SLURM allocation and,
         per open decision 5, FirecREST is assistant-driven via MCP — so there is no
         autonomous loop here; the assistant performs the phases in-session using the
         tools.coordinator helpers, and this command reports that. (A K8s *engine* target is
         deployed by the Benchmarker itself; there is no K8s Coordinator path.)
  merge  headless: merge a downloaded per-run DB into the centralized results DB
         (usable on any platform — e.g. after the assistant downloads a SLURM DB
         via the FirecREST MCP staged transfer).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tools.common.config import REPO_ROOT, load_global_config
from tools.common.runid import parse_run_id

from .merge import merge_run_db
from .state import RunState

log = logging.getLogger("coordinator")
DEFAULT_CENTRAL_DB = REPO_ROOT / "experiments" / "results.db"


def _target_from_run_id(run_id: str) -> str:
    parts = parse_run_id(run_id)
    if parts is None:
        raise SystemExit(f"cannot parse target from run_id {run_id!r} (expected §7.2 format)")
    return parts.target


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

    # The Benchmarker is ALWAYS a SLURM allocation; per open decision 5 the SLURM/FirecREST
    # path is assistant-driven via MCP in-session, so there is no autonomous loop here.
    # (A K8s *engine* target is deployed by the Benchmarker itself, not by this command.)
    note = "" if state.platform == "slurm" else f" — note: engine target {state.target!r} is K8s (E5)"
    print(
        f"Run {run_id}: the Benchmarker is a SLURM allocation driven via FirecREST MCP "
        f"(open decision 5){note}. Drive the phases through Claude using the "
        "tools.coordinator helpers + FirecREST MCP; this command runs no autonomous loop.",
        file=sys.stderr,
    )
    return 2


def _cmd_merge(args) -> int:
    counts = merge_run_db(args.per_run_db, args.central_db, args.run_id)
    print(f"OK: merged {args.run_id} into {args.central_db}: {counts}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinator (M8)")
    parser.add_argument("--central-db", type=Path, default=DEFAULT_CENTRAL_DB)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="report how to drive the run (SLURM Benchmarker; assistant-driven)")
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
