"""Headless report executor (SPECIFICATIONS.md §14.2, M9).

Injects parameters into the template notebook, executes it against a results DB
with nbclient (papermill is not a dependency), and writes the executed notebook
plus its rendered PNGs into the experiment directory (§13.8).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell

from tools.common.config import REPO_ROOT

log = logging.getLogger("reports.render")
DEFAULT_TEMPLATE = REPO_ROOT / "experiments" / "template_report.ipynb"


def _inject_params(nb: nbformat.NotebookNode, params: dict) -> None:
    # repr() — Python literals (None/True/{}), NOT json.dumps (null/true) which the kernel
    # would choke on for run_id=None.
    src = "\n".join(f"{k} = {v!r}" for k, v in params.items())
    for cell in nb.cells:
        if cell.cell_type == "code" and "parameters" in cell.get("metadata", {}).get("tags", []):
            cell.source = src
            return
    nb.cells.insert(0, new_code_cell(src, metadata={"tags": ["injected-parameters"]}))


def render_report(
    template_path: Path | str,
    db_path: Path | str,
    run_id: str | None,
    out_dir: Path | str,
    sessions_per_user_per_hour: dict | None = None,
    *,
    timeout_s: int = 600,
    kernel_name: str = "python3",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(str(template_path), as_version=4)
    _inject_params(nb, {
        "db_path": str(db_path),
        "run_id": run_id,
        "out_dir": str(out_dir),
        "sessions_per_user_per_hour": sessions_per_user_per_hour or {},
    })
    # The executing kernel must import `tools.*`; give it the repo root absolutely
    # (a relative PYTHONPATH would resolve against the kernel's cwd = out_dir).
    os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    NotebookClient(
        nb, timeout=timeout_s, kernel_name=kernel_name,
        resources={"metadata": {"path": str(out_dir)}},
    ).execute()
    executed = out_dir / "report.ipynb"
    nbformat.write(nb, str(executed))
    log.info("rendered %s (PNGs in %s)", executed, out_dir)
    return executed


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the report notebook headless (§14.2)")
    parser.add_argument("--db", type=Path, required=True, help="results DB (per-run or central)")
    parser.add_argument("--run-id", default=None, help="run to report (required if the DB has many)")
    parser.add_argument("--out-dir", type=Path, required=True, help="experiment dir for report.ipynb + PNGs")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--sessions-per-user-per-hour", default="{}",
                        help='JSON object mapping scenario → sessions/user/hour')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    executed = render_report(
        args.template, args.db, args.run_id, args.out_dir,
        json.loads(args.sessions_per_user_per_hour),
    )
    print(f"OK: {executed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
