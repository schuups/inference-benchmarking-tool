"""Headless report execution (§14.1/§14.2).

    python -m tools.reports.run <run.db> --out <experiment-dir>

Injects parameters into the template's `parameters` cell (papermill-style),
executes with nbclient, writes report.ipynb + rendered PNGs into the
experiment directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

from .build_template import REPO_ROOT, TEMPLATE_PATH


def execute_report(
    db_path: Path,
    out_dir: Path,
    sessions_per_user_per_hour: dict[str, float] | None = None,
    template: Path = TEMPLATE_PATH,
) -> Path:
    nb = nbf.read(template, as_version=4)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell in nb.cells:
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            override = (
                f"DB_PATH = {json.dumps(str(db_path))}\n"
                f"OUT_DIR = {json.dumps(str(out_dir))}\n"
            )
            if sessions_per_user_per_hour is not None:
                override += (
                    f"SESSIONS_PER_USER_PER_HOUR = {json.dumps(sessions_per_user_per_hour)}\n"
                )
                cell.source = override
            else:  # keep the template's editable defaults, override paths only
                cell.source = cell.source + "\n" + override
            break
    client = NotebookClient(nb, timeout=600, resources={"metadata": {"path": str(REPO_ROOT)}})
    client.execute()
    out_path = out_dir / "report.ipynb"
    nbf.write(nb, out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the §14.1 report notebook")
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="experiment directory (§13.8)")
    args = parser.parse_args()
    out = execute_report(args.db_path.resolve(), args.out.resolve())
    print(f"OK: executed report at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
