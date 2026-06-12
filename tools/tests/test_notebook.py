"""M9 DoD: the template executes headless against the fixture DB and reports
the analytically known λ* and users numbers."""

from pathlib import Path

import nbformat as nbf
import pytest

from tools.reports.build_template import TEMPLATE_PATH, build
from tools.reports.run import execute_report
from tools.testing.fixtures import (
    EXPECTED_LAMBDA_STAR,
    EXPECTED_POPULATION,
    USER_RATES,
    build_fixture_db,
)


def test_template_builds_and_is_committed(tmp_path):
    nb = build()
    kinds = [c.cell_type for c in nb.cells]
    assert kinds.count("code") >= 9 and kinds.count("markdown") >= 8
    assert any("parameters" in c.metadata.get("tags", []) for c in nb.cells)
    # the committed template must match the builder (regenerate when editing)
    committed = nbf.read(TEMPLATE_PATH, as_version=4)
    assert [c.source for c in committed.cells] == [c.source for c in nb.cells]


@pytest.mark.slow
def test_notebook_executes_on_fixture_db(tmp_path):
    db = build_fixture_db(tmp_path / "run.db")
    db.close()
    out_dir = tmp_path / "exp"
    report = execute_report(tmp_path / "run.db", out_dir, USER_RATES)

    executed = nbf.read(report, as_version=4)
    text = "\n".join(
        "".join(out.get("text", "")) for cell in executed.cells
        for out in cell.get("outputs", [])
    )
    assert f"λ* = {EXPECTED_LAMBDA_STAR}" in text
    expected_chat = EXPECTED_POPULATION["chat-short-turns"]
    assert f"≈{expected_chat:,.0f} users" in text
    # §14.1: not_modelled items visually distinguished
    assert "! NOT MODELLED:" in text
    # §14.2 rendered plots land in the experiment dir
    for png in ("ttft.png", "itl.png", "ttft_per_class.png"):
        assert (out_dir / png).exists(), png
    # no cell errored
    for cell in executed.cells:
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", out.get("evalue")
