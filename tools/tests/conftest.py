"""Shared pytest fixtures for the tools test suite.

`globals_cfg`, `canonical_dict`, and the canonical-config path were previously
copy-pasted into test_config.py and test_planner.py; they live here once so
pytest auto-discovers them for every test module.
"""

from pathlib import Path

import pytest
import yaml

from tools.common.config import load_global_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = REPO_ROOT / "examples" / "benchmark-configs" / "mixed-80-20.yaml"


@pytest.fixture(scope="module")
def globals_cfg():
    return load_global_config()


@pytest.fixture()
def canonical_dict():
    with open(CANONICAL) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def canonical_path():
    return CANONICAL
