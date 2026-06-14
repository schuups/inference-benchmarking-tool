"""M3 DoD: sampler /proc + nvidia-smi parsers (fixtures), laptop degradation, CLI."""

import json
import subprocess
import sys
from pathlib import Path

from tools.benchmarker.hw_sampler import (
    GPU_FIELDS,
    NODE_FIELDS,
    Sampler,
    cpu_pcts,
    parse_diskstats,
    parse_meminfo,
    parse_net_dev,
    parse_nvidia_smi_csv,
    parse_proc_stat,
)

SAMPLER_PATH = Path(__file__).resolve().parents[1] / "benchmarker" / "hw_sampler.py"


def test_sampler_is_stdlib_only():
    source = SAMPLER_PATH.read_text()
    body = source.split('"""', 2)[2]  # skip module docstring
    for forbidden in ("import aiohttp", "import pydantic", "import yaml", "from tools"):
        assert forbidden not in body


def test_parse_proc_stat_and_cpu_pcts():
    prev_text = "cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 50 0 50 350 50 0 0 0 0 0\n"
    cur_text = "cpu  200 0 200 1300 200 0 0 0 0 0\ncpu0 100 0 100 650 100 0 0 0 0 0\n"
    prev, cur = parse_proc_stat(prev_text), parse_proc_stat(cur_text)
    util, iowait = cpu_pcts(prev, cur)
    # deltas: busy 200, iowait 100, total 900 -> 22.2% util, 11.1% iowait
    assert round(util, 1) == 22.2
    assert round(iowait, 1) == 11.1
    assert cpu_pcts(None, cur) == (None, None)


def test_parse_meminfo():
    text = "MemTotal:       16384000 kB\nMemFree:        1000000 kB\nMemAvailable:    8192000 kB\n"
    used_gb, pct = parse_meminfo(text)
    assert round(used_gb, 2) == round(8192000 / 1024**2, 2)
    assert round(pct, 1) == 50.0
    assert parse_meminfo("Garbage: 1 kB\n") == (None, None)


def test_parse_diskstats_skips_partitions_and_virtual():
    text = (
        "   8  0 sda 100 0 2000 0 0 0 0 0 0 0 0\n"
        "   8  1 sda1 999 0 99999 0 0 0 0 0 0 0 0\n"      # partition: skipped
        "   7  0 loop0 999 0 99999 0 0 0 0 0 0 0 0\n"     # virtual: skipped
        " 259  0 nvme0n1 50 0 1000 0 0 0 0 0 0 0 0\n"
    )
    reads, sectors = parse_diskstats(text)
    assert reads == 150 and sectors == 3000


def test_parse_net_dev_skips_loopback():
    text = (
        "Inter-|   Receive                                                |  Transmit\n"
        " face |bytes    packets errs drop fifo frame compressed multicast|bytes ...\n"
        "    lo: 999999 0 0 0 0 0 0 0 999999 0 0 0 0 0 0 0\n"
        "  eth0: 1000 0 0 0 0 0 0 0 2000 0 0 0 0 0 0 0\n"
        "  hsn0: 3000 0 0 0 0 0 0 0 4000 0 0 0 0 0 0 0\n"
    )
    rx, tx = parse_net_dev(text)
    assert rx == 4000 and tx == 6000


def test_parse_nvidia_smi_csv():
    text = "95, 81920, 97871, 612.5, 64\n3, 1024, 97871, 88.0, 35\n"
    rows = parse_nvidia_smi_csv(text)
    assert len(rows) == 2
    assert rows[0]["gpu_util_pct"] == 95
    assert round(rows[0]["gpu_mem_used_gb"], 1) == 80.0
    assert round(rows[0]["gpu_mem_pct"], 1) == 83.7
    assert rows[0]["gpu_power_w"] == 612.5
    # "[N/A]" fields degrade to None
    assert parse_nvidia_smi_csv("[N/A], 100, 200, [N/A], 40\n")[0]["gpu_util_pct"] is None


def test_tick_degrades_gracefully_off_cluster():
    rows = Sampler().tick()
    assert rows, "at least the node-scoped row must be emitted"
    node_row = rows[0]
    assert node_row["gpu_index"] is None
    for field in (*GPU_FIELDS, *NODE_FIELDS):
        assert field in node_row  # every §14.5 signal present, possibly null
    json.dumps(rows)  # NDJSON-serializable


def test_cli_writes_ndjson(tmp_path):
    out = tmp_path / "hw.ndjson"
    proc = subprocess.run(
        [sys.executable, str(SAMPLER_PATH), "--out", str(out), "--interval", "0.1", "--duration", "0.35"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) >= 2
    assert all("ts" in row and "gpu_index" in row for row in lines)
