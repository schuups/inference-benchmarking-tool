"""M4 DoD (local half): parsers against fixtures, §8.3/§8.4 grading, gate policy.

Container execution on clariden is the cluster half, validated at E1.
NVSHMEM fixtures are provisional shapes — re-capture from the engine image at E1.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.benchmarker.prechecks.grade import (
    REFERENCE_PATH,
    build_rows,
    collect_measurements,
    grade,
    load_reference,
    metric_id,
    outcome_exit_code,
    parse_dd_output,
    parse_nccl_output,
    parse_nvshmem_output,
    size_label_to_bytes,
)

NCCL_FIXTURE = """\
# nThread 1 nGpus 1 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
    67108864      16777216     float     sum      -1    825.1   81.33  121.99      0    824.0   81.44  122.16      0
   134217728      33554432     float     sum      -1   1571.2   85.42  128.13      0   1569.8   85.50  128.25      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 95.21
"""

NVSHMEM_LATENCY_FIXTURE = """\
#  alltoall_latency
#  size(B)        latency(us)
       1024           8.12
     131072          19.46
"""

DD_FIXTURE = "4096+0 records in\n4096+0 records out\n4294967296 bytes (4.3 GB, 4.0 GiB) copied, 2.951 s, 1.5 GB/s\n"


def test_size_label_to_bytes():
    assert size_label_to_bytes("128 MiB") == 134217728
    assert size_label_to_bytes("128 KiB") == 131072
    assert size_label_to_bytes("1 MiB blocks") == 1048576
    assert size_label_to_bytes("nonsense") is None


def test_parse_nccl_output_busbw_at_target():
    assert parse_nccl_output(NCCL_FIXTURE, 134217728) == pytest.approx(128.13)
    assert parse_nccl_output(NCCL_FIXTURE, 67108864) == pytest.approx(121.99)
    assert parse_nccl_output(NCCL_FIXTURE, 999) is None
    assert parse_nccl_output("garbage\n", 134217728) is None


def test_parse_nvshmem_output():
    assert parse_nvshmem_output(NVSHMEM_LATENCY_FIXTURE, 131072) == pytest.approx(19.46)
    assert parse_nvshmem_output(NVSHMEM_LATENCY_FIXTURE, 7) is None


def test_parse_dd_output_units():
    assert parse_dd_output(DD_FIXTURE) == pytest.approx(1.5)
    assert parse_dd_output(DD_FIXTURE.replace("1.5 GB/s", "750 MB/s")) == pytest.approx(0.75)
    assert parse_dd_output("dd: error reading") is None


def test_grading_rules():
    # higher is better (tolerance -10): pass / warn / fail bands (§8.3-§8.4)
    assert grade(100.0, 100.0, -10) == "pass"
    assert grade(91.0, 100.0, -10) == "pass"
    assert grade(85.0, 100.0, -10) == "warn"
    assert grade(49.0, 100.0, -10) == "fail"      # < 50% of expected
    # lower is better (tolerance +20)
    assert grade(100.0, 100.0, +20) == "pass"
    assert grade(125.0, 100.0, +20) == "warn"
    assert grade(201.0, 100.0, +20) == "fail"     # > 2x expected
    # informational: no reference yet (§8.3 TBD)
    assert grade(100.0, None, None) == "pass"
    # benchmark errored
    assert grade(None, 100.0, -10) == "fail"


def test_reference_loads_real_yaml():
    refs = load_reference("clariden", REFERENCE_PATH)
    assert len(refs) == 12
    assert all(r["cluster"] == "clariden" for r in refs)
    # all clariden entries are still TBD placeholders -> informational grading
    rows = build_rows(
        [{"benchmark": "NCCL all_reduce", "scope": "4× GH200, 1 node", "size": "128 MiB", "measured": 128.1}],
        refs,
    )
    assert rows[0]["expected"] is None and rows[0]["status"] == "pass"


def test_outcome_exit_codes():
    rows_ok = [{"status": "pass"}]
    rows_warn = [{"status": "pass"}, {"status": "warn"}]
    rows_fail = [{"status": "warn"}, {"status": "fail"}]
    assert outcome_exit_code(rows_ok, "abort", "abort") == 0
    assert outcome_exit_code(rows_warn, "abort", "abort") == 3
    assert outcome_exit_code(rows_warn, "continue", "abort") == 0
    assert outcome_exit_code(rows_fail, "abort", "abort") == 4
    assert outcome_exit_code(rows_fail, "abort", "continue") == 3  # warn still aborts
    assert outcome_exit_code(rows_fail, "continue", "continue") == 0


def test_metric_id_slug():
    assert metric_id("NCCL all_reduce", "128 MiB") == "nccl_all_reduce_128_mib"
    assert metric_id("ROCm SHMEM alltoall_latency", "128 KiB") == "rocm_shmem_alltoall_latency_128_kib"


def _write_fixture_outputs(out_dir: Path):
    out_dir.mkdir()
    for c in ("all_reduce", "all_gather", "alltoall"):
        (out_dir / f"collective_{c}.out").write_text(NCCL_FIXTURE)
    (out_dir / "nvshmem_alltoall_latency.out").write_text(NVSHMEM_LATENCY_FIXTURE)
    (out_dir / "storage_read.out").write_text(DD_FIXTURE)


def test_collect_measurements_maps_files_to_reference_rows(tmp_path):
    out_dir = tmp_path / "out"
    _write_fixture_outputs(out_dir)
    refs = load_reference("clariden", REFERENCE_PATH)
    measurements = collect_measurements(
        out_dir, "clariden", "4× GH200, 1 node", "capstor weights mount (Lustre, HDD)", refs
    )
    by_benchmark = {m["benchmark"]: m for m in measurements}
    assert by_benchmark["NCCL all_reduce"]["measured"] == pytest.approx(128.13)
    assert by_benchmark["NVSHMEM alltoall_latency"]["measured"] == pytest.approx(19.46)
    assert by_benchmark["Sequential read"]["measured"] == pytest.approx(1.5)
    # nvshmem_put_bw.out absent -> skipped-with-warning, no row (§8.1)
    assert "NVSHMEM shmem_put_bw" not in by_benchmark


def test_nvshmem_skip_detected_by_content(tmp_path):
    """E1 attempt #4: the runner tees the §8.1 skip warning INTO the capture
    file, so a 'skipped' NVSHMEM must not grade as fail."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "nvshmem_alltoall_latency.out").write_text(
        "[nvshmem] NVSHMEM perftest not found in this engine image\n"
        "[nvshmem]   tried: /opt/nvshmem/bin/perftest ...\n"
    )
    refs = load_reference("clariden", REFERENCE_PATH)
    measurements = collect_measurements(out_dir, "clariden", "4× GH200, 1 node", "", refs)
    assert not any("NVSHMEM" in m["benchmark"] for m in measurements)


def test_grade_cli_end_to_end(tmp_path):
    out_dir = tmp_path / "out"
    _write_fixture_outputs(out_dir)
    results = tmp_path / "results.json"
    proc = subprocess.run(
        [
            sys.executable, "tools/benchmarker/prechecks/grade.py",
            "--out-dir", str(out_dir), "--cluster", "clariden",
            "--scope", "4× GH200, 1 node",
            "--storage-scope", "capstor weights mount (Lustre, HDD)",
            "--results", str(results), "--smoke",
        ],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
    )
    assert proc.returncode == 0, proc.stderr  # all TBD references -> informational pass
    payload = json.loads(results.read_text())
    assert payload["smoke_test_mode"] is True
    assert len(payload["rows"]) == 5
    assert "SMOKE-TEST MODE" in proc.stdout


def test_runner_script_syntax():
    for script in Path("tools/benchmarker/prechecks").glob("*.sh"):
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{script}: {proc.stderr}"
